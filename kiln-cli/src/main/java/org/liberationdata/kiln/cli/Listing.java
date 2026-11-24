package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.cli.ModelSets.Kind;
import org.liberationdata.kiln.cli.ModelSets.ModelSet;

import java.io.PrintStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

/** {@code kiln list}: what in the cache Kiln can run. */
final class Listing {
    static final String USAGE = """
            usage: kiln list

            Shows the GGUF language models in the Hugging Face cache, and which of the image and song
            aliases are downloaded, with sizes.""";

    /** A runnable GGUF language model in the cache. */
    record LlmEntry(String repo, String file, Path path, long size, int parts, String alias) {
        String spec() { return alias != null ? alias : repo + ":" + file; }
    }

    /** An image or song alias: what's there and what isn't. */
    record SetEntry(ModelSet set, long size, List<String> missing) {}

    static List<LlmEntry> llms(HubCache cache) {
        List<LlmEntry> out = new ArrayList<>();
        for (String repo : cache.repos()) {
            if (ModelSets.isEngineRepo(repo)) continue;
            var files = cache.files(repo).stream().collect(Collectors.toMap(HubCache.CachedFile::path, f -> f));
            for (var e : Gguf.models(files.keySet()).entrySet()) {
                if (!files.keySet().containsAll(e.getValue())) continue;   // an incomplete split model
                long size = e.getValue().stream().mapToLong(p -> files.get(p).size()).sum();
                String alias = ModelSets.llmFor(repo, e.getKey()).map(ModelSet::name).orElse(null);
                out.add(new LlmEntry(repo, e.getKey(), files.get(e.getKey()).location(), size, e.getValue().size(), alias));
            }
        }
        return out;
    }

    /** Image and song aliases with at least one file in the cache. */
    static List<SetEntry> sets(HubCache cache) {
        List<SetEntry> out = new ArrayList<>();
        for (ModelSet m : ModelSets.ALL) {
            if (m.kind() == Kind.LLM) continue;
            long size = 0;
            List<String> missing = new ArrayList<>();
            for (var c : m.components()) {
                Optional<Path> p = c.locate(cache);
                if (p.isPresent()) size += p.get().toFile().length();
                else missing.add(c.primary().fileName());
            }
            if (missing.size() < m.components().size()) out.add(new SetEntry(m, size, missing));
        }
        return out;
    }

    static int run(HubCache cache, PrintStream out) {
        out.println("Cache: " + cache.root());
        List<LlmEntry> llms = llms(cache);
        List<SetEntry> sets = sets(cache);
        Set<String> present = new HashSet<>();
        out.println();
        out.println("Language models (kiln chat <model>):");
        if (llms.isEmpty()) out.println("  none yet: try kiln pull qwen3-0.6b");
        int w = llms.stream().mapToInt(e -> e.spec().length()).max().orElse(0);
        for (LlmEntry e : llms) {
            if (e.alias() != null) present.add(e.alias());
            out.printf("  %-" + w + "s  %9s%s%s%n", e.spec(), Fmt.bytes(e.size()), e.parts() > 1 ? "  (" + e.parts() + " parts)" : "",
                    e.alias() != null ? "  " + e.repo() + ":" + e.file() : "");
        }
        for (Kind kind : List.of(Kind.IMAGE, Kind.SONG)) {
            List<SetEntry> of = sets.stream().filter(s -> s.set().kind() == kind).toList();
            if (of.isEmpty()) continue;
            out.println();
            out.println(kind == Kind.IMAGE ? "Image models (kiln test <model>):" : "Song models (kiln test <model>):");
            for (SetEntry s : of) {
                present.add(s.set().name());
                String state = s.missing().isEmpty() ? "" : "  incomplete, missing " + String.join(", ", s.missing())
                        + " (kiln pull " + s.set().name() + ")";
                out.printf("  %-16s %9s  %s%s%n", s.set().name(), Fmt.bytes(s.size()), s.set().description(), state);
            }
        }
        List<String> others = ModelSets.ALL.stream().map(ModelSet::name).filter(n -> !present.contains(n)).toList();
        if (!others.isEmpty()) {
            out.println();
            out.println("Not downloaded: " + String.join(", ", others) + "  (kiln pull <alias>)");
        }
        return 0;
    }
}
