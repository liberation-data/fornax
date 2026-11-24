package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.cli.ModelSets.Kind;
import org.liberationdata.kiln.cli.ModelSets.ModelSet;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Turns a {@code <model>} argument into something to load. Accepted, in this order:
 * <ul>
 *   <li>a path to a .gguf file</li>
 *   <li>an alias from {@link ModelSets} ({@code qwen3-4b}, {@code ace-step}, ...)</li>
 *   <li>{@code org/name:file.gguf} or {@code org/name:TAG} (a quant such as Q4_K_M) from the cache</li>
 *   <li>{@code org/name} when the cache holds exactly one GGUF model from it</li>
 * </ul>
 */
final class Models {
    private Models() {}

    sealed interface Target permits Llm, Set {}

    /** A GGUF language model. {@code name} is what the user typed, for messages. */
    record Llm(String name, Path path) implements Target {}

    /** An image or song alias (see {@link ModelSets}). */
    record Set(ModelSet set) implements Target {}

    /** "org/name" or "org/name:tag". */
    static final Pattern REPO = Pattern.compile("([\\w.-]+/[\\w.-]+)(?::(.+))?");

    static Target resolve(String spec, HubCache cache) {
        Path p = Path.of(spec);
        if (Files.isRegularFile(p)) return new Llm(spec, p);
        var alias = ModelSets.byName(spec);
        if (alias.isPresent()) {
            ModelSet m = alias.get();
            if (m.kind() != Kind.LLM) return new Set(m);
            return new Llm(spec, m.components().get(0).locate(cache).orElseThrow(() -> new Main.Failure(
                    spec + " isn't downloaded yet (" + m.description() + "). Fetch it with: kiln pull " + spec)));
        }
        Matcher rm = REPO.matcher(spec);
        if (rm.matches()) return new Llm(spec, fromRepo(rm.group(1), rm.group(2), cache));
        throw new Main.Failure("No such file, alias or repo: " + spec + "\n  aliases: "
                + ModelSets.ALL.stream().map(ModelSet::name).collect(Collectors.joining(", ")));
    }

    private static Path fromRepo(String repo, String tag, HubCache cache) {
        Map<String, HubCache.CachedFile> byPath = cache.files(repo).stream()
                .collect(Collectors.toMap(HubCache.CachedFile::path, f -> f));
        Map<String, List<String>> models = Gguf.models(byPath.keySet());
        List<String> hits = tag == null ? List.copyOf(models.keySet()) : Gguf.match(models, tag);
        String want = repo + (tag == null ? "" : ":" + tag);
        if (hits.size() == 1) {
            List<String> parts = models.get(hits.get(0));
            List<String> missing = parts.stream().filter(x -> !byPath.containsKey(x)).toList();
            if (!missing.isEmpty()) throw new Main.Failure(want + " is incomplete in the cache (missing " + missing
                    + "). Fetch it with: kiln pull " + repo + ":" + hits.get(0));
            return byPath.get(hits.get(0)).location();
        }
        if (hits.isEmpty()) {
            String have = models.isEmpty() ? "" : "\n  in the cache from " + repo + ": " + String.join(", ", models.keySet());
            throw new Main.Failure("No GGUF model " + want + " in the cache (" + cache.root() + ")." + have
                    + "\n  fetch one with: kiln pull " + want);
        }
        throw new Main.Failure(want + " matches several models; name one:\n" + hits.stream()
                .map(h -> "  " + repo + ":" + h).collect(Collectors.joining("\n")));
    }
}
