package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.cli.Args.UsageException;
import org.liberationdata.kiln.cli.RepoIndex.RepoFile;
import org.liberationdata.kiln.cli.RepoIndex.RepoInfo;
import org.liberationdata.kiln.hub.KilnHub;
import org.liberationdata.kiln.hub.ModelCatalog;
import org.liberationdata.kiln.hub.ModelSet;
import org.liberationdata.kiln.hub.ModelSet.Kind;
import org.liberationdata.kiln.hub.ModelSet.Part;
import org.liberationdata.kiln.hub.ModelSet.Source;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;

/**
 * {@code kiln pull}: an alias from {@link ModelCatalog}, or files picked from a repo, downloaded by {@link KilnHub}
 * into the Hugging Face cache. A repo's files become an ad hoc {@link ModelSet}, one part per file.
 */
final class Pull {
    static final java.util.Set<String> OPTIONS = java.util.Set.of("revision");

    static final String USAGE = """
            usage: kiln pull <alias> | <org/name>[:TAG] [file|glob...] [--revision REV]

              kiln pull qwen3-4b                                  an alias: every file it needs
              kiln pull unsloth/Qwen3-4B-Instruct-2507-GGUF       list the repo's GGUF models (and ask)
              kiln pull unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
              kiln pull Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf README.md
              kiln pull some/repo '*.json'                        globs match paths in the repo

            Files go to the Hugging Face cache (HF_HUB_CACHE, HF_HOME/hub or ~/.cache/huggingface/hub), the one
            `hf download` uses, so each sees the other's files. Files already there (or in KILN_MODEL_DIRS) are
            used as they are; an interrupted download resumes. Gated repos need their terms accepted on
            huggingface.co and HF_TOKEN (or `hf auth login`).""";

    private final KilnHub hub;
    private final RepoIndex index;
    private final PrintStream out, err;
    private final BufferedReader in;
    private final boolean interactive;

    Pull(KilnHub hub, RepoIndex index, PrintStream out, PrintStream err, BufferedReader in, boolean interactive) {
        this.hub = hub;
        this.index = index;
        this.out = out;
        this.err = err;
        this.in = in;
        this.interactive = interactive;
    }

    int run(Args a) throws IOException {
        String target = a.positional(0);
        if (target == null) throw new UsageException("which model? Give an alias or a Hugging Face repo");
        String revision = a.option("revision", "main");
        var alias = ModelCatalog.byName(target);
        if (alias.isPresent()) {
            if (!a.rest(1).isEmpty()) throw new UsageException(target + " is an alias; it names its own files");
            if (a.options().containsKey("revision")) throw new UsageException(target + " is an alias; it pins its own revisions");
            ModelSet m = alias.get();
            out.println(m.id() + ": " + m.title() + " (" + m.engine() + ")");
            if (m.license() != null) out.println("  License: " + m.license());
            if (m.gated()) out.println("  Gated: accept the terms on huggingface.co/" + m.parts().get(0).primary().repo()
                    + " and set HF_TOKEN (or run `hf auth login`) first.");
            pull(m);
            out.println(tryIt(m));
            return 0;
        }
        Matcher rm = Models.REPO.matcher(target);
        if (!rm.matches()) throw new UsageException("not an alias or an org/name repo: " + target + "\n  aliases: " + Models.aliases());
        String repo = rm.group(1), tag = rm.group(2);
        List<String> patterns = a.rest(1);
        if (tag != null && !patterns.isEmpty()) throw new UsageException("give either " + repo + ":TAG or file names, not both");
        RepoInfo info = index.info(repo, revision);
        List<RepoFile> files = select(info, tag, patterns);
        if (files.isEmpty()) {
            files = choose(info);
            if (files.isEmpty()) return 0;
        }
        pull(adHoc(target, repo, revision, files));
        out.println(repo + " @ " + info.commit().substring(0, Math.min(12, info.commit().length())) + " in " + hub.cacheRoot());
        var models = Gguf.models(files.stream().map(RepoFile::path).toList());
        if (models.size() == 1) {
            String file = models.keySet().iterator().next();
            String name = ModelCatalog.all().stream()
                    .filter(m -> m.kind() == Kind.CHAT && m.parts().get(0).uses(repo, file))
                    .map(ModelSet::id).findFirst().orElse(repo + ":" + file);
            out.println("Try it: kiln chat " + name);
        }
        return 0;
    }

    private void pull(ModelSet set) throws IOException {
        hub.pull(set, new PullLine(err, interactive));
    }

    static String tryIt(ModelSet m) {
        return switch (m.kind()) {
            case CHAT, VISION -> "Try it: kiln chat " + m.id();
            case IMAGE, SONG -> "Try it: kiln test " + m.id();
            default -> "Load it from Java with kiln-" + switch (m.kind()) {
                case SPEECH -> "tts";
                case TRANSCRIPTION -> "whisper";
                default -> "sd";
            } + " (" + m.engine() + "); the kiln command doesn't run " + m.kind().name().toLowerCase(java.util.Locale.ROOT) + " models yet.";
        };
    }

    /** Files picked from a repo, as a set KilnHub can pull: one part per file, its role the file's path. */
    static ModelSet adHoc(String name, String repo, String revision, List<RepoFile> files) {
        List<Part> parts = files.stream().map(f -> new Part(f.path(), List.of(new Source(repo, f.path(), revision)))).toList();
        return new ModelSet(name, Kind.CHAT, name, "files from " + repo, parts, null, false, List.of());
    }

    /**
     * The files that {@code tag} or {@code patterns} name (empty when both are absent). A name matches a path or a
     * unique file name; a pattern with {@code * ? [} or {@code {} is a glob over paths. Naming any part of a split
     * GGUF selects all of its parts.
     */
    static List<RepoFile> select(RepoInfo info, String tag, List<String> patterns) {
        List<String> paths = info.files().stream().map(RepoFile::path).toList();
        Map<String, List<String>> models = Gguf.models(paths);
        LinkedHashSet<String> chosen = new LinkedHashSet<>();
        if (tag != null) {
            List<String> hits = Gguf.match(models, tag);
            if (hits.isEmpty()) throw new Main.Failure("No GGUF model matching " + tag + " in " + info.repo()
                    + (models.isEmpty() ? " (it has no GGUF files)." : ". It has:\n  " + String.join("\n  ", models.keySet())));
            if (hits.size() > 1) throw new Main.Failure(tag + " matches several models in " + info.repo() + "; name one:\n  "
                    + String.join("\n  ", hits.stream().map(h -> info.repo() + ":" + h).toList()));
            chosen.addAll(models.get(hits.get(0)));
        }
        for (String p : patterns) {
            List<String> hits;
            if (p.matches(".*[*?\\[{].*")) {
                var glob = FileSystems.getDefault().getPathMatcher("glob:" + p);
                hits = paths.stream().filter(x -> glob.matches(Path.of(x))).toList();
            } else if (paths.contains(p)) {
                hits = List.of(p);
            } else {
                hits = paths.stream().filter(x -> Gguf.name(x).equals(p)).toList();
                if (hits.size() > 1) throw new Main.Failure(p + " is ambiguous in " + info.repo() + ":\n  " + String.join("\n  ", hits));
            }
            if (hits.isEmpty()) throw new Main.Failure("No file " + p + " in " + info.repo() + " at " + info.commit()
                    + (models.isEmpty() ? "" : ". GGUF models there:\n  " + String.join("\n  ", models.keySet())));
            for (String h : hits) {
                var split = models.values().stream().filter(parts -> parts.size() > 1 && parts.contains(h)).findFirst();
                if (split.isPresent()) chosen.addAll(split.get()); else chosen.add(h);
            }
        }
        return chosen.stream().map(x -> info.file(x).orElseThrow()).toList();
    }

    /** No files named: list the repo's GGUF models and, on a terminal, ask which to fetch. */
    private List<RepoFile> choose(RepoInfo info) throws IOException {
        Map<String, List<String>> models = Gguf.models(info.files().stream().map(RepoFile::path).toList());
        if (models.isEmpty()) {
            out.println(info.repo() + " has no GGUF files. Name the files to download. It has:");
            info.files().stream().limit(60).forEach(f -> out.printf("  %-60s %10s%n", f.path(), Fmt.bytes(f.size())));
            if (info.files().size() > 60) out.println("  ... and " + (info.files().size() - 60) + " more");
            out.println("e.g. kiln pull " + info.repo() + " " + (info.files().isEmpty() ? "FILE" : info.files().get(0).path()));
            return List.of();
        }
        List<String> keys = List.copyOf(models.keySet());
        out.println(info.repo() + " has " + keys.size() + " GGUF model" + (keys.size() == 1 ? "" : "s") + ":");
        for (int i = 0; i < keys.size(); i++) {
            long size = models.get(keys.get(i)).stream().mapToLong(p -> info.file(p).orElseThrow().size()).sum();
            int parts = models.get(keys.get(i)).size();
            out.printf("  %3d  %-60s %10s%s%n", i + 1, keys.get(i), Fmt.bytes(size), parts > 1 ? "  (" + parts + " parts)" : "");
        }
        if (!interactive) {
            String example = keys.stream().filter(k -> k.contains("Q4_K_M")).findFirst().orElse(keys.get(0));
            out.println("Pick one, e.g.: kiln pull " + info.repo() + ":" + example);
            return List.of();
        }
        out.print("Download which? Numbers separated by spaces, or Enter to cancel: ");
        out.flush();
        String line = in.readLine();
        List<RepoFile> picked = new ArrayList<>();
        if (line == null) return picked;
        for (String tok : line.trim().split("[\\s,]+")) {
            if (tok.isEmpty()) continue;
            int n;
            try { n = Integer.parseInt(tok); } catch (NumberFormatException e) { n = -1; }
            if (n < 1 || n > keys.size()) throw new UsageException("not a number from the list: " + tok);
            for (String p : models.get(keys.get(n - 1))) picked.add(info.file(p).orElseThrow());
        }
        return picked;
    }
}
