package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.hub.GgufFiles;
import org.liberationdata.fornax.cli.Args.UsageException;
import org.liberationdata.fornax.hub.RepoIndex;
import org.liberationdata.fornax.hub.RepoIndex.RepoFile;
import org.liberationdata.fornax.hub.RepoIndex.RepoInfo;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.HubException;
import org.liberationdata.fornax.hub.ModelCatalog;
import org.liberationdata.fornax.hub.ModelRef;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelSet.Part;
import org.liberationdata.fornax.hub.ModelSet.Source;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * {@code fornax pull}: an alias from {@link ModelCatalog}, or files picked from a repo, downloaded by {@link FornaxHub}
 * into the Hugging Face cache. A repo's files become an ad hoc {@link ModelSet}, one part per file.
 */
final class Pull {
    static final java.util.Set<String> OPTIONS = java.util.Set.of("revision");

    static final String USAGE = """
            usage: fornax pull <alias> | <org/name>[@REV][:TAG] [file|glob...] [--revision REV]

              fornax pull qwen3-4b                                  an alias: every file it needs
              fornax pull unsloth/Qwen3-4B-Instruct-2507-GGUF       list the repo's GGUF models (and ask)
              fornax pull unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
              fornax pull Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf README.md
              fornax pull some/repo '*.json'                        globs match paths in the repo

            Files go to the Hugging Face cache (HF_HUB_CACHE, HF_HOME/hub or ~/.cache/huggingface/hub), the one
            `hf download` uses, so each sees the other's files. Files already there (or in FORNAX_MODEL_DIRS) are
            used as they are; an interrupted download resumes. Gated repos need their terms accepted on
            huggingface.co and HF_TOKEN (or `hf auth login`).""";

    private final FornaxHub hub;
    private final RepoIndex index;
    private final PrintStream out, err;
    private final BufferedReader in;
    private final boolean interactive;

    Pull(FornaxHub hub, RepoIndex index, PrintStream out, PrintStream err, BufferedReader in, boolean interactive) {
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
            notes(m, out);
            pull(m);
            out.println(tryIt(m));
            return 0;
        }
        ModelRef ref = ModelRef.tryParse(target).orElseThrow(() -> new UsageException(ModelRef.urlHint(target)
                .map(r -> "a URL isn't a model reference: use fornax pull " + r)
                .orElse("not an alias or an org/name repo: " + target + "\n  aliases: " + Models.aliases())));
        if (ref.revision() != null && a.options().containsKey("revision"))
            throw new UsageException("give the revision once: " + ref.repo() + "@REV or --revision REV");
        if (ref.revision() != null) revision = ref.revision();
        String repo = ref.repo(), tag = ref.tag();
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
        var models = GgufFiles.models(files.stream().map(RepoFile::path).toList());
        if (models.size() == 1) {
            String file = models.keySet().iterator().next();
            String name = ModelCatalog.all().stream()
                    .filter(m -> m.kind() == Kind.CHAT && m.parts().get(0).uses(repo, file))
                    .map(ModelSet::id).findFirst().orElse(repo + ":" + file);
            out.println("Try it: fornax chat " + name);
        }
        return 0;
    }

    private void pull(ModelSet set) throws IOException {
        hub.pull(set, new PullLine(err, interactive));
    }

    /** What to know before pulling a set: its license, where it limits use, and whether its repo is gated. */
    static void notes(ModelSet m, PrintStream out) {
        if (m.license() != null) out.println("  License: " + m.license());
        if (m.gated()) out.println("  Gated: accept the terms on huggingface.co/" + m.parts().get(0).primary().repo()
                + " and set HF_TOKEN (or run `hf auth login`) first.");
    }

    static String tryIt(ModelSet m) {
        return switch (m.kind()) {
            case CHAT -> "Try it: fornax chat " + m.id();
            case VISION -> "Try it: fornax see " + m.id() + " picture.png";
            case SPEECH -> "Try it: fornax speak --model " + m.id() + " \"Hello from Fornax\"";
            case TRANSCRIPTION -> "Try it: fornax transcribe --model " + m.id() + " speech.wav";
            case IMAGE -> "Try it: fornax image " + m.id() + (Image.needsRef(m) ? " --ref picture.png \"make it night\"" : " \"a brick kiln at dusk\"");
            case SONG -> "Try it: fornax song " + m.id() + " \"acoustic folk, warm male vocal\"";
            case VIDEO -> "Load it from Java with fornax-sd (" + m.engine() + "); the fornax command doesn't make video yet.";
        };
    }

    /** Files picked from a repo, as a set FornaxHub can pull: one part per file, its role the file's path. */
    static ModelSet adHoc(String name, String repo, String revision, List<RepoFile> files) {
        List<Part> parts = files.stream().map(f -> new Part(f.path(), List.of(new Source(repo, f.path(), revision)))).toList();
        return new ModelSet(name, Kind.CHAT, name, "files from " + repo, parts, null, false, List.of());
    }

    /**
     * The files that {@code tag} ({@link ModelRef#pick}) or {@code patterns} name (empty when both are absent). A name matches a path or a
     * unique file name; a pattern with {@code * ? [} or {@code {} is a glob over paths. Naming any part of a split
     * GGUF selects all of its parts.
     */
    static List<RepoFile> select(RepoInfo info, String tag, List<String> patterns) {
        List<String> paths = info.files().stream().map(RepoFile::path).toList();
        Map<String, List<String>> models = GgufFiles.models(paths);
        LinkedHashSet<String> chosen = new LinkedHashSet<>();
        if (tag != null) {   // as FornaxHub.fetch picks it, so the reference names the same files in code and here
            try { chosen.addAll(new ModelRef(info.repo(), null, tag).pick(paths)); }
            catch (HubException | IllegalArgumentException e) { throw new Main.Failure(e.getMessage()); }
        }
        for (String p : patterns) {
            List<String> hits;
            if (p.matches(".*[*?\\[{].*")) {
                var glob = FileSystems.getDefault().getPathMatcher("glob:" + p);
                hits = paths.stream().filter(x -> glob.matches(Path.of(x))).toList();
            } else if (paths.contains(p)) {
                hits = List.of(p);
            } else {
                hits = paths.stream().filter(x -> GgufFiles.name(x).equals(p)).toList();
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
        Map<String, List<String>> models = GgufFiles.models(info.files().stream().map(RepoFile::path).toList());
        if (models.isEmpty()) {
            out.println(info.repo() + " has no GGUF files. Name the files to download. It has:");
            info.files().stream().limit(60).forEach(f -> out.printf("  %-60s %10s%n", f.path(), Fmt.bytes(f.size())));
            if (info.files().size() > 60) out.println("  ... and " + (info.files().size() - 60) + " more");
            out.println("e.g. fornax pull " + info.repo() + " " + (info.files().isEmpty() ? "FILE" : info.files().get(0).path()));
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
            out.println("Pick one, e.g.: fornax pull " + info.repo() + ":" + example);
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
