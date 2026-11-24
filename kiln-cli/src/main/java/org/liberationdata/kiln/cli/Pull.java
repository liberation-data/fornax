package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.cli.Args.UsageException;
import org.liberationdata.kiln.cli.Hub.RepoFile;
import org.liberationdata.kiln.cli.Hub.RepoInfo;
import org.liberationdata.kiln.cli.ModelSets.Kind;
import org.liberationdata.kiln.cli.ModelSets.ModelSet;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;

/** {@code kiln pull}: downloads into the Hugging Face cache, in the layout {@code hf download} uses. */
final class Pull {
    static final java.util.Set<String> OPTIONS = java.util.Set.of("revision");

    static final String USAGE = """
            usage: kiln pull <alias> | <org/name>[:TAG] [file|glob...] [--revision REV]

              kiln pull qwen3-4b                                  an alias: every file it needs
              kiln pull unsloth/Qwen3-4B-Instruct-2507-GGUF       list the repo's GGUF models (and ask)
              kiln pull unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
              kiln pull Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf README.md
              kiln pull some/repo '*.json'                        globs match paths in the repo

            Files go to the Hugging Face cache (HF_HUB_CACHE, HF_HOME/hub or ~/.cache/huggingface/hub), so
            `hf` and Kiln's HfCache see them too. Complete files are skipped; interrupted ones resume.
            Gated or private repos need HF_TOKEN (or `hf auth login`).""";

    private final Hub hub;
    private final HubCache cache;
    private final PrintStream out, err;
    private final BufferedReader in;
    private final boolean interactive;

    Pull(Hub hub, HubCache cache, PrintStream out, PrintStream err, BufferedReader in, boolean interactive) {
        this.hub = hub;
        this.cache = cache;
        this.out = out;
        this.err = err;
        this.in = in;
        this.interactive = interactive;
    }

    int run(Args a) throws IOException {
        String target = a.positional(0);
        if (target == null) throw new UsageException("which model? Give an alias or a Hugging Face repo");
        String revision = a.option("revision", "main");
        var alias = ModelSets.byName(target);
        if (alias.isPresent()) {
            if (!a.rest(1).isEmpty()) throw new UsageException(target + " is an alias; it names its own files");
            ModelSet m = alias.get();
            out.println(m.name() + ": " + m.description());
            for (var c : m.components()) {
                var have = c.locate(cache);
                if (have.isPresent() && cache.find(c.primary().repo(), c.primary().file()).isEmpty())
                    out.println("  " + c.primary().file() + "  (already in the cache as " + have.get() + ")");
            }
            for (var e : aliasFiles(m, cache).entrySet()) {
                RepoInfo info = hub.info(e.getKey(), revision);
                fetch(info, revision, select(info, null, e.getValue()));
            }
            out.println(m.kind() == Kind.LLM ? "Try it: kiln chat " + m.name() : "Try it: kiln test " + m.name());
            return 0;
        }
        Matcher rm = Models.REPO.matcher(target);
        if (!rm.matches()) throw new UsageException("not an alias or an org/name repo: " + target);
        String repo = rm.group(1), tag = rm.group(2);
        List<String> patterns = a.rest(1);
        if (tag != null && !patterns.isEmpty()) throw new UsageException("give either " + repo + ":TAG or file names, not both");
        RepoInfo info = hub.info(repo, revision);
        List<RepoFile> files = select(info, tag, patterns);
        if (files.isEmpty()) {
            files = choose(info);
            if (files.isEmpty()) return 0;
        }
        fetch(info, revision, files);
        var models = Gguf.models(files.stream().map(RepoFile::path).toList());
        if (models.size() == 1) {
            String file = models.keySet().iterator().next();
            out.println("Try it: kiln chat " + ModelSets.llmFor(repo, file).map(ModelSet::name).orElse(repo + ":" + file));
        }
        return 0;
    }

    /**
     * An alias's files to fetch, grouped by repo: each component's first source, except components already
     * in the cache from another of their sources.
     */
    static Map<String, List<String>> aliasFiles(ModelSet m, HubCache cache) {
        Map<String, List<String>> byRepo = new LinkedHashMap<>();
        for (var c : m.components()) {
            var s = c.primary();
            if (c.locate(cache).isPresent() && cache.find(s.repo(), s.file()).isEmpty()) continue;
            List<String> files = byRepo.computeIfAbsent(s.repo(), k -> new ArrayList<>());
            if (!files.contains(s.file())) files.add(s.file());
        }
        return byRepo;
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

    /** Downloads what's missing, links the snapshot and records the ref. */
    void fetch(RepoInfo info, String revision, List<RepoFile> files) throws IOException {
        String repo = info.repo(), commit = info.commit();
        List<RepoFile> todo = new ArrayList<>();
        long need = 0;
        for (RepoFile f : files) {
            Path snap = cache.snapshotFile(repo, commit, f.path());
            if (complete(snap, f)) { out.println("  " + f.path() + "  " + Fmt.bytes(f.size()) + "  (already downloaded)"); continue; }
            todo.add(f);
            Path blob = cache.blob(repo, f.blobId()), part = partOf(blob);
            if (!complete(blob, f)) need += Math.max(0, f.size() - (Files.exists(part) ? Files.size(part) : 0));
        }
        if (need > 0) {
            Files.createDirectories(cache.root());
            long free = Files.getFileStore(cache.root()).getUsableSpace();
            if (free < need) throw new Main.Failure("Not enough disk space in " + cache.root() + ": need "
                    + Fmt.bytes(need) + ", " + Fmt.bytes(free) + " free.");
        }
        for (RepoFile f : todo) {
            Path blob = cache.blob(repo, f.blobId());
            if (!complete(blob, f)) {
                Path part = partOf(blob);
                Progress progress = new Progress(f.path(), f.size(), err, Main.terminal());
                hub.download(repo, commit, f, part, progress);
                progress.finish();
                Files.move(part, blob, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } else {
                out.println("  " + f.path() + "  " + Fmt.bytes(f.size()) + "  (blob already cached; linked)");
            }
            cache.link(repo, commit, f.path(), f.blobId());
        }
        if (!revision.matches("[0-9a-f]{40}")) cache.writeRef(repo, revision, commit);
        if (!files.isEmpty()) out.println(repo + " @ " + commit.substring(0, Math.min(12, commit.length())) + " -> "
                + cache.repoDir(repo).resolve("snapshots").resolve(commit));
    }

    private static Path partOf(Path blob) { return blob.resolveSibling(blob.getFileName() + ".incomplete"); }

    private static boolean complete(Path p, RepoFile f) throws IOException {
        return Files.isRegularFile(p) && (f.size() < 0 || Files.size(p) == f.size());
    }
}
