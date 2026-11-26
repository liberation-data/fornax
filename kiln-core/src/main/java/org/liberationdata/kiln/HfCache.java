package org.liberationdata.kiln;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Finds files already downloaded into a Hugging Face hub cache (for example with {@code hf download}).
 * To download them too, or find them in a ComfyUI models folder, see {@link org.liberationdata.kiln.hub.KilnHub}.
 *
 * <p>Cache location: {@code HF_HUB_CACHE}, else {@code $HF_HOME/hub}, else {@code $XDG_CACHE_HOME/huggingface/hub},
 * else {@code ~/.cache/huggingface/hub}, as {@code huggingface_hub} has it (a leading {@code ~} is the home directory).
 */
public final class HfCache {
    private HfCache() {}

    public static Path root() {
        String hub = System.getenv("HF_HUB_CACHE");
        return hub != null && !hub.isBlank() ? expandHome(hub) : home().resolve("hub");
    }

    /**
     * Where {@code huggingface_hub} keeps its files (the cache, the token): {@code HF_HOME}, else
     * {@code $XDG_CACHE_HOME/huggingface}, else {@code ~/.cache/huggingface}.
     */
    public static Path home() {
        String home = System.getenv("HF_HOME");
        if (home != null && !home.isBlank()) return expandHome(home);
        String xdg = System.getenv("XDG_CACHE_HOME");
        return (xdg != null && !xdg.isBlank() ? expandHome(xdg) : Path.of(System.getProperty("user.home"), ".cache")).resolve("huggingface");
    }

    /** The path, with a leading {@code ~} as the home directory, as Python's {@code expanduser} reads it. */
    public static Path expandHome(String path) {
        String p = path.trim();
        if (p.equals("~") || p.startsWith("~/") || p.startsWith("~" + java.io.File.separator))
            return Path.of(System.getProperty("user.home") + p.substring(1));
        return Path.of(p);
    }

    /** The file at {@code path} inside any downloaded snapshot of {@code repo} (e.g. "unsloth/Qwen3-4B-GGUF"). */
    public static Optional<Path> find(String repo, String path) {
        Path snapshots = root().resolve("models--" + repo.replace("/", "--")).resolve("snapshots");
        if (!Files.isDirectory(snapshots)) return Optional.empty();
        try (Stream<Path> s = Files.list(snapshots)) {
            return s.map(snap -> snap.resolve(path)).filter(Files::exists).findFirst();
        } catch (IOException e) { throw new UncheckedIOException(e); }
    }

    /** Like {@link #find} but fails with the {@code hf download} command that would fetch the file. */
    public static Path require(String repo, String path) {
        return find(repo, path).orElseThrow(() -> new IllegalStateException(
                "Not in the Hugging Face cache (" + root() + "): " + repo + " " + path
                        + "\n  fetch it with: hf download " + repo + " " + path));
    }

    /** The first of several (repo, path) pairs that is present; pairs are given flat: repo1, path1, repo2, path2... */
    public static Path requireAny(String... repoPathPairs) {
        for (int i = 0; i + 1 < repoPathPairs.length; i += 2) {
            Optional<Path> p = find(repoPathPairs[i], repoPathPairs[i + 1]);
            if (p.isPresent()) return p.get();
        }
        return require(repoPathPairs[0], repoPathPairs[1]);
    }

    /** The model repos ({@code "org/name"}) in the cache at {@code root}, sorted ignoring case. */
    public static List<String> repos(Path root) {
        if (!Files.isDirectory(root)) return List.of();
        try (Stream<Path> s = Files.list(root)) {
            List<String> out = new ArrayList<>();
            s.map(p -> repoOf(p.getFileName().toString())).filter(r -> r != null).sorted(String.CASE_INSENSITIVE_ORDER)
                    .forEach(out::add);
            return out;
        } catch (IOException e) { throw new UncheckedIOException(e); }
    }

    /**
     * Every complete file of {@code repo} in the cache at {@code root}, by its path in the repo, in path order. A path in
     * several snapshots is the one in the snapshot {@code refs/main} names, else in the first other. A link whose blob
     * isn't there (a download still in progress) doesn't count.
     */
    public static Map<String, Path> files(Path root, String repo) {
        Path snapshots = root.resolve("models--" + repo.replace("/", "--")).resolve("snapshots");
        if (!Files.isDirectory(snapshots)) return Map.of();
        String main = "";
        try {
            Path ref = snapshots.resolveSibling("refs").resolve("main");
            if (Files.isRegularFile(ref)) main = Files.readString(ref).trim();
        } catch (IOException e) { throw new UncheckedIOException(e); }
        String head = main;
        Map<String, Path> byPath = new LinkedHashMap<>();
        try (Stream<Path> s = Files.list(snapshots)) {
            List<Path> snaps = s.filter(Files::isDirectory)
                    .sorted(Comparator.comparing((Path p) -> !p.getFileName().toString().equals(head)).thenComparing(Path::toString))
                    .toList();
            for (Path snap : snaps) {
                try (Stream<Path> w = Files.walk(snap)) {
                    for (Path f : w.filter(Files::isRegularFile).sorted().toList())   // follows links; drops dangling ones
                        byPath.putIfAbsent(snap.relativize(f).toString().replace('\\', '/'), f);
                }
            }
        } catch (IOException e) { throw new UncheckedIOException(e); }
        Map<String, Path> sorted = new LinkedHashMap<>();
        byPath.keySet().stream().sorted().forEach(k -> sorted.put(k, byPath.get(k)));
        return sorted;
    }

    /** {@code "models--org--name"} to {@code "org/name"}, or null for anything else (datasets, spaces, {@code .locks}). */
    static String repoOf(String folderName) {
        if (!folderName.startsWith("models--")) return null;
        String rest = folderName.substring("models--".length());
        int sep = rest.indexOf("--");
        return sep <= 0 ? null : rest.substring(0, sep) + "/" + rest.substring(sep + 2);
    }
}
