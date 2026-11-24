package org.liberationdata.kiln;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * Finds files already downloaded into a Hugging Face hub cache (for example with {@code hf download}).
 * Kiln never downloads anything itself.
 *
 * <p>Cache location: {@code HF_HUB_CACHE}, else {@code $HF_HOME/hub}, else {@code ~/.cache/huggingface/hub}.
 */
public final class HfCache {
    private HfCache() {}

    public static Path root() {
        String hub = System.getenv("HF_HUB_CACHE");
        if (hub != null && !hub.isBlank()) return Path.of(hub);
        String home = System.getenv("HF_HOME");
        if (home != null && !home.isBlank()) return Path.of(home, "hub");
        return Path.of(System.getProperty("user.home"), ".cache", "huggingface", "hub");
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
}
