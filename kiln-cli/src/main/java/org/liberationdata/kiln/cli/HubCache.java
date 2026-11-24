package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.HfCache;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.FileSystemException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * A Hugging Face hub cache, in the layout {@code hf download} writes and {@link HfCache} reads:
 * <pre>
 * models--org--name/
 *   blobs/&lt;etag&gt;                       file contents (sha256 for LFS files, git blob id otherwise)
 *   snapshots/&lt;commit&gt;/&lt;path&gt;         relative symlinks into blobs/
 *   refs/main                          the commit "main" pointed at when last downloaded
 * </pre>
 */
final class HubCache {
    /** A file in some snapshot of a cached repo. {@code location} is the snapshot path (usually a symlink). */
    record CachedFile(String repo, String path, Path location, long size) {}

    private final Path root;

    HubCache(Path root) { this.root = root; }

    /** The cache {@link HfCache} uses: {@code HF_HUB_CACHE}, {@code $HF_HOME/hub} or {@code ~/.cache/huggingface/hub}. */
    static HubCache standard() { return new HubCache(HfCache.root()); }

    Path root() { return root; }

    static String folderName(String repo) { return "models--" + repo.replace("/", "--"); }

    /** "models--org--name" to "org/name", or null for anything else (datasets, spaces, stray files). */
    static String repoOf(String folderName) {
        if (!folderName.startsWith("models--")) return null;
        String rest = folderName.substring("models--".length());
        int sep = rest.indexOf("--");
        return sep <= 0 ? null : rest.substring(0, sep) + "/" + rest.substring(sep + 2);
    }

    Path repoDir(String repo) { return root.resolve(folderName(repo)); }

    Path blob(String repo, String blobId) { return repoDir(repo).resolve("blobs").resolve(blobId); }

    Path snapshotFile(String repo, String commit, String path) {
        return repoDir(repo).resolve("snapshots").resolve(commit).resolve(path);
    }

    Optional<String> ref(String repo, String name) {
        Path p = repoDir(repo).resolve("refs").resolve(name);
        try { return Files.isRegularFile(p) ? Optional.of(Files.readString(p).trim()) : Optional.empty(); }
        catch (IOException e) { throw new UncheckedIOException(e); }
    }

    void writeRef(String repo, String name, String commit) throws IOException {
        Path p = repoDir(repo).resolve("refs").resolve(name);
        Files.createDirectories(p.getParent());
        Files.writeString(p, commit);   // no trailing newline, like huggingface_hub
    }

    /** Points {@code snapshots/<commit>/<path>} at {@code blobs/<blobId>} with a relative symlink (a copy where symlinks fail). */
    Path link(String repo, String commit, String path, String blobId) throws IOException {
        Path file = snapshotFile(repo, commit, path), blob = blob(repo, blobId);
        Files.createDirectories(file.getParent());
        Files.deleteIfExists(file);
        try {
            Files.createSymbolicLink(file, file.getParent().relativize(blob));
        } catch (UnsupportedOperationException | FileSystemException e) {
            Files.copy(blob, file, StandardCopyOption.REPLACE_EXISTING);
        }
        return file;
    }

    /**
     * {@code path} in a snapshot of {@code repo}: the one {@code refs/main} names first, then any other
     * (the same files {@link HfCache#find} accepts).
     */
    Optional<Path> find(String repo, String path) {
        return files(repo).stream().filter(f -> f.path().equals(path)).map(CachedFile::location).findFirst();
    }

    /** Every complete file of {@code repo}, one entry per path, preferring the snapshot {@code refs/main} names. */
    List<CachedFile> files(String repo) {
        Path snapshots = repoDir(repo).resolve("snapshots");
        if (!Files.isDirectory(snapshots)) return List.of();
        String main = ref(repo, "main").orElse("");
        Map<String, CachedFile> byPath = new LinkedHashMap<>();
        try (Stream<Path> s = Files.list(snapshots)) {
            List<Path> snaps = s.filter(Files::isDirectory)
                    .sorted(Comparator.comparing((Path p) -> !p.getFileName().toString().equals(main)))
                    .toList();
            for (Path snap : snaps) {
                try (Stream<Path> w = Files.walk(snap)) {
                    for (Path f : w.filter(Files::isRegularFile).sorted().toList()) {   // follows links; drops dangling ones
                        String rel = snap.relativize(f).toString().replace('\\', '/');
                        byPath.putIfAbsent(rel, new CachedFile(repo, rel, f, Files.size(f)));
                    }
                }
            }
        } catch (IOException e) { throw new UncheckedIOException(e); }
        return List.copyOf(byPath.values());
    }

    /** The repos in the cache, sorted. */
    List<String> repos() {
        if (!Files.isDirectory(root)) return List.of();
        try (Stream<Path> s = Files.list(root)) {
            List<String> out = new ArrayList<>();
            s.map(p -> repoOf(p.getFileName().toString())).filter(r -> r != null).sorted(String.CASE_INSENSITIVE_ORDER)
                    .forEach(out::add);
            return out;
        } catch (IOException e) { throw new UncheckedIOException(e); }
    }
}
