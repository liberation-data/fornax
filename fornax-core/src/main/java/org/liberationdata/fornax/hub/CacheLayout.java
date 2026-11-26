package org.liberationdata.fornax.hub;

import org.liberationdata.fornax.hub.ModelSet.Source;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileSystemException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * The Hugging Face hub cache's layout, as {@code huggingface_hub} (and so {@code hf download}, transformers and
 * diffusers) writes and reads it, so that what Fornax downloads Python tools find, and the reverse:
 * <pre>
 * &lt;root&gt;/models--org--name/
 *   blobs/&lt;etag&gt;                   file contents, named by the sha256 (LFS files) or git blob id (others)
 *   blobs/&lt;etag&gt;.incomplete        a download in progress; resumed from where it stopped
 *   snapshots/&lt;commit&gt;/&lt;path&gt;     relative symlinks into blobs/
 *   refs/&lt;branch&gt;                  the commit the branch pointed at when last downloaded
 * &lt;root&gt;/.locks/models--org--name/&lt;etag&gt;.lock      held while a blob is downloaded
 * &lt;root&gt;/.locks/models--org--name/&lt;etag&gt;.pending   Fornax's: which file an .incomplete blob is, for status
 * </pre>
 * The lock files are where {@code huggingface_hub} keeps its own, and its cache scan skips {@code .locks/}.
 */
final class CacheLayout {
    private final Path root;

    CacheLayout(Path root) { this.root = root; }

    Path root() { return root; }

    static String folderName(String repo) { return "models--" + repo.replace("/", "--"); }

    static boolean isCommit(String revision) { return revision.matches("[0-9a-f]{40}"); }

    Path repoDir(String repo) { return root.resolve(folderName(repo)); }

    Path blob(String repo, String etag) { return repoDir(repo).resolve("blobs").resolve(etag); }

    Path incomplete(String repo, String etag) { return repoDir(repo).resolve("blobs").resolve(etag + ".incomplete"); }

    Path lockDir(String repo) { return root.resolve(".locks").resolve(folderName(repo)); }

    Path lock(String repo, String etag) { return lockDir(repo).resolve(etag + ".lock"); }

    Path pending(String repo, String etag) { return lockDir(repo).resolve(etag + ".pending"); }

    Path snapshots(String repo) { return repoDir(repo).resolve("snapshots"); }

    Path snapshotFile(String repo, String commit, String path) { return snapshots(repo).resolve(commit).resolve(path); }

    Optional<String> ref(String repo, String name) {
        Path p = repoDir(repo).resolve("refs").resolve(name);
        try { return Files.isRegularFile(p) ? Optional.of(Files.readString(p).trim()) : Optional.empty(); }
        catch (IOException e) { throw new UncheckedIOException(e); }
    }

    /** Records that {@code name} pointed at {@code commit}; atomically, as two pulls may write it at once. */
    void writeRef(String repo, String name, String commit) throws IOException {
        Path p = repoDir(repo).resolve("refs").resolve(name);
        Files.createDirectories(p.getParent());
        Path tmp = Files.createTempFile(p.getParent(), ".ref", ".tmp");
        Files.writeString(tmp, commit);   // no trailing newline, like huggingface_hub
        Files.move(tmp, p, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    /** Points {@code snapshots/<commit>/<path>} at {@code blobs/<etag>} with a relative symlink, or a copy where links fail. */
    Path link(String repo, String commit, String path, String etag) throws IOException {
        Path file = snapshotFile(repo, commit, path), blob = blob(repo, etag);
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
     * The file in a snapshot of the source's repo. A commit revision is that snapshot only. A branch prefers the
     * snapshot its ref names, then {@code main}'s, then any other, as {@link org.liberationdata.fornax.HfCache#find}
     * accepts any: an older snapshot's file is still the model, and offline it is the only one there is.
     */
    Optional<Path> find(Source s) {
        if (isCommit(s.revision())) {
            Path p = snapshotFile(s.repo(), s.revision(), s.file());
            return Files.isRegularFile(p) ? Optional.of(p) : Optional.empty();
        }
        Path snaps = snapshots(s.repo());
        if (!Files.isDirectory(snaps)) return Optional.empty();
        List<String> order = new ArrayList<>();
        ref(s.repo(), s.revision()).ifPresent(order::add);
        ref(s.repo(), "main").ifPresent(order::add);
        try (Stream<Path> l = Files.list(snaps)) {
            l.map(p -> p.getFileName().toString()).sorted().forEach(order::add);
        } catch (IOException e) { throw new UncheckedIOException(e); }
        for (String commit : order) {
            Path p = snapshotFile(s.repo(), commit, s.file());
            if (Files.isRegularFile(p)) return Optional.of(p);   // follows the link; a dangling one doesn't count
        }
        return Optional.empty();
    }

    /** A download in progress for {@code path} of {@code repo}: its etag and expected size, from the .pending note. */
    record Pending(String etag, String path, long size) {}

    void writePending(String repo, String etag, String path, long size) throws IOException {
        Path p = pending(repo, etag);
        Files.createDirectories(p.getParent());
        Files.writeString(p, path + "\n" + size + "\n", StandardCharsets.UTF_8);
    }

    List<Pending> pendings(String repo) {
        Path dir = lockDir(repo);
        if (!Files.isDirectory(dir)) return List.of();
        List<Pending> out = new ArrayList<>();
        try (Stream<Path> l = Files.list(dir)) {
            for (Path p : l.filter(x -> x.getFileName().toString().endsWith(".pending")).sorted().toList()) {
                String name = p.getFileName().toString();
                try {
                    String[] lines = Files.readString(p, StandardCharsets.UTF_8).split("\n");
                    if (lines.length >= 2)
                        out.add(new Pending(name.substring(0, name.length() - ".pending".length()), lines[0], Long.parseLong(lines[1].trim())));
                } catch (NoSuchFileException | NumberFormatException ignored) {   // finished (or garbled) meanwhile
                }
            }
        } catch (IOException e) { throw new UncheckedIOException(e); }
        return out;
    }

    /** Every snapshot entry of {@code repo} (links and copies alike), deepest first, for eviction. */
    List<Path> snapshotEntries(String repo) throws IOException {
        Path snaps = snapshots(repo);
        if (!Files.isDirectory(snaps)) return List.of();
        try (Stream<Path> w = Files.walk(snaps)) {
            return w.filter(p -> !Files.isDirectory(p, LinkOption.NOFOLLOW_LINKS)).sorted(Comparator.reverseOrder()).toList();
        }
    }

    /** The blob a snapshot entry links to, or null for a copy (or a link outside blobs/). */
    Path blobOf(String repo, Path entry) throws IOException {
        if (!Files.isSymbolicLink(entry)) return null;
        Path target = entry.getParent().resolve(Files.readSymbolicLink(entry)).toAbsolutePath().normalize();
        Path blobs = repoDir(repo).resolve("blobs").toAbsolutePath().normalize();
        return blobs.equals(target.getParent()) ? target : null;
    }
}
