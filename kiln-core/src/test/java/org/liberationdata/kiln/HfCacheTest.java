package org.liberationdata.kiln;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class HfCacheTest {
    @TempDir Path root;

    Path link(String repo, String commit, String file, String blob) throws Exception {
        Path dir = root.resolve("models--" + repo.replace("/", "--"));
        Path b = dir.resolve("blobs").resolve(blob);
        Files.createDirectories(b.getParent());
        if (!Files.exists(b)) Files.writeString(b, blob);
        Path snap = dir.resolve("snapshots").resolve(commit).resolve(file);
        Files.createDirectories(snap.getParent());
        Files.createSymbolicLink(snap, snap.getParent().relativize(b));
        return snap;
    }

    @Test
    void listsReposAndTheirFiles() throws Exception {
        Path old = link("org/Model-GGUF", "c1", "a.gguf", "old");
        Path main = link("org/Model-GGUF", "c2", "a.gguf", "new");
        link("org/Model-GGUF", "c1", "sub/b.json", "b");
        Path dangling = root.resolve("models--org--Model-GGUF/snapshots/c2/partial.gguf");
        Files.createSymbolicLink(dangling, Path.of("../../blobs/not-there"));
        Files.createDirectories(root.resolve("models--org--Model-GGUF/refs"));
        Files.writeString(root.resolve("models--org--Model-GGUF/refs/main"), "c2");
        link("another/one", "c", "x.bin", "x");
        Files.createDirectories(root.resolve("datasets--org--data"));
        Files.createDirectories(root.resolve(".locks/models--org--Model-GGUF"));

        assertEquals(List.of("another/one", "org/Model-GGUF"), HfCache.repos(root));
        var files = HfCache.files(root, "org/Model-GGUF");
        assertEquals(List.of("a.gguf", "sub/b.json"), List.copyOf(files.keySet()));
        assertEquals(main, files.get("a.gguf"), "refs/main's snapshot first");
        assertNotEquals(old, files.get("a.gguf"));
        assertTrue(HfCache.files(root, "no/such").isEmpty());
        assertTrue(HfCache.repos(root.resolve("missing")).isEmpty());
    }
}
