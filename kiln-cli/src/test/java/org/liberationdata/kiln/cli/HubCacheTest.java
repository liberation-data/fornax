package org.liberationdata.kiln.cli;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class HubCacheTest {
    @TempDir Path root;

    static final String REPO = "Qwen/Qwen3-0.6B-GGUF";

    @Test
    void folderNamesMatchTheHub() {
        assertEquals("models--Qwen--Qwen3-0.6B-GGUF", HubCache.folderName(REPO));
        assertEquals(REPO, HubCache.repoOf("models--Qwen--Qwen3-0.6B-GGUF"));
        assertEquals("a/b--c", HubCache.repoOf("models--a--b--c"));
        assertNull(HubCache.repoOf("datasets--a--b"));
        assertNull(HubCache.repoOf("models--nosep"));
        assertNull(HubCache.repoOf("CACHEDIR.TAG"));
    }

    @Test
    void linksSnapshotsToBlobsLikeHfDownload() throws Exception {
        var cache = new HubCache(root);
        Path blob = cache.blob(REPO, "9465e6");
        Files.createDirectories(blob.getParent());
        Files.writeString(blob, "weights");
        Path link = cache.link(REPO, "c0ffee", "sub/dir/model.gguf", "9465e6");
        cache.writeRef(REPO, "main", "c0ffee");

        Path dir = root.resolve("models--Qwen--Qwen3-0.6B-GGUF");
        assertEquals(dir.resolve("snapshots/c0ffee/sub/dir/model.gguf"), link);
        assertTrue(Files.isSymbolicLink(link));
        assertEquals(Path.of("../../../../blobs/9465e6"), Files.readSymbolicLink(link));
        assertEquals("weights", Files.readString(link));
        assertEquals("c0ffee", Files.readString(dir.resolve("refs/main")));
        assertEquals("c0ffee", cache.ref(REPO, "main").orElseThrow());

        assertEquals(link, cache.find(REPO, "sub/dir/model.gguf").orElseThrow());
        assertTrue(cache.find(REPO, "other.gguf").isEmpty());
        assertTrue(cache.find("x/y", "model.gguf").isEmpty());
        assertEquals(List.of(REPO), cache.repos());
        var f = cache.files(REPO).get(0);
        assertEquals("sub/dir/model.gguf", f.path());
        assertEquals(7, f.size());
    }

    @Test
    void prefersTheMainSnapshotAndSkipsDanglingLinks() throws Exception {
        var cache = new HubCache(root);
        for (String id : List.of("old", "new")) {
            Files.createDirectories(cache.blob(REPO, id).getParent());
            Files.writeString(cache.blob(REPO, id), id);
        }
        cache.link(REPO, "aaa-old", "m.gguf", "old");
        cache.link(REPO, "zzz-new", "m.gguf", "new");
        cache.writeRef(REPO, "main", "zzz-new");
        assertEquals("new", Files.readString(cache.find(REPO, "m.gguf").orElseThrow()));

        cache.link(REPO, "zzz-new", "gone.gguf", "missing-blob");   // e.g. a blob someone deleted
        assertTrue(cache.find(REPO, "gone.gguf").isEmpty());
        assertEquals(List.of("m.gguf"), cache.files(REPO).stream().map(HubCache.CachedFile::path).toList());
    }

    @Test
    void emptyOrMissingCacheIsFine() {
        var cache = new HubCache(root.resolve("nope"));
        assertEquals(List.of(), cache.repos());
        assertEquals(List.of(), cache.files(REPO));
    }
}
