package org.liberationdata.kiln.studio;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class CatalogTest {
    @TempDir Path root;

    private void file(String repo, String snapshot, String path, int bytes) throws Exception {
        Path p = root.resolve("models--" + repo.replace("/", "--")).resolve("snapshots").resolve(snapshot).resolve(path);
        Files.createDirectories(p.getParent());
        Files.write(p, new byte[bytes]);
    }

    @Test void findsChatModelsButNotTheSongEnginesOrEmbedders() throws Exception {
        file("Qwen/Qwen3-0.6B-GGUF", "a", "Qwen3-0.6B-Q8_0.gguf", 10);
        file("Qwen/Qwen3-0.6B-GGUF", "b", "Qwen3-0.6B-Q8_0.gguf", 10);  // the same file in a second snapshot
        file("big/Model-GGUF", "a", "q4/Model-Q4-00001-of-00002.gguf", 5);
        file("big/Model-GGUF", "a", "q4/Model-Q4-00002-of-00002.gguf", 7);
        file("Serveurperso/ACE-Step-1.5-GGUF", "a", "acestep-v15-turbo-Q8_0.gguf", 1);
        file("some/Embedder-GGUF", "a", "Qwen3-Embedding-0.6B-Q8_0.gguf", 1);
        file("some/safetensors", "a", "model.safetensors", 1);

        List<Catalog.Model> chat = new Catalog(root).chatModels();
        assertEquals(List.of("chat:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf", "chat:big/Model-GGUF/q4/Model-Q4-00001-of-00002.gguf"),
                chat.stream().map(Catalog.Model::id).sorted().toList());
        var sharded = chat.stream().filter(m -> m.id().contains("big")).findFirst().orElseThrow();
        assertEquals(12, sharded.sizeBytes());
        assertEquals("Model-Q4", sharded.name());
        assertTrue(sharded.available());
    }

    @Test void aPresetIsAvailableOnlyWithAllItsFiles() throws Exception {
        String ace = "Serveurperso/ACE-Step-1.5-GGUF";
        file(ace, "a", "Qwen3-Embedding-0.6B-Q8_0.gguf", 1);
        file(ace, "a", "acestep-v15-turbo-Q8_0.gguf", 2);
        var catalog = new Catalog(root);
        assertFalse(catalog.find("song:acestep-turbo").orElseThrow().available());
        file(ace, "a", "vae-BF16.gguf", 3);
        var turbo = catalog.find("song:acestep-turbo").orElseThrow();
        assertTrue(turbo.available());
        assertEquals(6, turbo.sizeBytes());
        assertFalse(catalog.find("song:acestep-turbo-lm4b").orElseThrow().available());
        assertFalse(catalog.find("image:flux2-klein").orElseThrow().available());
    }

    @Test void anEmptyCacheHasNoChatModels() {
        assertTrue(new Catalog(root.resolve("missing")).chatModels().isEmpty());
    }
}
