package org.liberationdata.kiln.cli;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class ModelsTest {
    @TempDir Path root;

    HubCache cache() throws Exception {
        var cache = new HubCache(root.resolve("hub"));
        put(cache, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf");
        put(cache, "some/Multi-GGUF", "m-Q4_K_M.gguf");
        put(cache, "some/Multi-GGUF", "m-Q8_0.gguf");
        put(cache, "some/Multi-GGUF", "mmproj-m-F16.gguf");
        return cache;
    }

    static void put(HubCache cache, String repo, String file) throws Exception {
        Path blob = cache.blob(repo, file + ".blob");
        Files.createDirectories(blob.getParent());
        Files.writeString(blob, file);
        cache.link(repo, "abc", file, file + ".blob");
    }

    @Test
    void resolvesPathsAliasesAndRepos() throws Exception {
        var cache = cache();
        Path file = Files.writeString(root.resolve("local.gguf"), "x");
        assertEquals(file, ((Models.Llm) Models.resolve(file.toString(), cache)).path());

        var alias = (Models.Llm) Models.resolve("qwen3-0.6b", cache);
        assertEquals("Qwen3-0.6B-Q8_0.gguf", alias.path().getFileName().toString());
        assertEquals(alias.path(), ((Models.Llm) Models.resolve("Qwen/Qwen3-0.6B-GGUF", cache)).path());   // one model in the repo
        assertEquals("m-Q8_0.gguf", ((Models.Llm) Models.resolve("some/Multi-GGUF:q8_0", cache)).path().getFileName().toString());
        assertEquals("m-Q4_K_M.gguf", ((Models.Llm) Models.resolve("some/Multi-GGUF:m-Q4_K_M.gguf", cache)).path().getFileName().toString());

        assertInstanceOf(Models.Set.class, Models.resolve("ace-step", cache));
    }

    @Test
    void explainsWhatIsMissing() throws Exception {
        var cache = cache();
        var notPulled = assertThrows(Main.Failure.class, () -> Models.resolve("qwen3-4b", cache));
        assertTrue(notPulled.getMessage().contains("kiln pull qwen3-4b"), notPulled.getMessage());
        var several = assertThrows(Main.Failure.class, () -> Models.resolve("some/Multi-GGUF", cache));
        assertTrue(several.getMessage().contains("some/Multi-GGUF:m-Q8_0.gguf"), several.getMessage());
        assertFalse(several.getMessage().contains("mmproj"), several.getMessage());
        var none = assertThrows(Main.Failure.class, () -> Models.resolve("some/Multi-GGUF:IQ2", cache));
        assertTrue(none.getMessage().contains("kiln pull some/Multi-GGUF:IQ2"), none.getMessage());
        var unknown = assertThrows(Main.Failure.class, () -> Models.resolve("what", cache));
        assertTrue(unknown.getMessage().contains("qwen3-4b"), unknown.getMessage());
    }
}
