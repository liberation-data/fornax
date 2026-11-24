package org.liberationdata.kiln.cli;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.kiln.cli.ModelSets.Kind;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class ModelSetsTest {
    @TempDir Path root;

    @Test
    void namesAreUniqueAndLookupIgnoresCase() {
        Set<String> names = new HashSet<>();
        for (var m : ModelSets.ALL) assertTrue(names.add(m.name()), "duplicate alias " + m.name());
        assertEquals("qwen3-4b", ModelSets.byName("Qwen3-4B").orElseThrow().name());
        assertTrue(ModelSets.byName("nope").isEmpty());
        for (String n : List.of("qwen3-4b", "ace-step", "yue2", "flux2-klein")) assertTrue(names.contains(n), n);
    }

    @Test
    void everySetHasTheRolesItsEngineNeeds() {
        for (var m : ModelSets.ALL) {
            List<String> roles = m.components().stream().map(ModelSets.Component::role).toList();
            List<String> want = switch (m.kind()) {
                case LLM -> List.of("model");
                case IMAGE -> List.of("diffusion", "vae", "llm");
                case SONG -> m.name().startsWith("ace-step") ? List.of("lm", "text-encoder", "dit", "vae") : List.of("model", "vae");
            };
            assertEquals(want, roles, m.name());
            assertTrue(m.components().stream().allMatch(c -> Models.REPO.matcher(c.primary().repo()).matches()), m.name());
        }
    }

    @Test
    void llmAliasesAndEngineRepos() {
        assertEquals("qwen3-0.6b", ModelSets.llmFor("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf").orElseThrow().name());
        assertTrue(ModelSets.llmFor("Qwen/Qwen3-0.6B-GGUF", "other.gguf").isEmpty());
        assertTrue(ModelSets.isEngineRepo("Serveurperso/ACE-Step-1.5-GGUF"));
        assertFalse(ModelSets.isEngineRepo("unsloth/Qwen3-4B-Instruct-2507-GGUF"));
    }

    @Test
    void pullGroupsFilesByRepoAndSkipsComponentsCachedElsewhere() throws Exception {
        var cache = new HubCache(root);
        var ace = ModelSets.byName("ace-step").orElseThrow();
        assertEquals(Map.of("Serveurperso/ACE-Step-1.5-GGUF", List.of("acestep-5Hz-lm-4B-Q8_0.gguf",
                "Qwen3-Embedding-0.6B-Q8_0.gguf", "acestep-v15-turbo-Q8_0.gguf", "vae-BF16.gguf")), Pull.aliasFiles(ace, cache));

        var qwen = ModelSets.byName("qwen-image").orElseThrow();
        assertEquals(3, Pull.aliasFiles(qwen, cache).get("Comfy-Org/Qwen-Image_ComfyUI").size());   // diffusion, vae, llm
        // the text encoder is already in the cache from its alternate repo: don't fetch it again
        String alt = "Comfy-Org/HunyuanVideo_1.5_repackaged", enc = "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors";
        Files.createDirectories(cache.blob(alt, "b").getParent());
        Files.writeString(cache.blob(alt, "b"), "x");
        cache.link(alt, "c", enc, "b");
        assertEquals(Map.of("Comfy-Org/Qwen-Image_ComfyUI", List.of(
                "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors", "split_files/vae/qwen_image_vae.safetensors")),
                Pull.aliasFiles(qwen, cache));
        assertEquals(cache.find(alt, enc), qwen.component("llm").locate(cache));
        assertEquals(Kind.IMAGE, qwen.kind());
    }
}
