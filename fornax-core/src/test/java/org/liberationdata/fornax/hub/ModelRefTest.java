package org.liberationdata.fornax.hub;

import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class ModelRefTest {
    @Test void parsesRepoRevisionAndTag() {
        assertEquals(new ModelRef("unsloth/Qwen3-4B-Instruct-2507-GGUF", null, "Q4_K_M"),
                ModelRef.parse("unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M"));
        assertEquals(new ModelRef("Qwen/Qwen3-0.6B-GGUF", null, null), ModelRef.parse("Qwen/Qwen3-0.6B-GGUF"));
        assertEquals(new ModelRef("some/repo", "v1.2", "Q8_0"), ModelRef.parse("some/repo@v1.2:Q8_0"));
        assertEquals(new ModelRef("some/repo", "refs/pr/3", null), ModelRef.parse(" some/repo@refs/pr/3 "));
        assertEquals(new ModelRef("ggerganov/whisper.cpp", null, "ggml-base.en.bin"), ModelRef.parse("ggerganov/whisper.cpp:ggml-base.en.bin"));
        assertEquals(new ModelRef("a/b", null, "sub/dir/m.gguf"), ModelRef.parse("a/b:sub/dir/m.gguf"));
    }

    @Test void printsAsItParses() {
        for (String s : new String[]{"a/b", "a/b:Q4_K_M", "a/b@dev", "a/b@dev:x.gguf"}) assertEquals(s, ModelRef.parse(s).toString());
    }

    @Test void refusesWhatIsntAReference() {
        for (String s : new String[]{"qwen3-4b", "/abs/path/m.gguf", "a/b/c", "a/b:", "a/b:x:y", "", "a b/c"})
            assertEquals(Optional.empty(), ModelRef.tryParse(s), s);
        var e = assertThrows(IllegalArgumentException.class, () -> ModelRef.parse("qwen3-4b"));
        assertTrue(e.getMessage().contains("owner/repo"), e.getMessage());
    }

    @Test void aUrlSaysWhichReferenceToUse() {
        var e = assertThrows(IllegalArgumentException.class, () -> ModelRef.parse(
                "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/blob/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"));
        assertTrue(e.getMessage().contains("use unsloth/Qwen3-4B-Instruct-2507-GGUF:Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), e.getMessage());
        assertEquals(Optional.of("a/b@dev:x.gguf"), ModelRef.urlHint("https://hf.co/a/b/resolve/dev/x.gguf?download=true"));
        assertEquals(Optional.of("a/b"), ModelRef.urlHint("https://huggingface.co/a/b"));
        assertEquals(Optional.of("a/b"), ModelRef.urlHint("https://huggingface.co/a/b/tree/main"));
        assertEquals(Optional.empty(), ModelRef.urlHint("https://example.com/a/b"));
    }

    @Test void aRelativeModelPathIsntAReference() {
        assertTrue(ModelRef.looksLikeReference("unsloth/Qwen3-4B-GGUF:Q4_K_M"));
        assertTrue(ModelRef.looksLikeReference("Qwen/Qwen3-0.6B-GGUF"));
        assertFalse(ModelRef.looksLikeReference("models/qwen3.gguf"));       // a path that isn't there
        assertFalse(ModelRef.looksLikeReference("weights/whisper.bin"));
        assertTrue(ModelRef.looksLikeReference("models/qwen3.gguf:Q4_K_M")); // a tag makes it a reference
    }
}
