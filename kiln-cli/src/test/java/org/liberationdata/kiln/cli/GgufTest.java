package org.liberationdata.kiln.cli;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GgufTest {
    static final List<String> FILES = List.of(
            "README.md", "Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B-UD-Q4_K_M.gguf", "Qwen3-4B-Q8_0.gguf", "mmproj-F16.gguf",
            "Q6_K/big-Q6_K-00002-of-00002.gguf", "Q6_K/big-Q6_K-00001-of-00002.gguf");

    @Test
    void groupsSplitModelsAndSkipsProjectors() {
        var models = Gguf.models(FILES);
        assertEquals(List.of("Q6_K/big-Q6_K-00001-of-00002.gguf", "Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B-Q8_0.gguf",
                "Qwen3-4B-UD-Q4_K_M.gguf"), List.copyOf(models.keySet()));
        assertEquals(List.of("Q6_K/big-Q6_K-00001-of-00002.gguf", "Q6_K/big-Q6_K-00002-of-00002.gguf"),
                models.get("Q6_K/big-Q6_K-00001-of-00002.gguf"));
    }

    @Test
    void matchesTagsAndNames() {
        var models = Gguf.models(FILES);
        assertEquals(List.of("Qwen3-4B-Q8_0.gguf"), Gguf.match(models, "q8_0"));
        assertEquals(List.of("Q6_K/big-Q6_K-00001-of-00002.gguf"), Gguf.match(models, "Q6_K"));
        assertEquals(List.of("Qwen3-4B-Q4_K_M.gguf"), Gguf.match(models, "Qwen3-4B-Q4_K_M.gguf"));   // exact beats substring
        assertEquals(List.of("Q6_K/big-Q6_K-00001-of-00002.gguf"), Gguf.match(models, "Q6_K/big-Q6_K-00002-of-00002.gguf"));
        assertEquals(2, Gguf.match(models, "Q4_K_M").size());   // ambiguous: Q4_K_M and UD-Q4_K_M
        assertEquals(List.of(), Gguf.match(models, "IQ2"));
    }

    @Test
    void allPartsNamesEverySplitPartAsTheFilesSpellIt() {
        assertEquals(List.of("Qwen3-4B-Q8_0.gguf"), Gguf.allParts(List.of("Qwen3-4B-Q8_0.gguf")));
        assertEquals(List.of("Q6_K/big-Q6_K-00001-of-00002.gguf", "Q6_K/big-Q6_K-00002-of-00002.gguf"),
                Gguf.allParts(List.of("Q6_K/big-Q6_K-00002-of-00002.gguf")));   // part 1 missing
        assertEquals(List.of("BIG-00001-OF-00002.GGUF", "BIG-00002-OF-00002.GGUF"),
                Gguf.allParts(List.of("BIG-00001-OF-00002.GGUF", "BIG-00002-OF-00002.GGUF")));
    }
}
