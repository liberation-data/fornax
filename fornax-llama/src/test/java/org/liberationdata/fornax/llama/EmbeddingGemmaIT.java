package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Vectors;
import org.liberationdata.fornax.llama.EmbeddingsIT.Reference;
import org.liberationdata.fornax.llama.LlamaModel.EmbedOptions;
import org.liberationdata.fornax.llama.LlamaModel.Mode;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Runs EmbeddingGemma 2 (#152) when it's in the Hugging Face cache and the native library is built; skipped otherwise.
 * The reference embeddings are llama.cpp's own llama-embedding on the same model and texts, with EmbeddingGemma's
 * prefixes written out (scripts/embedding-reference.sh ... embeddinggemma-2 writes them).
 */
class EmbeddingGemmaIT {
    static final Weights W = Weights.EMBEDDINGGEMMA_2;
    static LlamaModel model;

    @BeforeAll
    static void load() {
        assumeTrue(W.downloaded(), "model not downloaded: " + W.downloadCommand());
        model = new LlamaModel(W, Mode.EMBED);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    static List<Reference> reference() throws IOException { return EmbeddingsIT.reference("embeddinggemma-2"); }

    @Test
    void embedsInSevenHundredAndSixtyEightDimensions() {
        assertEquals(768, model.embeddingSize());
    }

    @Test
    void matchesLlamaEmbedding() throws IOException {
        List<Reference> ref = reference();
        float[][] v = model.embed(ref.stream().map(Reference::text).toList());
        assertEquals(ref.size(), v.length);
        for (int i = 0; i < v.length; i++) {
            for (float x : v[i]) assertTrue(Float.isFinite(x), "text " + i + " has a NaN or infinity");
            double cos = Vectors.cosine(v[i], ref.get(i).embedding());
            System.out.printf("llama-embedding parity, text %d: cosine %.7f%n", i, cos);
            assertTrue(cos > 0.999, "text " + i + ": cosine " + cos);
        }
    }

    @Test
    void thePrefixOptionsAreGemmasFormat() throws IOException {
        List<Reference> ref = reference();
        String doc = ref.get(0).text().substring("title: none | text: ".length());
        String query = ref.get(2).text().substring("task: search result | query: ".length());
        float[] d = model.embed(List.of(doc), EmbedOptions.gemmaDocument())[0];
        float[] q = model.embed(List.of(query), EmbedOptions.gemmaQuery("search result"))[0];
        assertTrue(Vectors.cosine(d, ref.get(0).embedding()) > 0.999);
        assertTrue(Vectors.cosine(q, ref.get(2).embedding()) > 0.999);
    }

    @Test
    void aQueryIsNearerItsAnswer() throws IOException {
        List<Reference> ref = reference();
        float[][] v = model.embed(ref.stream().map(Reference::text).toList());
        double kiln = Vectors.cosine(v[2], v[0]), nashville = Vectors.cosine(v[2], v[1]);
        assertTrue(kiln > nashville + 0.1, "kiln " + kiln + ", nashville " + nashville);
    }
}
