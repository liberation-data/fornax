package org.liberationdata.fornax.snippets;

// --8<-- [start:embed]
import org.liberationdata.fornax.Vectors;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.EmbedOptions;
import org.liberationdata.fornax.llama.LlamaModel.Mode;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import java.util.List;
// --8<-- [end:embed]
// --8<-- [start:rerank]
import org.liberationdata.fornax.Scored;
// --8<-- [end:rerank]

/** The snippets of website/src/content/docs/docs/guides/embeddings.md: compiled, never run. */
class Embeddings {
    void embed() {
        // --8<-- [start:embed]
        try (var emb = new LlamaModel(Weights.QWEN3_EMBEDDING_0_6B, Mode.EMBED)) {
            List<String> docs = List.of(
                    "A pottery fornax fires stoneware at around 1200 to 1300 degrees Celsius.",
                    "Nashville is the capital of Tennessee and the home of country music.");
            float[][] d = emb.embed(docs);                                    // one 1024-float vector each, length 1

            String task = "Given a web search query, retrieve relevant passages that answer the query";
            float[] q = emb.embed(List.of("How hot does a fornax get?"), EmbedOptions.query(task))[0];

            var best = Vectors.topK(q, d, 1).get(0);                         // Scored[index=0, score=0.69]
            System.out.println(docs.get(best.index()));
        }
        // --8<-- [end:embed]
    }

    void gemma() {
        // --8<-- [start:gemma]
        try (var emb = new LlamaModel(Weights.EMBEDDINGGEMMA_2, Mode.EMBED)) {
            List<String> docs = List.of(
                    "A pottery fornax fires stoneware at around 1200 to 1300 degrees Celsius.",
                    "Nashville is the capital of Tennessee and the home of country music.");
            float[][] d = emb.embed(docs, EmbedOptions.gemmaDocument());      // 768 floats each, length 1
            float[] q = emb.embed(List.of("How hot does a fornax get?"), EmbedOptions.gemmaQuery("search result"))[0];
            var best = Vectors.topK(q, d, 1).get(0);
        }
        // --8<-- [end:gemma]
    }

    void rerank() {
        // --8<-- [start:rerank]
        try (var rr = new LlamaModel(Weights.QWEN3_RERANKER_0_6B, Mode.RERANK)) {
            List<Scored> ranked = rr.rerank("How hot does a fornax get?", List.of(
                    "A pottery fornax fires stoneware at around 1200 to 1300 degrees Celsius.",
                    "Nashville is the capital of Tennessee and the home of country music.",
                    "A fornax is a thermally insulated chamber."));
            // best first: [Scored[index=0, score=0.997], Scored[index=2, score=0.564], Scored[index=1, score=0.00002]]
        }
        // --8<-- [end:rerank]
    }
}
