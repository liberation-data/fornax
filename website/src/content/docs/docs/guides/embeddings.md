---
title: "Embeddings and reranking"
description: Embed text and rerank documents from Java with Qwen3-Embedding and Qwen3-Reranker through llama.cpp, and search your own documents in memory, with no vector database.
---

An embedding model turns a text into a vector, so that texts about the same thing get vectors that point the same way.
A reranker reads a query and a document together and scores how well the document answers it: slower than comparing
vectors, and more accurate. Together they are the retrieval half of RAG: embed your documents once, embed each
question, take the nearest few dozen passages, and rerank those.

Both are a `LlamaModel` loaded in another `Mode`. They run through llama.cpp's pooled embeddings output, the way
`llama-embedding` and `llama-server --embedding` / `--reranking` run them, and Fornax's embeddings match
`llama-embedding`'s for the same model and text.

## Embed

```sh
hf download Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-Q8_0.gguf
```

<!-- snippet: Embeddings#embed -->
```java
import org.liberationdata.fornax.Vectors;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.EmbedOptions;
import org.liberationdata.fornax.llama.LlamaModel.Mode;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import java.util.List;

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
```

- **Normalised by default.** Each vector has length 1, so the dot product is the cosine similarity.
  `EmbedOptions.DEFAULT.withNormalize(false)` returns the pooled vector as the model made it.
- **Queries take an instruction; documents don't.** Qwen3-Embedding was trained with
  `Instruct: <task>\nQuery:<query>` for queries, and nothing before documents. `EmbedOptions.query(task)` writes
  that prefix; `EmbedOptions.DEFAULT.withPrefix(...)` puts any other text before each input, for a model with its own format.
- **Batched.** `embed` takes a list, and runs as many texts through the model at once as fit in the context (up to
  256), each as its own sequence. A text embedded with others comes out very slightly different from one embedded
  alone (cosine 0.9999), in `llama-embedding` too: the batch changes the shapes of the GPU kernels.
- **One text fits in the context.** In EMBED and RERANK mode the context size is also the batch, so the default
  8192 tokens is the longest text (or query plus document) one call takes. A longer one fails with an error that
  says so; split long documents into passages first. `new LlamaModel(w, gpuLayers, contextSize, Mode.EMBED)` sets it.
- **The pooling is the model's.** The GGUF says how to pool the model's outputs into one vector (Qwen3-Embedding
  takes the last token, the end-of-text the tokenizer appends; others take the mean or the first token).
- Text isn't parsed for special tokens: a document that contains `<|im_end|>` embeds those characters as text.

`embeddingSize()` is the vector length. `embedAsync(texts, options)` runs `embed` on the model's own thread, and
`cancel()` stops a long `embed` between batches with a `CancellationException`.

## Rerank

```sh
hf download ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF qwen3-reranker-0.6b-q8_0.gguf
```

<!-- snippet: Embeddings#rerank -->
```java
import org.liberationdata.fornax.Scored;

try (var rr = new LlamaModel(Weights.QWEN3_RERANKER_0_6B, Mode.RERANK)) {
    List<Scored> ranked = rr.rerank("How hot does a fornax get?", List.of(
            "A pottery fornax fires stoneware at around 1200 to 1300 degrees Celsius.",
            "Nashville is the capital of Tennessee and the home of country music.",
            "A fornax is a thermally insulated chamber."));
    // best first: [Scored[index=0, score=0.997], Scored[index=2, score=0.564], Scored[index=1, score=0.00002]]
}
```

`rerank` returns every document's score, best first; `index` is the document's position in the list you passed. Fornax
builds each pair's prompt as `llama-server`'s `/rerank` does, from the reranker GGUF's own `rerank` template, and for
Qwen3-Reranker a score is the probability, 0 to 1, that the document answers the query (a near-even one moves by a few
hundredths depending on what else is in the batch). As in `llama-server`, the filled-in prompt is parsed for special
tokens, so keep chat-template markers such as `<|im_end|>` out of queries and documents. `rerankAsync(query,
documents)` runs it on the model's thread, and `cancel()` stops it between batches.

## Search a folder

`Vectors` (in `fornax-core`) is all the search there is: `topK(query, vectors, k)` scans the vectors for the k most
similar by cosine. A scan over 200,000 1024-float vectors takes about 0.2 s on one core, so many applications
don't need a vector database: keep the vectors in memory, or save them next to the documents.
`Vectors` also has `cosine`, `dot`, `normalize`, and `rank(scores)` (indexes sorted by score).

The `search` example does the whole thing over a folder of `.txt` and `.md` files: splits them into passages of
about 800 characters, embeds them, takes the 20 nearest to the question, reranks them, and prints the best five
with their file and line.

```sh
./gradlew search --args='docs "Why does a long video lose the GPU on Vulkan?"'
./gradlew search --args='my-notes "what did we decide about pricing?"' -Pk=10 -Prerank=false
```

## Speed

On a Radeon 8060S (Strix Halo) with the Vulkan build, Qwen3-Embedding 0.6B (Q8_0) embeds about 420 short passages
a second (about 20 tokens each, 256 to a batch), about what `llama-embedding` does on the same machine and texts;
800-character passages go at about 50 a second.

## Errors

A model loaded for one mode can't do the others: `embed` on a chat model, or `chat` on an embedding model, throws an
`IllegalStateException` that names the mode to load it with. Loading a model in a mode it wasn't made for fails too:
an embedding GGUF as `Mode.RERANK` ("not a reranker"), a reranker as `Mode.EMBED`, or a chat model with no pooling
type as `Mode.EMBED`.
