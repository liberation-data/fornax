package org.liberationdata.fornax.examples;

import org.liberationdata.fornax.Scored;
import org.liberationdata.fornax.Vectors;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.EmbedOptions;
import org.liberationdata.fornax.llama.LlamaModel.Mode;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import java.io.IOException;
import java.nio.charset.MalformedInputException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * Searches a folder of text files: splits them into passages, embeds the passages with Qwen3-Embedding, finds the
 * passages nearest the query, and reranks those with Qwen3-Reranker. Everything in memory, no vector database.
 * Usage: Search [folder] ["query"] [-Dk=5] [-Drerank=false]. Defaults: the repository's docs/ and a sample question.
 * Models: hf download Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-Q8_0.gguf
 *         hf download ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF qwen3-reranker-0.6b-q8_0.gguf
 */
public class Search {
    static final String TASK = "Given a question, retrieve passages from the documentation that answer it";
    static final int PASSAGE_CHARS = 800;  // paragraphs are merged up to about this, and longer ones split

    record Passage(Path file, int line, String text) {}

    public static void main(String[] args) throws Exception {
        Path folder = Path.of(args.length > 0 ? args[0] : "docs");
        String query = args.length > 1 ? args[1] : "Why does a long video lose the GPU on Vulkan?";
        int k = Integer.getInteger("k", 5);
        boolean rerank = Boolean.parseBoolean(System.getProperty("rerank", "true"));

        List<Passage> passages = passages(folder);
        if (passages.isEmpty()) {
            System.err.println("no .txt or .md files under " + folder);
            System.exit(1);
        }
        try (var emb = new LlamaModel(Weights.QWEN3_EMBEDDING_0_6B, Mode.EMBED)) {
            long t0 = System.nanoTime();
            float[][] vectors = emb.embed(passages.stream().map(Passage::text).toList());
            double s = (System.nanoTime() - t0) / 1e9;
            System.out.printf("[fornax] embedded %d passages from %s in %.2fs (%.0f a second)%n",
                    passages.size(), folder, s, passages.size() / s);

            float[] q = emb.embed(List.of(query), EmbedOptions.query(TASK))[0];
            List<Scored> nearest = Vectors.topK(q, vectors, rerank ? Math.max(k, 20) : k), hits = nearest;
            if (rerank) {
                try (var rr = new LlamaModel(Weights.QWEN3_RERANKER_0_6B, Mode.RERANK)) {
                    long r0 = System.nanoTime();
                    List<Scored> reranked = rr.rerank(query, nearest.stream().map(h -> passages.get(h.index()).text()).toList());
                    System.out.printf("[fornax] reranked the nearest %d in %.2fs%n", nearest.size(), (System.nanoTime() - r0) / 1e9);
                    // a reranked index is a position in nearest; map it back to the passage
                    hits = reranked.stream().map(r -> new Scored(nearest.get(r.index()).index(), r.score())).toList();
                }
            }
            System.out.println("\n" + query + "\n");
            for (Scored h : hits.subList(0, Math.min(k, hits.size()))) {
                Passage p = passages.get(h.index());
                String text = p.text().replaceAll("\\s+", " ");
                System.out.printf("%.3f  %s:%d%n       %s%n%n", h.score(), folder.relativize(p.file()), p.line(),
                        text.length() > 300 ? text.substring(0, 300) + "..." : text);
            }
        }
    }

    /** Every .txt and .md file's paragraphs, merged into passages of about PASSAGE_CHARS. */
    static List<Passage> passages(Path folder) throws IOException {
        List<Path> files;
        try (Stream<Path> s = Files.walk(folder)) {
            files = s.filter(Files::isRegularFile)
                    .filter(f -> f.toString().endsWith(".txt") || f.toString().endsWith(".md")).sorted().toList();
        }
        List<Passage> out = new ArrayList<>();
        for (Path f : files) {
            List<String> lines;
            try { lines = Files.readAllLines(f); }
            catch (MalformedInputException e) { continue; }  // not UTF-8 text
            StringBuilder cur = new StringBuilder();
            int start = 1;
            for (int i = 0; i <= lines.size(); i++) {
                String line = i < lines.size() ? lines.get(i) : "";
                // a passage ends at a blank line once it's long enough, before it gets too long, and at the end
                if (i == lines.size() || line.isBlank() && cur.length() >= PASSAGE_CHARS || cur.length() + line.length() > 2 * PASSAGE_CHARS) {
                    if (!cur.toString().isBlank()) out.add(new Passage(f, start, cur.toString().strip()));
                    cur.setLength(0);
                }
                if (i == lines.size()) break;
                if (cur.isEmpty() && line.isBlank()) continue;  // so a passage's line is its first text, not a blank before it
                for (; line.length() > 2 * PASSAGE_CHARS; line = line.substring(2 * PASSAGE_CHARS))  // one very long line
                    out.add(new Passage(f, i + 1, line.substring(0, 2 * PASSAGE_CHARS)));
                if (cur.isEmpty()) start = i + 1;
                cur.append(line).append('\n');
            }
        }
        return out;
    }
}
