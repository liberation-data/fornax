import org.liberationdata.fornax.Scored;
import org.liberationdata.fornax.Vectors;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.EmbedOptions;
import org.liberationdata.fornax.llama.LlamaModel.Mode;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * How well embedders and rerankers find the right page of Fornax's own docs: 16 questions, each with the page that
 * answers it, over the website's docs split into passages of about 800 characters. For each embedder, the rank of the
 * first passage from the answering page; for each reranker, the same after reranking an embedder's nearest 20. The
 * numbers in the embeddings guide come from it.
 *
 * <pre>
 * ./gradlew :fornax-core:jar :fornax-llama:jar
 * java --enable-native-access=ALL-UNNAMED -Dfornax.library.path=build/native/linux-x86_64-vulkan \
 *     -cp fornax-core/build/libs/fornax-core-VERSION.jar:fornax-llama/build/libs/fornax-llama-VERSION.jar \
 *     scripts/RetrievalEval.java embed qwen3|gemma MODEL.gguf
 * java ... scripts/RetrievalEval.java rerank QWEN3_EMBEDDING.gguf RERANKER.gguf...
 * </pre>
 * Run from the repository root. -Ddocs=DIR reads another folder of .md and .mdx files.
 */
public class RetrievalEval {
    record Passage(String page, String text) {}

    /** A question, and the page (file name without extension) that answers it. */
    static final String[][] Q = {
        {"How do I install the Vulkan driver on Fedora?", "gpu-drivers"},
        {"How do I separate the vocals from a song?", "stem-separation"},
        {"Transcribe an audio file with word timings", "speech-to-text"},
        {"Make a song from lyrics", "songs"},
        {"Generate a video from a picture", "video"},
        {"Turn text into spoken audio in a chosen voice", "speech"},
        {"Serve the models behind an OpenAI-compatible HTTP API", "openai-server"},
        {"Keep several models loaded and evict them when memory runs short", "model-pool"},
        {"Run a model in a separate JVM so that a native crash doesn't take my app down", "isolation"},
        {"Ask a question about a photo", "vision"},
        {"Add the dependency to a Maven or Gradle build", "maven-and-gradle"},
        {"Should I use HIP or Vulkan on an AMD GPU?", "gpu-backends"},
        {"Semantic search over my documents with embeddings and a reranker", "embeddings"},
        {"How many tokens per second does it generate?", "benchmarks"},
        {"Queue requests from many users with priorities across workers", "dispatch"},
        {"Pull a model from Hugging Face with the command line tool", "cli"},
    };

    public static void main(String[] args) throws Exception {
        if (!(args.length == 3 && args[0].equals("embed") && args[1].matches("qwen3|gemma")
                || args.length >= 3 && args[0].equals("rerank"))) {
            System.err.println("usage: RetrievalEval embed qwen3|gemma MODEL.gguf");
            System.err.println("       RetrievalEval rerank QWEN3_EMBEDDING.gguf RERANKER.gguf...");
            System.exit(2);
        }
        List<Passage> ps = passages(Path.of(System.getProperty("docs", "website/src/content/docs/docs")));
        if (args[0].equals("embed")) {
            boolean gemma = args[1].equals("gemma");
            try (var m = new LlamaModel(args[2], Mode.EMBED)) {
                long t0 = System.nanoTime();
                float[][] v = m.embed(ps.stream().map(Passage::text).toList(), gemma ? EmbedOptions.gemmaDocument() : EmbedOptions.DEFAULT);
                System.out.printf("%s: %d passages, %d dimensions, %.2fs%n", Path.of(args[2]).getFileName(), ps.size(),
                        v[0].length, (System.nanoTime() - t0) / 1e9);
                List<List<Integer>> ranked = new ArrayList<>();
                for (String[] q : Q) ranked.add(Vectors.topK(query(m, q[0], gemma), v, v.length).stream().map(Scored::index).toList());
                report("embedding", ps, ranked);
            }
        } else {
            List<List<Integer>> nearest = new ArrayList<>();
            try (var m = new LlamaModel(args[1], Mode.EMBED)) {
                float[][] v = m.embed(ps.stream().map(Passage::text).toList());
                for (String[] q : Q) nearest.add(Vectors.topK(query(m, q[0], false), v, 20).stream().map(Scored::index).toList());
            }
            report("embedding only (nearest 20)", ps, nearest);
            for (int r = 2; r < args.length; r++) {
                try (var rr = new LlamaModel(args[r], Mode.RERANK)) {
                    long t0 = System.nanoTime();
                    List<List<Integer>> ranked = new ArrayList<>();
                    for (int i = 0; i < Q.length; i++) {
                        List<Integer> c = nearest.get(i);
                        ranked.add(rr.rerank(Q[i][0], c.stream().map(j -> ps.get(j).text()).toList()).stream()
                                .map(s -> c.get(s.index())).toList());
                    }
                    report(String.format("%s (%.1fs)", Path.of(args[r]).getFileName(), (System.nanoTime() - t0) / 1e9), ps, ranked);
                }
            }
        }
    }

    static float[] query(LlamaModel m, String q, boolean gemma) {
        EmbedOptions o = gemma ? EmbedOptions.gemmaQuery("search result")
                : EmbedOptions.query("Given a question, retrieve passages from the documentation that answer it");
        return m.embed(List.of(q), o)[0];
    }

    /** Mean reciprocal rank, and how often the answering page came first and in the top five. */
    static void report(String what, List<Passage> ps, List<List<Integer>> ranked) {
        double mrr = 0;
        int first = 0, top5 = 0;
        StringBuilder ranks = new StringBuilder();
        for (int i = 0; i < Q.length; i++) {
            List<Integer> r = ranked.get(i);
            int rank = 0;
            for (int k = 0; k < r.size() && rank == 0; k++) if (ps.get(r.get(k)).page().equals(Q[i][1])) rank = k + 1;
            if (rank > 0) mrr += 1.0 / rank;
            if (rank == 1) first++;
            if (rank >= 1 && rank <= 5) top5++;
            ranks.append(rank).append(' ');
        }
        System.out.printf("%-45s MRR %.3f, first %d/%d, top 5 %d/%d, ranks %s%n", what, mrr / Q.length, first, Q.length,
                top5, Q.length, ranks.toString().strip());
    }

    /** Each .md and .mdx file's paragraphs, merged into passages of about 800 characters. */
    static List<Passage> passages(Path dir) throws Exception {
        List<Passage> out = new ArrayList<>();
        try (Stream<Path> s = Files.walk(dir)) {
            for (Path f : s.filter(p -> p.toString().matches(".*\\.mdx?$")).sorted().toList()) {
                String page = f.getFileName().toString().replaceAll("\\.mdx?$", "");
                StringBuilder cur = new StringBuilder();
                for (String para : Files.readString(f).split("\n\\s*\n")) {
                    if (cur.length() + para.length() > 800 && !cur.isEmpty()) {
                        out.add(new Passage(page, cur.toString()));
                        cur.setLength(0);
                    }
                    cur.append(para).append("\n\n");
                }
                if (!cur.isEmpty()) out.add(new Passage(page, cur.toString()));
            }
        }
        return out;
    }
}
