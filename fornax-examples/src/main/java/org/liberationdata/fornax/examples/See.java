package org.liberationdata.fornax.examples;

import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Image;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Asks a vision language model about an image. Usage: See image.png ["question"]
 * [-Dmodel=qwen3-vl-4b|paddleocr-vl|medgemma-4b, or -Dmodel=/path/model.gguf -Dmmproj=/path/mmproj.gguf]
 * [-Dschema=schema.json: reply with JSON that follows that JSON schema].
 * Default model: hf download Qwen/Qwen3-VL-4B-Instruct-GGUF Qwen3VL-4B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
 */
public class See {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("usage: See image.png [\"question\"] [-Dmodel=qwen3-vl-4b] [-Dschema=schema.json]");
            System.exit(2);
        }
        Path image = Path.of(args[0]);
        String m = System.getProperty("model", "qwen3-vl-4b");
        String question = args.length > 1 ? args[1]
                : m.startsWith("paddleocr") ? "OCR:" : "Describe this image in a few sentences.";
        String schema = System.getProperty("schema");

        long t0 = System.nanoTime();
        try (var llm = m.endsWith(".gguf") ? new LlamaModel(m, System.getProperty("mmproj"), -1, 8192)
                                           : new LlamaModel(Weights.named(m))) {
            System.out.printf("[fornax] loaded %s in %.1fs%n", m, (System.nanoTime() - t0) / 1e9);
            var msg = List.of(Message.user(question, Image.read(image)));
            Options o = Options.DEFAULT.withTemperature(0).withMaxTokens(1024);
            long g0 = System.nanoTime();
            if (schema != null) {
                System.out.println(llm.extract(msg, Files.readString(Path.of(schema)), o));
                System.out.printf("[fornax] answered in %.1fs%n", (System.nanoTime() - g0) / 1e9);
            } else {
                long[] first = {0};
                var r = llm.chat(msg, o, piece -> {
                    if (first[0] == 0) first[0] = System.nanoTime();
                    System.out.print(piece);
                });
                long end = System.nanoTime();
                double ttft = (first[0] - g0) / 1e9, gen = (end - first[0]) / 1e9;
                System.out.printf("%n[fornax] %d prompt tokens (with the image) read in %.2fs; %d tokens generated in %.1fs (%.1f tok/s)%n",
                        r.promptTokens(), ttft, r.generatedTokens(), gen, Math.max(0, r.generatedTokens() - 1) / Math.max(gen, 1e-9));
            }
        }
    }
}
