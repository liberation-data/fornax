package org.liberationdata.kiln.examples;

import org.liberationdata.kiln.HfCache;
import org.liberationdata.kiln.llama.LlamaModel;
import org.liberationdata.kiln.llama.LlamaModel.Message;
import org.liberationdata.kiln.llama.LlamaModel.Options;

import java.util.List;

/**
 * Streams a chat reply from a GGUF model. Usage: Chat ["prompt"] [-Dmodel=/path/to/model.gguf]
 * Default model: hf download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf
 */
public class Chat {
    public static void main(String[] args) {
        String prompt = args.length > 0 ? String.join(" ", args)
                : "Give me five country song titles about a robot with a honky-tonk heart. Just the list.";
        String model = System.getProperty("model", HfCache.require("unsloth/Qwen3-4B-Instruct-2507-GGUF",
                "Qwen3-4B-Instruct-2507-Q4_K_M.gguf").toString());
        long t0 = System.nanoTime();
        try (var llm = new LlamaModel(model)) {
            System.out.printf("[kiln] loaded %s in %.1fs%n", model, (System.nanoTime() - t0) / 1e9);
            long g0 = System.nanoTime();
            var r = llm.chat(List.of(Message.user(prompt)), Options.DEFAULT, System.out::print);
            double s = (System.nanoTime() - g0) / 1e9;
            System.out.printf("%n[kiln] %d prompt + %d generated tokens in %.1fs (%.1f tok/s)%n",
                    r.promptTokens(), r.generatedTokens(), s, r.generatedTokens() / s);
        }
    }
}
