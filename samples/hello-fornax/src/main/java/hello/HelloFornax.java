package hello;

import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;

import java.util.List;

/** Streams a chat reply from Qwen3 4B Instruct, running on the GPU inside this JVM. */
public class HelloFornax {
    static final String REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF";
    static final String FILE = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf";

    public static void main(String[] args) {
        String prompt = args.length > 0 ? String.join(" ", args) : "Write a haiku about a fornax.";

        // Fornax downloads nothing: find the weights in the Hugging Face cache.
        var model = HfCache.find(REPO, FILE);
        if (model.isEmpty()) {
            System.err.println("Fetch the model first:\n  hf download " + REPO + " " + FILE);
            System.exit(1);
        }

        // The natives jar on the classpath supplies llama.cpp for this OS and GPU.
        try (var llm = new LlamaModel(model.get().toString())) {
            var reply = llm.chat(List.of(Message.user(prompt)), Options.DEFAULT, System.out::print);
            System.out.printf("%n[%d tokens]%n", reply.generatedTokens());
        }
    }
}
