package hello;

import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;

import java.util.List;

/** Streams a chat reply from Qwen3 4B Instruct, running on the GPU inside this JVM. */
public class HelloFornax {
    /** A Hugging Face reference: the repo, and the quant to take from it. */
    static final String MODEL = "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M";

    public static void main(String[] args) {
        String prompt = args.length > 0 ? String.join(" ", args) : "Write a haiku about a fornax. Just the haiku.";

        // Downloaded into the Hugging Face cache on first use (2.5 GB), then loaded onto the GPU. The natives jar on the
        // classpath supplies llama.cpp for this OS and GPU.
        try (var llm = new LlamaModel(MODEL)) {
            var reply = llm.chat(List.of(Message.user(prompt)), Options.DEFAULT, System.out::print);
            System.out.printf("%n[%d tokens]%n", reply.generatedTokens());
        }
    }
}
