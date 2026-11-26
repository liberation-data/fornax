package org.liberationdata.fornax.snippets;

// --8<-- [start:chat]
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;

import java.util.List;
// --8<-- [end:chat]

/** The snippets of website/src/content/docs/docs/index.mdx: compiled, never run. */
class Index {
    void chat() {
        // --8<-- [start:chat]
        try (var llm = new LlamaModel("Qwen3-4B-Instruct-2507-Q4_K_M.gguf")) {   // weights load onto the GPU once
            llm.chat(List.of(Message.user("Name a country song about a robot.")), Options.DEFAULT, System.out::print);
        }
        // --8<-- [end:chat]
    }
}
