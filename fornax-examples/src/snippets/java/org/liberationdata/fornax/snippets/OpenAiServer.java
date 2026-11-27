package org.liberationdata.fornax.snippets;

// --8<-- [start:start]
import org.liberationdata.fornax.server.FornaxServer;
// --8<-- [end:start]

/** The snippets of website/src/content/docs/docs/guides/openai-server.md and README.md's server block: compiled, never run. */
class OpenAiServer {
    void start() throws Exception {
        // --8<-- [start:start]
        try (var server = FornaxServer.builder().port(54154).model("qwen3-4b").build().start()) {
            System.out.println("OpenAI base URL: " + server.url());   // http://127.0.0.1:54154/v1
            server.await();
        }
        // --8<-- [end:start]
    }

    void readme() throws Exception {
        // --8<-- [start:readme]
        try (var server = FornaxServer.builder().model("qwen3-4b").build().start()) {
            server.await();   // OpenAI base URL: server.url(), http://127.0.0.1:54154/v1
        }
        // --8<-- [end:readme]
    }
}
