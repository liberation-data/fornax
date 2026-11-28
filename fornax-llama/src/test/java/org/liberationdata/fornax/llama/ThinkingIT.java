package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Mode;
import org.liberationdata.fornax.llama.LlamaModel.Options;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Options.withThinking(false) against Qwen3 0.6B, which thinks unless its template is told not to: in process, in an
 * isolated worker and batched, the reply comes without reasoning, streamed as it is returned. Skipped unless the model
 * is in the Hugging Face cache (or at -Dfornax.test.smallGguf) and the native library is built.
 */
class ThinkingIT {
    static final List<Message> ASK = List.of(Message.user("Reply with exactly the word: fornax"));
    static final Options OFF = Options.DEFAULT.withTemperature(0).withMaxTokens(64).withThinking(false);

    static String gguf() {
        var p = LlamaIsolatedIT.gguf();
        assumeTrue(p.isPresent(), "model not downloaded: hf download " + LlamaIsolatedIT.REPO + " " + LlamaIsolatedIT.FILE);
        return p.get().toString();
    }

    @Test
    void inProcessTheReplyComesWithoutReasoning() {
        try (var m = new LlamaModel(gguf(), -1, 2048)) {
            // the switch's default leaves Qwen3 thinking: its <think> block is in the reply
            assertTrue(m.chat(ASK, Options.DEFAULT.withTemperature(0).withMaxTokens(16)).text().contains("<think>"));
            assertPlainAnswer(m);
        }
    }

    @Test
    void inAWorkerTheReplyComesWithoutReasoning() {
        try (var m = LlamaModel.isolated(gguf(), -1, 2048, Mode.CHAT)) {
            assertPlainAnswer(m);
        }
    }

    @Test
    void batchedTheReplyComesWithoutReasoning() {
        try (var m = new LlamaModel(gguf(), -1, 2048, 2)) {   // two slots: the batcher
            assertPlainAnswer(m);
        }
    }

    private static void assertPlainAnswer(LlamaModel m) {
        var streamed = new StringBuilder();
        var r = m.chat(ASK, OFF, streamed::append);
        assertFalse(r.text().contains("<think>") || r.text().contains("</think>"), r.text());
        assertTrue(r.text().toLowerCase().contains("fornax"), r.text());
        assertEquals(r.text(), streamed.toString());
        assertTrue(r.promptTokens() > 0 && r.generatedTokens() > 0);
    }
}
