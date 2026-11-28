package org.liberationdata.fornax.jitllm;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Wire;
import org.liberationdata.fornax.llama.LlamaModel.Embeddings;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class JitLlmServiceTest {

    /** A worker's embeddings reach the caller with their token count, as in-process ones do. */
    @Test void embeddingsCrossTheWorkerWithTheirTokens() {
        float[][] v = {{0.6f, 0.8f, 0f}, {0f, -1f, 0f}};
        Embeddings back = JitLlmService.decodeEmbeddings(JitLlmService.embeddings(new Embeddings(v, 7)));
        assertEquals(7, back.tokens());
        assertEquals(v.length, back.vectors().length);
        for (int i = 0; i < v.length; i++) assertArrayEquals(v[i], back.vectors()[i]);

        Embeddings none = JitLlmService.decodeEmbeddings(JitLlmService.embeddings(new Embeddings(new float[0][], 0)));
        assertEquals(0, none.vectors().length);
        assertEquals(0, none.tokens());
    }

    /** A chat's options reach the worker whole: --no-think's thinking false too. */
    @Test void chatOptionsCrossTheWorkerWithThinking() {
        Options o = Options.DEFAULT.withMaxTokens(9).withSeed(3).withThinking(false);
        Wire.In in = new Wire.In(JitLlmService.chatArgs(List.of(Message.user("hi")), o, true));
        assertEquals(List.of(Message.user("hi")), JitLlmService.messages(in));
        assertEquals(o, JitLlmService.options(in));
        assertTrue(in.bool());   // stream
    }
}
