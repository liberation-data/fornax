package org.liberationdata.fornax.server;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.liberationdata.fornax.ModelThread;
import org.liberationdata.fornax.server.Engines.Stopper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

/**
 * A chat on a model with slots, as the server runs it ({@link PooledEngines#chatInSlot}), against a ModelThread with
 * lanes standing in for the model: an engine that streams numbered pieces and, stopped, ends with what it has, as the
 * batched llama.cpp shim does. No model is loaded.
 */
@Timeout(20)
class ChatInSlotTest {
    final ModelThread thread = new ModelThread("fake", 2, () -> {});

    @AfterEach void close() { thread.close(); }

    /** Streams up to n pieces ("0", "1", ...), then answers "all"; stopped, answers how many it had sent. */
    String chat(int n, java.util.function.Consumer<String> onContent, Stopper stop) {
        return PooledEngines.chatInSlot(sub -> thread.<String, String>submit((c, sink) -> {
            for (int i = 0; i < n; i++) {
                if (c.stopRequested()) return "stopped after " + i;
                sink.accept(String.valueOf(i));
            }
            return "all";
        }, sub), onContent, stop);
    }

    @Test void everyPieceIsDeliveredBeforeTheReplyIsReturned() {
        // the SSE stream writes its last chunk once chat returns: content still in the buffer then would be lost
        List<String> got = Collections.synchronizedList(new ArrayList<>());
        String r = chat(40, p -> {
            try { Thread.sleep(5); } catch (InterruptedException e) { throw new RuntimeException(e); }
            got.add(p);
        }, new Stopper());
        assertEquals("all", r);
        assertEquals(40, got.size(), "pieces delivered by the time it returned");
        assertEquals("39", got.getLast());
    }

    @Test void aStopFromTheContentConsumerEndsWithTheTextSoFarAsTheSerialPathDoes() {
        // a stop sequence: FornaxServer stops from onContent and then wants the engine's reply, not a CancellationException
        Stopper stop = new Stopper();
        List<String> got = Collections.synchronizedList(new ArrayList<>());
        AtomicBoolean afterStop = new AtomicBoolean();
        String r = chat(100_000, p -> {
            if (stop.requested()) afterStop.set(true);
            got.add(p);
            if (p.equals("3")) stop.stop();
        }, stop);
        assertTrue(r.startsWith("stopped after"), r);
        assertFalse(afterStop.get(), "no piece after the stop: " + got);
    }

    @Test void aStopBeforeTheChatStartsEndsItAtOnce() {
        Stopper stop = new Stopper();
        stop.stop();   // the client went before the model was attached
        List<String> got = Collections.synchronizedList(new ArrayList<>());
        String r = chat(100_000, got::add, stop);
        assertTrue(r.startsWith("stopped after"), r);
        assertTrue(got.isEmpty(), got.toString());
    }

    @Test void aConsumerThatThrowsBecauseTheClientHasGoneFailsTheChatWithThat() {
        // Sse.data throws a CancellationException for a client that has gone: FornaxServer answers 499
        assertThrows(CancellationException.class, () -> chat(100_000, p -> { throw new CancellationException("the client has gone"); },
                new Stopper()));
    }

    @Test void theEnginesRefusalIsTheSameApiErrorAsTheSerialPath() {
        ApiError e = assertThrows(ApiError.class, () -> PooledEngines.chatInSlot(sub -> thread.<String, String>submit((c, sink) -> {
            throw new IllegalStateException("llama chat failed (rc=2): the prompt is 9000 tokens, more than the context");
        }, sub), null, new Stopper()));
        assertEquals(400, e.status);
        assertTrue(e.getMessage().contains("more than the context"), e.getMessage());
    }
}
