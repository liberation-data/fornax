package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.ModelThread;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The batch loop with a fake engine (no natives): requests of n tokens each emit "t0".."t(n-1)", one a step, together.
 * It runs under a {@link ModelThread} with as many lanes as slots, as a model with slots does.
 */
class BatcherTest {
    static final int SLOTS = 3;

    /** The fake engine. Its state is touched only from the loop's thread, except what the test reads after. */
    static final class Fake implements Batcher.Engine {
        final Map<Integer, int[]> running = new HashMap<>();   // seq -> {generated, wanted}
        final List<Integer> perStep = Collections.synchronizedList(new ArrayList<>());
        volatile boolean failNextDecode, explode;
        volatile int mostAtOnce;

        int add(int wanted) {
            if (wanted < 0) throw new IllegalStateException("llama chat failed (rc=2): prompt too long");
            for (int seq = 0; seq < SLOTS; seq++) {
                if (!running.containsKey(seq)) { running.put(seq, new int[]{0, wanted}); return seq; }
            }
            throw new AssertionError("an add with every slot busy: the lanes should prevent it");
        }

        final java.util.Set<Integer> cancelled = java.util.concurrent.ConcurrentHashMap.newKeySet();

        @Override public void cancel(int seq) { cancelled.add(seq); }

        @Override public void step(Batcher.Events e) {
            if (explode) throw new IllegalStateException("boom");
            mostAtOnce = Math.max(mostAtOnce, running.size());
            perStep.add(running.size());
            if (failNextDecode) {
                failNextDecode = false;
                for (int seq : new ArrayList<>(running.keySet())) { running.remove(seq); e.done(seq, 3, "llama_process failed: 1"); }
                return;
            }
            for (int seq : new ArrayList<>(running.keySet())) {
                int[] s = running.get(seq);
                boolean stop = cancelled.remove(seq) || s[0] >= s[1];
                if (!stop) stop = e.piece(seq, "t" + s[0]++);
                if (stop || s[0] >= s[1]) { running.remove(seq); e.done(seq, 0, "{\"n\":" + s[0] + "}"); }
            }
            try { Thread.sleep(1); } catch (InterruptedException ignored) {}
        }
    }

    final Fake engine = new Fake();
    final ModelThread[] owner = new ModelThread[1];
    final Batcher batcher = new Batcher("fake", engine, () -> owner[0].markCallbackThread());
    final ModelThread thread = owner[0] = new ModelThread("fake", SLOTS, batcher::close);

    @AfterEach void close() { thread.close(); }

    CompletableFuture<String> submit(int tokens, Consumer<String> onPiece) {
        return thread.submit(c -> batcher.run(c, () -> engine.add(tokens), onPiece));
    }

    @Test void requestsRunTogetherUpToTheSlotsAndTheRestQueue() throws Exception {
        List<CompletableFuture<String>> fs = new ArrayList<>();
        List<List<String>> pieces = new ArrayList<>();
        for (int i = 0; i < 7; i++) {
            List<String> p = Collections.synchronizedList(new ArrayList<>());
            pieces.add(p);
            fs.add(submit(20, p::add));
        }
        for (var f : fs) assertEquals("{\"n\":20}", f.get(10, TimeUnit.SECONDS));
        for (var p : pieces) assertEquals(20, p.size());
        assertEquals("t0", pieces.getFirst().getFirst());
        assertEquals(SLOTS, engine.mostAtOnce, "never more than the slots, and they were all used");
        assertTrue(engine.perStep.stream().anyMatch(n -> n > 1), "requests shared steps");
    }

    @Test void cancellingOneEndsItWithWhatItHasAndTheOthersCarryOn() throws Exception {
        CountDownLatch going = new CountDownLatch(3);
        var cancelled = submit(100_000, p -> going.countDown());
        var other = submit(200, null);
        assertTrue(going.await(5, TimeUnit.SECONDS));
        cancelled.cancel(true);
        assertEquals("{\"n\":200}", other.get(10, TimeUnit.SECONDS));
        assertTrue(cancelled.isCancelled());
        for (int i = 0; i < 200 && !engine.running.isEmpty(); i++) Thread.sleep(5);
        assertTrue(engine.running.isEmpty(), "the cancelled request's slot was freed");
    }

    @Test void aConsumerThatThrowsFailsOnlyItsOwnRequest() throws Exception {
        var bad = submit(50, p -> { if (p.equals("t3")) throw new IllegalArgumentException("consumer broke"); });
        var good = submit(50, null);
        var e = assertThrows(ExecutionException.class, () -> bad.get(10, TimeUnit.SECONDS));
        assertInstanceOf(IllegalArgumentException.class, e.getCause());
        assertEquals("{\"n\":50}", good.get(10, TimeUnit.SECONDS));
    }

    @Test void aRefusedRequestFailsAloneAndTheLoopGoesOn() throws Exception {
        var running = submit(50, null);
        var refused = submit(-1, null);
        var e = assertThrows(ExecutionException.class, () -> refused.get(10, TimeUnit.SECONDS));
        assertTrue(e.getCause().getMessage().contains("prompt too long"));
        assertEquals("{\"n\":50}", running.get(10, TimeUnit.SECONDS));
        assertEquals("{\"n\":5}", submit(5, null).get(10, TimeUnit.SECONDS));
    }

    @Test void aFailedDecodeFailsItsRequestsAndLaterOnesStillRun() throws Exception {
        CountDownLatch going = new CountDownLatch(1);
        var a = submit(100_000, p -> going.countDown());
        assertTrue(going.await(5, TimeUnit.SECONDS));
        engine.failNextDecode = true;
        var e = assertThrows(ExecutionException.class, () -> a.get(10, TimeUnit.SECONDS));
        assertTrue(e.getCause().getMessage().contains("llama_process failed"), e.getCause().getMessage());
        assertEquals("{\"n\":5}", submit(5, null).get(10, TimeUnit.SECONDS));
    }

    @Test void anEngineThatThrowsFailsEveryRequestInsteadOfHangingThem() {
        CountDownLatch going = new CountDownLatch(1);
        var a = submit(100_000, p -> going.countDown());
        var b = submit(100_000, null);
        engine.explode = true;
        for (var f : List.of(a, b)) {
            var e = assertThrows(CompletionException.class, f::join);
            assertTrue(e.getCause().getMessage().contains("batch loop failed"), e.getCause().getMessage());
        }
    }

    @Test void aBlockingCallFromAPieceConsumerIsRefusedNotDeadlocked() throws Exception {
        CompletableFuture<Throwable> nested = new CompletableFuture<>();
        var f = submit(5, p -> {
            if (nested.isDone()) return;
            try { thread.run(c -> "nested"); nested.complete(null); } catch (Throwable t) { nested.complete(t); }
        });
        assertEquals("{\"n\":5}", f.get(10, TimeUnit.SECONDS));
        assertInstanceOf(IllegalStateException.class, nested.get(5, TimeUnit.SECONDS));
    }

    @Test void closeWithRequestsInFlightEndsThemAndStopsTheLoop() {
        List<CompletableFuture<String>> fs = new ArrayList<>();
        for (int i = 0; i < 5; i++) fs.add(submit(100_000, null));
        thread.close();
        // futures complete off the model's threads, just after close returns: with text so far, or cancelled
        for (var f : fs) assertDoesNotThrow(() -> { try { f.get(5, TimeUnit.SECONDS); } catch (ExecutionException | java.util.concurrent.CancellationException ok) {} });
        assertThrows(IllegalStateException.class, () -> thread.run(c -> 1));
    }
}
