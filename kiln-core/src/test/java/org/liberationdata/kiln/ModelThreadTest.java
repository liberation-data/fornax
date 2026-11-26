package org.liberationdata.kiln;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Flow;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.*;

/** ModelThread with a fake engine: a loop that emits tokens and polls the call, as Kiln's upcalls do. */
class ModelThreadTest {

    /** Emits up to n tokens, one a millisecond, until asked to stop; returns how many it emitted. */
    static int generate(Call call, int n, Consumer<String> sink) {
        int i = 0;
        for (; i < n && !call.stopRequested(); i++) {
            try {
                if (sink != null) sink.accept("t" + i);
            } catch (Throwable t) {  // what every upcall does: record, stop, return normally
                call.fail(t);
                break;
            }
            sleep(1);
        }
        return i;
    }

    static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException e) { throw new RuntimeException(e); }
    }

    /** A subscriber that records what it gets; requests everything unless told otherwise. */
    static class Recorder implements Flow.Subscriber<String> {
        final List<String> items = Collections.synchronizedList(new ArrayList<>());
        final CountDownLatch done = new CountDownLatch(1);
        final long initialRequest;
        final long delayMs;
        final int cancelAfter;
        volatile Flow.Subscription subscription;
        volatile Throwable error;
        volatile boolean completed;

        Recorder() { this(Long.MAX_VALUE, 0, -1); }

        Recorder(long initialRequest, long delayMs, int cancelAfter) {
            this.initialRequest = initialRequest; this.delayMs = delayMs; this.cancelAfter = cancelAfter;
        }

        @Override public void onSubscribe(Flow.Subscription s) {
            subscription = s;
            if (initialRequest > 0) s.request(initialRequest);
        }

        @Override public void onNext(String item) {
            items.add(item);
            if (delayMs > 0) sleep(delayMs);
            if (cancelAfter > 0 && items.size() == cancelAfter) subscription.cancel();
        }

        @Override public void onError(Throwable t) { error = t; done.countDown(); }

        @Override public void onComplete() { completed = true; done.countDown(); }

        boolean await() throws InterruptedException { return done.await(10, TimeUnit.SECONDS); }
    }

    static Throwable cause(CompletableFuture<?> f) {
        ExecutionException e = assertThrows(ExecutionException.class, () -> f.get(10, TimeUnit.SECONDS));
        return e.getCause();
    }

    @Test
    void asyncCallsRunOnTheModelThreadInOrder() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            List<String> threads = Collections.synchronizedList(new ArrayList<>());
            List<Integer> order = Collections.synchronizedList(new ArrayList<>());
            List<CompletableFuture<Integer>> fs = new ArrayList<>();
            for (int i = 0; i < 5; i++) {
                int n = i;
                fs.add(t.submit(call -> { threads.add(Thread.currentThread().getName()); order.add(n); return n; }));
            }
            for (int i = 0; i < 5; i++) assertEquals(i, fs.get(i).get(10, TimeUnit.SECONDS));
            assertEquals(List.of(0, 1, 2, 3, 4), order);
            assertTrue(threads.stream().allMatch("kiln-fake"::equals), threads.toString());
        }
    }

    /** How deep this thread can recurse before its stack overflows. */
    static int depth() {
        try { return 1 + depth(); } catch (StackOverflowError e) { return 0; }
    }

    @Test
    void theModelThreadIsADaemonWithABigStack() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            assertTrue(t.submit(call -> Thread.currentThread().isDaemon()).get(10, TimeUnit.SECONDS),
                    "an unclosed model must not keep the JVM alive");
            int big = t.submit(call -> depth()).get(10, TimeUnit.SECONDS);
            int[] small = new int[1];
            Thread s = Thread.ofPlatform().stackSize(1L << 20).start(() -> small[0] = depth());
            s.join();
            assertTrue(big > 4 * small[0], "8 MB stack reached depth " + big + ", 1 MB reached " + small[0]);
        }
    }

    @Test
    void blockingAndAsyncCallsNeverOverlap() throws Exception {
        AtomicInteger inside = new AtomicInteger(), most = new AtomicInteger();
        try (var t = new ModelThread("fake", () -> {})) {
            java.util.function.Function<Call, Integer> body = call -> {
                most.accumulateAndGet(inside.incrementAndGet(), Math::max);
                sleep(5);
                inside.decrementAndGet();
                return 1;
            };
            List<CompletableFuture<Integer>> fs = new ArrayList<>();
            List<Thread> callers = new ArrayList<>();
            for (int i = 0; i < 10; i++) {
                fs.add(t.submit(body));
                callers.add(Thread.ofVirtual().start(() -> t.run(body)));
            }
            for (var f : fs) f.get(10, TimeUnit.SECONDS);
            for (var c : callers) c.join();
            assertEquals(1, most.get());
        }
    }

    @Test
    void manyVirtualThreadCallersAllFinish() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            int n = 200;
            AtomicInteger sum = new AtomicInteger();
            List<Thread> vts = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                int k = i;
                vts.add(Thread.ofVirtual().start(() -> sum.addAndGet(t.submit(call -> k).join())));
            }
            for (Thread v : vts) v.join();
            assertEquals(n * (n - 1) / 2, sum.get());
        }
    }

    @Test
    void aThrowingCallbackSurfacesFromTheBlockingCall() {
        try (var t = new ModelThread("fake", () -> {})) {
            var boom = new IllegalStateException("consumer failed");
            var thrown = assertThrows(IllegalStateException.class,
                    () -> t.run(call -> generate(call, 100, piece -> { if (piece.equals("t3")) throw boom; })));
            assertSame(boom, thrown);
            assertEquals(100, (int) t.run(call -> generate(call, 100, null)), "the model works afterwards");
        }
    }

    @Test
    void aCheckedFailureIsWrapped() {
        try (var t = new ModelThread("fake", () -> {})) {
            var checked = new java.io.IOException("disk");
            var thrown = assertThrows(RuntimeException.class, () -> t.run(call -> { call.fail(checked); return 0; }));
            assertSame(checked, thrown.getCause());
        }
    }

    @Test
    void theFirstCallbackFailureWinsAndLaterOnesAreSuppressed() {
        try (var t = new ModelThread("fake", () -> {})) {
            var first = new IllegalStateException("first");
            var second = new IllegalArgumentException("second");
            var thrown = assertThrows(IllegalStateException.class, () -> t.run(call -> {
                call.fail(first);
                call.fail(second);
                call.fail(first);  // the same one again isn't added to itself
                return 0;
            }));
            assertSame(first, thrown);
            assertArrayEquals(new Throwable[] {second}, thrown.getSuppressed());
            var oom = new OutOfMemoryError("an Error is rethrown as itself");
            assertSame(oom, assertThrows(OutOfMemoryError.class, () -> t.run(call -> { call.fail(oom); return 0; })));
        }
    }

    @Test
    void streamsEveryItemThenCompletes() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            var r = new Recorder();
            int n = t.<String, Integer>submit((call, sink) -> generate(call, 50, sink), r).get(10, TimeUnit.SECONDS);
            assertTrue(r.await());
            assertEquals(50, n);
            assertEquals(50, r.items.size());
            assertEquals("t0", r.items.get(0));
            assertTrue(r.completed);
            assertNotEquals("kiln-fake", Thread.currentThread().getName());
        }
    }

    @Test
    void aSlowSubscriberDoesNotSlowTheGeneration() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            var r = new Recorder(Long.MAX_VALUE, 20, -1);  // 20 ms an item: 100 items take 2 s to consume
            long start = System.nanoTime();
            int n = t.<String, Integer>submit((call, sink) -> generate(call, 100, sink), r).get(10, TimeUnit.SECONDS);
            long genMs = (System.nanoTime() - start) / 1_000_000;
            int consumedWhenDone = r.items.size();
            assertEquals(100, n);
            assertTrue(genMs < 1500, "generation took " + genMs + " ms");
            assertTrue(consumedWhenDone < 100, "the subscriber had already consumed everything");
            assertTrue(r.await());
            assertEquals(100, r.items.size(), "nothing dropped");
        }
    }

    @Test
    void overflowStopsTheGenerationAndTellsTheSubscriber() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            t.streamOn(r -> Thread.ofVirtual().start(r), 8);
            var r = new Recorder(0, 0, -1);  // never requests anything
            AtomicInteger emitted = new AtomicInteger();
            var f = t.<String, Integer>submit((call, sink) -> {
                int n = generate(call, 1000, sink);
                emitted.set(n);
                return n;
            }, r);
            assertInstanceOf(StreamOverflowException.class, cause(f));
            assertTrue(r.await());
            assertInstanceOf(StreamOverflowException.class, r.error);
            assertTrue(emitted.get() <= 9, "emitted " + emitted.get());
        }
    }

    @Test
    void cancellingTheSubscriptionStopsTheGeneration() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            var r = new Recorder(Long.MAX_VALUE, 0, 5);
            int n = t.<String, Integer>submit((call, sink) -> generate(call, 5000, sink), r).get(10, TimeUnit.SECONDS);
            assertTrue(n < 5000, "ran to the end");
            assertEquals(5, r.items.size());
        }
    }

    @Test
    void aThrowingSubscriberFailsTheCall() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            var boom = new IllegalArgumentException("bad item");
            var r = new Recorder() {
                @Override public void onNext(String item) { if (item.equals("t2")) throw boom; }
            };
            var f = t.<String, Integer>submit((call, sink) -> generate(call, 5000, sink), r);
            assertSame(boom, cause(f));
        }
    }

    @Test
    void cancelStopsTheCallInProgress() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            var started = new CountDownLatch(1);
            var f = t.submit(call -> { started.countDown(); return generate(call, 100_000, null); });
            assertTrue(started.await(5, TimeUnit.SECONDS));
            t.cancel();
            assertTrue(f.get(10, TimeUnit.SECONDS) < 100_000);
        }
    }

    @Test
    void cancellingTheFutureStopsItsCallAndOnlyIt() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            var started = new CountDownLatch(1);
            AtomicInteger emitted = new AtomicInteger();
            var r = new Recorder();
            var f = t.<String, Integer>submit((call, sink) -> {
                started.countDown();
                int n = generate(call, 100_000, sink);
                emitted.set(n);
                return n;
            }, r);
            var next = t.submit(call -> generate(call, 10, null));
            assertTrue(started.await(5, TimeUnit.SECONDS));
            assertTrue(f.cancel(true));
            assertEquals(10, next.get(10, TimeUnit.SECONDS), "the queued call still ran, uncancelled");
            assertTrue(emitted.get() < 100_000);
            assertTrue(r.await());
            assertInstanceOf(CancellationException.class, r.error);
        }
    }

    @Test
    void aTimeoutCancelsTheCall() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            AtomicInteger emitted = new AtomicInteger(-1);
            var f = t.submit(call -> { int n = generate(call, 100_000, null); emitted.set(n); return n; })
                    .orTimeout(100, TimeUnit.MILLISECONDS);
            assertInstanceOf(java.util.concurrent.TimeoutException.class, cause(f));
            t.submit(call -> 0).get(10, TimeUnit.SECONDS);  // runs after the timed-out call returned
            assertTrue(emitted.get() >= 0 && emitted.get() < 100_000, "emitted " + emitted.get());
        }
    }

    @Test
    void nativeCancelOnlyReachesTheRunningCall() throws Exception {
        List<String> log = Collections.synchronizedList(new ArrayList<>());
        try (var t = new ModelThread("fake", () -> {}, () -> log.add("cancel"), () -> log.add("reset"))) {
            var started = new CountDownLatch(1);
            var running = t.submit(call -> { started.countDown(); return generate(call, 100_000, null); });
            var queued = t.submit(call -> 1);
            assertTrue(started.await(5, TimeUnit.SECONDS));
            queued.cancel(true);     // not running: no native cancel
            assertEquals(List.of("reset"), log);
            t.cancel();
            running.get(10, TimeUnit.SECONDS);
            assertEquals(List.of("reset", "cancel"), log);
            assertTrue(queued.isCancelled());
        }
    }

    @Test
    void futuresCompleteOffTheModelThread() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            var started = new CountDownLatch(1);
            var gate = new CountDownLatch(1);
            var f = t.submit(call -> { started.countDown(); awaitQuietly(gate); return 1; });
            var where = f.thenApply(v -> Thread.currentThread().getName());
            gate.countDown();
            assertNotEquals("kiln-fake", where.get(10, TimeUnit.SECONDS));
        }
    }

    static void awaitQuietly(CountDownLatch l) {
        try { l.await(); } catch (InterruptedException e) { throw new RuntimeException(e); }
    }

    @Test
    void closeIsIdempotent() {
        AtomicInteger destroys = new AtomicInteger();
        var t = new ModelThread("fake", destroys::incrementAndGet);
        t.close();
        t.close();
        assertEquals(1, destroys.get());
        assertThrows(IllegalStateException.class, () -> t.run(call -> 1));
        assertInstanceOf(IllegalStateException.class, cause(t.submit(call -> 1)));
    }

    @Test
    void closeDuringACallCancelsItAndFreesAfterItReturns() throws Exception {
        List<String> log = Collections.synchronizedList(new ArrayList<>());
        var t = new ModelThread("fake", () -> log.add("destroy"));
        var started = new CountDownLatch(1);
        var running = t.submit(call -> {
            started.countDown();
            int n = generate(call, 100_000, null);
            log.add("returned");
            return n;
        });
        var queued = t.submit(call -> { log.add("queued ran"); return 0; });
        assertTrue(started.await(5, TimeUnit.SECONDS));
        t.close();
        assertEquals(List.of("returned", "destroy"), log);
        assertTrue(running.get(10, TimeUnit.SECONDS) < 100_000);
        assertThrows(CancellationException.class, () -> queued.get(10, TimeUnit.SECONDS));  // get() throws it unwrapped
    }

    /** A queued call's own failure handler closing the model, the "close when done" idiom, mustn't hang close. */
    @Test
    void closeFromADependentOfAQueuedCallDoesNotDeadlock() throws Exception {
        AtomicInteger destroys = new AtomicInteger();
        var t = new ModelThread("fake", destroys::incrementAndGet);
        var started = new CountDownLatch(1);
        var running = t.submit(call -> { started.countDown(); return generate(call, 100_000, null); });
        var queued = t.submit(call -> 0);
        var dependentClosed = new CountDownLatch(1);
        queued.whenComplete((v, e) -> { t.close(); dependentClosed.countDown(); });
        assertTrue(started.await(5, TimeUnit.SECONDS));
        var closer = Thread.ofPlatform().daemon(true).start(t::close);
        closer.join(10_000);
        assertFalse(closer.isAlive(), "close() hung: the queued call's dependent ran on the closing thread and waited for it");
        assertTrue(dependentClosed.await(5, TimeUnit.SECONDS));
        assertEquals(1, destroys.get());
        assertTrue(running.get(10, TimeUnit.SECONDS) < 100_000);
        assertThrows(CancellationException.class, () -> queued.get(10, TimeUnit.SECONDS));
    }

    @Test
    void closeDuringABlockingCallWaitsForIt() throws Exception {
        List<String> log = Collections.synchronizedList(new ArrayList<>());
        var t = new ModelThread("fake", () -> log.add("destroy"));
        var started = new CountDownLatch(1);
        var caller = Thread.ofPlatform().start(() -> t.run(call -> {
            started.countDown();
            generate(call, 100_000, null);
            log.add("returned");
            return 0;
        }));
        assertTrue(started.await(5, TimeUnit.SECONDS));
        t.close();
        caller.join();
        assertEquals(List.of("returned", "destroy"), log);
    }

    /** The model's thread, once it has taken a job off the queue and is waiting for the engine lock. */
    static Thread workerWaitingForTheEngine(String name) throws InterruptedException {
        for (int i = 0; i < 5000; i++) {
            for (var e : Thread.getAllStackTraces().entrySet()) {
                if (!e.getKey().getName().equals("kiln-" + name)) continue;
                for (var frame : e.getValue()) if (frame.getClassName().endsWith("ModelThread$Job")) return e.getKey();
            }
            Thread.sleep(1);
        }
        throw new AssertionError("the model thread never picked up the job");
    }

    @Test
    void closeFailsAnAsyncCallThatLeftTheQueueButHadNotStartedWithCancellation() throws Exception {
        var t = new ModelThread("dequeued", () -> {});
        var started = new CountDownLatch(1);
        var blocking = Thread.ofPlatform().start(() -> t.run(call -> { started.countDown(); return generate(call, 100_000, null); }));
        assertTrue(started.await(5, TimeUnit.SECONDS));
        var waiting = t.submit(call -> 1);
        workerWaitingForTheEngine("dequeued");  // out of the queue, behind the blocking call for the lock
        t.close();
        blocking.join();
        assertThrows(CancellationException.class, () -> waiting.get(10, TimeUnit.SECONDS));
    }

    /** A future cancelled once its job has left the queue, while it waits for the engine, never runs its body. */
    @Test
    void aCallCancelledWhileWaitingForTheEngineNeverRuns() throws Exception {
        List<String> log = Collections.synchronizedList(new ArrayList<>());
        try (var t = new ModelThread("waiting", () -> {}, null, () -> log.add("reset"))) {
            var started = new CountDownLatch(1);
            var release = new CountDownLatch(1);
            var blocking = Thread.ofPlatform().start(() -> t.run(call -> { started.countDown(); awaitQuietly(release); return 0; }));
            assertTrue(started.await(5, TimeUnit.SECONDS));
            var waiting = t.submit(call -> { log.add("ran"); return 1; });
            workerWaitingForTheEngine("waiting");
            assertTrue(waiting.cancel(true));
            release.countDown();
            blocking.join();
            assertEquals(2, t.submit(call -> 2).get(10, TimeUnit.SECONDS));
            assertEquals(List.of("reset", "reset"), log, "the cancelled call neither ran nor reset the engine");
        }
    }

    /** From inside a call, an async call is queued rather than refused: it runs once the outer call returns. */
    @Test
    void anAsyncCallFromInsideACallRunsAfterIt() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            List<String> order = Collections.synchronizedList(new ArrayList<>());
            var inner = t.run(call -> {
                var f = t.submit(c -> { order.add("inner"); return 1; });
                order.add("outer");
                return f;
            });
            assertEquals(1, inner.get(10, TimeUnit.SECONDS));
            assertEquals(List.of("outer", "inner"), order);
        }
    }

    @Test
    void futuresStillCompleteWhenTheStreamExecutorRejects() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            assertThrows(IllegalArgumentException.class, () -> t.streamOn(Runnable::run, 0));
            t.streamOn(r -> { throw new java.util.concurrent.RejectedExecutionException("shut down"); }, 8);
            assertEquals(1, t.submit(call -> 1).get(10, TimeUnit.SECONDS));
            assertInstanceOf(IllegalStateException.class, cause(t.submit(call -> { throw new IllegalStateException("x"); })));
        }
    }

    @Test
    void concurrentClosesFreeOnceAndAllWait() throws Exception {
        AtomicInteger destroys = new AtomicInteger();
        var freed = new CountDownLatch(1);
        var t = new ModelThread("fake", () -> { sleep(100); destroys.incrementAndGet(); freed.countDown(); });
        List<Thread> closers = new ArrayList<>();
        List<Boolean> sawFreed = Collections.synchronizedList(new ArrayList<>());
        for (int i = 0; i < 4; i++) closers.add(Thread.ofVirtual().start(() -> { t.close(); sawFreed.add(freed.getCount() == 0); }));
        for (Thread c : closers) c.join();
        assertEquals(1, destroys.get());
        assertEquals(List.of(true, true, true, true), sawFreed);
    }

    @Test
    void closeFromInsideACallIsRefused() {
        var t = new ModelThread("fake", () -> {});
        assertThrows(IllegalStateException.class, () -> t.run(call -> { t.close(); return 0; }));
        assertFalse(t.isClosed());
        t.close();
    }

    @Test
    void aCallFromInsideACallIsRefused() {
        try (var t = new ModelThread("fake", () -> {})) {
            assertThrows(IllegalStateException.class, () -> t.run(call -> t.run(inner -> 0)));
        }
    }

    @Test
    void aStreamSubscribedAfterCloseGetsAnError() throws Exception {
        var t = new ModelThread("fake", () -> {});
        t.close();
        var r = new Recorder();
        t.<String>publisher((call, sink) -> generate(call, 10, sink)).subscribe(r);
        assertTrue(r.await());
        assertInstanceOf(IllegalStateException.class, r.error);
    }

    @Test
    void eachSubscriptionToAPublisherRunsTheCallOnce() throws Exception {
        try (var t = new ModelThread("fake", () -> {})) {
            AtomicInteger runs = new AtomicInteger();
            Flow.Publisher<String> p = t.publisher((call, sink) -> { runs.incrementAndGet(); return generate(call, 3, sink); });
            var a = new Recorder();
            var b = new Recorder();
            p.subscribe(a);
            p.subscribe(b);
            assertTrue(a.await() && b.await());
            assertEquals(2, runs.get());
            assertEquals(List.of("t0", "t1", "t2"), a.items);
            assertEquals(List.of("t0", "t1", "t2"), b.items);
        }
    }

    @Test
    void joinUnwrapsToCompletionException() {
        try (var t = new ModelThread("fake", () -> {})) {
            var e = assertThrows(CompletionException.class, () -> t.submit(call -> { throw new IllegalStateException("x"); }).join());
            assertInstanceOf(IllegalStateException.class, e.getCause());
        }
    }
}
