package org.liberationdata.kiln;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.Dispatch.QoS;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Dispatch over real worker processes hosting {@link TestService} (no engine, model or GPU): routing and affinity,
 * eviction within the budget, priority and aging, cancel while queued, back-pressure, idle unloading, and a real
 * native crash followed by a reroute to the other worker. Each "model" is a TestService instance, size 1.
 */
class DispatchTest {
    static final String SERVICE = TestService.class.getName();

    final List<Dispatch> dispatches = new ArrayList<>();
    final Map<String, Worker.Instance> instances = new ConcurrentHashMap<>();
    final Map<String, Integer> loadsOf = new ConcurrentHashMap<>();

    /** n workers and a budget of {@code budget} models of size 1. */
    Dispatch dispatch(int workers, long budget) { return dispatch(opts(workers, budget)); }

    Dispatch dispatch(Dispatch.Options o) {
        var d = new Dispatch(o);
        dispatches.add(d);
        return d;
    }

    /** No per-worker overhead, aging off unless a test sets it, and quick restarts. */
    static Dispatch.Options opts(int workers, long budget) {
        return Dispatch.Options.of(workers, budget).withWorkerOverhead(0).withAging(null).withWorker(WorkerTest.FAST);
    }

    Dispatch.Model<Worker.Instance> model(Dispatch d, String key) { return model(d, key, ""); }

    Dispatch.Model<Worker.Instance> model(Dispatch d, String key, String config) { return model(d, key, config, 1); }

    Dispatch.Model<Worker.Instance> model(Dispatch d, String key, String config, long size) {
        return d.register(key, size, w -> {
            var i = w.open(SERVICE, WorkerTest.str(config));
            instances.put(key, i);
            loadsOf.merge(key, 1, Integer::sum);
            return i;
        });
    }

    @AfterEach
    void close() { dispatches.forEach(Dispatch::close); }

    static long pid(Dispatch.Model<Worker.Instance> m) { return m.call(QoS.INTERACTIVE, WorkerTest::pid); }

    /** Takes the model's worker until the returned future is completed. Returns once it holds it. */
    static CompletableFuture<Object> hold(Dispatch.Model<Worker.Instance> m) throws InterruptedException {
        var gate = new CompletableFuture<Object>();
        var holding = new CountDownLatch(1);
        m.submit(QoS.INTERACTIVE, i -> { holding.countDown(); return gate; });
        assertTrue(holding.await(30, TimeUnit.SECONDS), "never granted");
        return gate;
    }

    /** A request whose body records its name and returns at once. */
    static CompletableFuture<String> record(Dispatch.Model<Worker.Instance> m, QoS qos, String name, List<String> order) {
        return m.submit(qos, i -> { order.add(name); return CompletableFuture.completedFuture(name); });
    }

    static void awaitQueued(Dispatch d, int interactive, int batch) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (true) {
            var s = d.stats();
            if (s.queuedInteractive() == interactive && s.queuedBatch() == batch) return;
            if (System.nanoTime() > deadline) fail("queued " + s);
            Thread.sleep(5);
        }
    }

    @Test
    void routesEachModelToTheWorkerThatHasItLoaded() {
        var d = dispatch(2, 10);
        var a = model(d, "a");
        var b = model(d, "b");
        assertThrows(IllegalArgumentException.class, () -> model(d, "a"), "one model per key");
        long pa = pid(a), pb = pid(b);
        assertNotEquals(pa, pb, "the second model loads into the idle, empty worker");
        assertEquals(d.worker(0).pid(), pa);
        assertEquals(d.worker(1).pid(), pb);
        for (int n = 0; n < 3; n++) {
            assertEquals(pa, pid(a), "an affinity hit");
            assertEquals(pb, pid(b));
        }
        var s = d.stats();
        assertEquals(2, s.loads());
        assertEquals(6, s.hits());
        assertEquals(Map.of("a", 1, "b", 1), loadsOf, "each loaded once");
        assertEquals(2, s.loadedBytes());
    }

    @Test
    void aMissUnloadsTheLeastRecentlyUsedIdleModelFromAnyWorker() {
        var d = dispatch(2, 2);
        var a = model(d, "a");
        var b = model(d, "b");
        var c = model(d, "c");
        pid(a);
        pid(b);
        pid(a);                                                     // b is now least recently used
        pid(c);
        assertEquals(List.of("a", "c"), d.entries().stream().map(Dispatch.Entry::key).sorted().toList());
        assertEquals(1, d.stats().evictions());
        var e = assertThrows(IllegalStateException.class, () -> instances.get("b").invoke("echo", WorkerTest.str("x")));
        assertTrue(e.getMessage().contains("closed"), e.getMessage());
        pid(b);                                                     // a miss again: it reloads
        assertEquals(2, loadsOf.get("b"));
        assertEquals(2, d.stats().loadedBytes());
    }

    @Test
    void aModelInUseIsNeverUnloadedAndALoadWaitsForRoom() throws Exception {
        var d = dispatch(2, 1);
        var a = model(d, "a");
        var b = model(d, "b");
        var gate = hold(a);
        var later = b.submit(QoS.INTERACTIVE, i -> CompletableFuture.completedFuture(WorkerTest.pid(i)));
        Thread.sleep(300);
        assertFalse(later.isDone(), "b waits: the only model it could unload is in use");
        assertEquals(1, d.stats().queuedInteractive());
        assertNull(loadsOf.get("b"));
        gate.complete(null);
        long pb = later.get(30, TimeUnit.SECONDS);
        assertEquals(d.worker(1).pid(), pb, "then a is unloaded and b loads, in the idle worker");
        assertEquals(List.of("b"), d.entries().stream().map(Dispatch.Entry::key).toList());
    }

    /** Without the reservation, the small batch model would take the room the interactive one is waiting for. */
    @Test
    void aLoadWaitingForRoomHoldsBackLoadsRankedBelowIt() throws Exception {
        var d = dispatch(3, 3);
        var a = model(d, "a", "", 2);
        var big = model(d, "big", "", 2);
        var small = model(d, "small", "", 1);
        var gate = hold(a);
        List<String> order = Collections.synchronizedList(new ArrayList<>());
        var fb = record(big, QoS.INTERACTIVE, "big", order);
        awaitQueued(d, 1, 0);
        var fs = record(small, QoS.BATCH, "small", order);
        Thread.sleep(300);
        assertEquals(List.of(), order, "small would fit now, but big is ranked first and waits for room");
        gate.complete(null);
        fb.get(30, TimeUnit.SECONDS);
        fs.get(30, TimeUnit.SECONDS);                               // small may finish first: big's load takes longer
        assertEquals(List.of("big", "small"), d.entries().stream().map(Dispatch.Entry::key).sorted().toList());
    }

    /** The small load was held back only for the big one: once that is cancelled, it goes, while a is still in use. */
    @Test
    void cancellingTheLoadWaitingForRoomLetsTheLoadsBelowItGo() throws Exception {
        var d = dispatch(3, 3);
        var a = model(d, "a", "", 2);
        var big = model(d, "big", "", 2);
        var small = model(d, "small", "", 1);
        var gate = hold(a);
        List<String> order = Collections.synchronizedList(new ArrayList<>());
        var fb = record(big, QoS.INTERACTIVE, "big", order);
        awaitQueued(d, 1, 0);
        var fs = record(small, QoS.BATCH, "small", order);
        awaitQueued(d, 1, 1);
        assertTrue(fb.cancel(true));
        assertEquals("small", fs.get(30, TimeUnit.SECONDS), "small fits now, and nothing ranked above it is waiting");
        assertFalse(gate.isDone());
        assertEquals(List.of("small"), order);
    }

    /** Room that an unload in progress is freeing goes to the next load as soon as the unload finishes. */
    @Test
    void aLoadWaitingOnAnUnloadInProgressGoesWhenItFinishes() throws Exception {
        var d = dispatch(2, 3);
        var slow = d.register("slow", 2, w -> {
            var i = w.open(SERVICE, null);
            return (AutoCloseable) () -> { Thread.sleep(1500); i.close(); };
        });
        var a = model(d, "a", "", 2);
        var b = model(d, "b", "", 1);
        slow.call(QoS.INTERACTIVE, m -> 0);
        var gate = new CompletableFuture<Object>();
        a.submit(QoS.INTERACTIVE, i -> gate);                       // unloads slow, then holds a
        Thread.sleep(300);
        var fb = b.submit(QoS.INTERACTIVE, i -> CompletableFuture.completedFuture(1));
        assertEquals(1, fb.get(30, TimeUnit.SECONDS), "b fits once slow is unloaded, while a is still in use");
        assertFalse(gate.isDone());
    }

    @Test
    void aModelTooBigForTheBudgetLoadsWhenNothingElseIs() {
        var d = dispatch(1, 1);
        var a = model(d, "a");
        var big = d.register("big", 5, w -> w.open(SERVICE, null));
        pid(a);
        assertTrue(big.call(QoS.BATCH, WorkerTest::pid) > 0);
        assertEquals(List.of("big"), d.entries().stream().map(Dispatch.Entry::key).toList());
    }

    @Test
    void interactiveRequestsGoBeforeBatchOnesInArrivalOrder() throws Exception {
        var d = dispatch(1, 10);
        var a = model(d, "a");
        var gate = hold(a);
        List<String> order = Collections.synchronizedList(new ArrayList<>());
        var fs = List.of(record(a, QoS.BATCH, "b1", order), record(a, QoS.INTERACTIVE, "i1", order),
                record(a, QoS.BATCH, "b2", order), record(a, QoS.INTERACTIVE, "i2", order));
        awaitQueued(d, 2, 2);
        gate.complete(null);
        for (var f : fs) f.get(30, TimeUnit.SECONDS);
        assertEquals(List.of("i1", "i2", "b1", "b2"), order);
    }

    @Test
    void aBatchRequestThatHasWaitedOneAgingPeriodGoesBeforeNewerInteractiveOnes() throws Exception {
        var d = dispatch(opts(1, 10).withAging(Duration.ofMillis(500)));
        var a = model(d, "a");
        var gate = hold(a);
        List<String> order = Collections.synchronizedList(new ArrayList<>());
        var old = record(a, QoS.BATCH, "old batch", order);
        Thread.sleep(700);
        var fresh = record(a, QoS.BATCH, "fresh batch", order);
        var i1 = record(a, QoS.INTERACTIVE, "i1", order);
        var i2 = record(a, QoS.INTERACTIVE, "i2", order);
        awaitQueued(d, 2, 2);
        gate.complete(null);
        for (var f : List.of(old, fresh, i1, i2)) f.get(30, TimeUnit.SECONDS);
        assertEquals(List.of("old batch", "i1", "i2", "fresh batch"), order);
    }

    @Test
    void aRequestCancelledOrTimedOutWhileQueuedNeverRuns() throws Exception {
        var d = dispatch(1, 10);
        var a = model(d, "a");
        var gate = hold(a);
        var ran = new AtomicBoolean();
        var cancelled = a.submit(QoS.BATCH, i -> { ran.set(true); return CompletableFuture.completedFuture(1); });
        var timedOut = a.submit(QoS.INTERACTIVE, i -> { ran.set(true); return CompletableFuture.completedFuture(2); })
                .orTimeout(100, TimeUnit.MILLISECONDS);
        awaitQueued(d, 1, 1);
        assertTrue(cancelled.cancel(true));
        var e = assertThrows(ExecutionException.class, () -> timedOut.get(10, TimeUnit.SECONDS));
        assertInstanceOf(TimeoutException.class, e.getCause());
        awaitQueued(d, 0, 0);
        gate.complete(null);
        assertEquals(7L, (long) a.call(QoS.BATCH, i -> 7L), "the worker is free for the next request");
        assertFalse(ran.get());
        assertThrows(CancellationException.class, cancelled::join);
    }

    @Test
    void cancellingARunningRequestCancelsItsBodysFutureAndFreesTheWorker() throws Exception {
        var d = dispatch(1, 10);
        var a = model(d, "a");
        var inner = new CompletableFuture<Object>();
        var started = new CountDownLatch(1);
        var f = a.submit(QoS.INTERACTIVE, i -> { started.countDown(); return inner; });
        assertTrue(started.await(30, TimeUnit.SECONDS));
        f.cancel(true);
        assertTrue(inner.isCancelled(), "the cancel reached the body's future (a Kiln model stops its generation)");
        assertEquals(3L, (long) a.call(QoS.BATCH, i -> 3L));
    }

    @Test
    void aFullQueueRejectsAtOnce() throws Exception {
        var d = dispatch(opts(1, 10).withMaxQueued(1));
        var a = model(d, "a");
        var gate = hold(a);
        var queued = a.submit(QoS.BATCH, i -> CompletableFuture.completedFuture(1));
        var refused = a.submit(QoS.BATCH, i -> CompletableFuture.completedFuture(2));
        var e = assertThrows(ExecutionException.class, () -> refused.get(1, TimeUnit.SECONDS));
        assertInstanceOf(RejectedExecutionException.class, e.getCause());
        var interactive = a.submit(QoS.INTERACTIVE, i -> CompletableFuture.completedFuture(3));
        gate.complete(null);
        assertEquals(3, interactive.get(30, TimeUnit.SECONDS), "each class has its own limit");
        assertEquals(1, queued.get(30, TimeUnit.SECONDS));
        assertEquals(1, d.stats().rejected());
    }

    @Test
    void aWorkerCrashDropsItsModelsAndTheNextRequestGoesToTheOtherWorker() {
        var d = dispatch(2, 10);
        var a = model(d, "a");
        var b = model(d, "b");
        long pa = pid(a), pb = pid(b);
        long w0 = d.worker(0).pid();
        assertEquals(w0, pa);
        var e = assertThrows(WorkerCrashedException.class, () -> a.call(QoS.INTERACTIVE, i -> i.invoke("abort", null)));
        assertEquals(6, e.signal().orElseThrow());
        assertEquals(1, d.stats().crashes());
        assertEquals(List.of("b"), d.entries().stream().map(Dispatch.Entry::key).toList(), "a went with its worker");
        assertFalse(d.worker(0).isRunning());
        assertEquals(pb, pid(a), "a reloads in the worker that didn't crash, not the one backing off");
        assertEquals(2, loadsOf.get("a"));
        assertEquals(pb, pid(b), "b was untouched");
        assertEquals(1, d.worker(1).starts());
    }

    @Test
    void aCrashFailsTheFutureOfASubmittedRequestToo() throws Exception {
        var d = dispatch(1, 10);
        var a = model(d, "a");
        var f = a.submit(QoS.BATCH, i -> CompletableFuture.supplyAsync(() -> i.invoke("segv", null)));
        var e = assertThrows(ExecutionException.class, () -> f.get(60, TimeUnit.SECONDS));
        assertInstanceOf(WorkerCrashedException.class, e.getCause());
        assertTrue(d.entries().isEmpty());
        assertTrue(pid(a) > 0, "the worker restarts, and the model reloads");
        assertEquals(2, d.worker(0).starts());
    }

    @Test
    void aModelThatFailsToLoadFailsItsRequestAndFreesTheWorker() {
        var d = dispatch(1, 10);
        var bad = model(d, "bad", "fail");
        var good = model(d, "good");
        var e = assertThrows(IllegalStateException.class, () -> pid(bad));
        assertEquals("bad config", e.getMessage());
        assertTrue(pid(good) > 0);
        assertEquals(List.of("good"), d.entries().stream().map(Dispatch.Entry::key).toList());
    }

    @Test
    void idleModelsUnloadAfterTheTimeout() throws Exception {
        var d = dispatch(opts(1, 10).withIdleTimeout(Duration.ofMillis(200)));
        var a = model(d, "a");
        pid(a);
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (!d.entries().isEmpty() && System.nanoTime() < deadline) Thread.sleep(20);
        assertTrue(d.entries().isEmpty());
        assertEquals(1, d.stats().evictions());
        pid(a);
        assertEquals(2, loadsOf.get("a"));
    }

    @Test
    void closeFailsQueuedRequestsAndStopsTheWorkers() throws Exception {
        var d = dispatch(1, 10);
        var a = model(d, "a");
        hold(a);
        var queued = a.submit(QoS.BATCH, i -> CompletableFuture.completedFuture(1));
        awaitQueued(d, 0, 1);
        d.close();
        assertThrows(CancellationException.class, () -> queued.get(10, TimeUnit.SECONDS));
        assertFalse(d.worker(0).isRunning());
        assertThrows(IllegalStateException.class, () -> a.call(QoS.BATCH, i -> 1));
    }

    @Test
    void leasesGiveTheModelBackWhenClosed() throws Exception {
        var d = dispatch(1, 10);
        var a = model(d, "a");
        var lease = a.acquire(QoS.INTERACTIVE);
        var next = a.acquireAsync(QoS.BATCH);
        Thread.sleep(100);
        assertFalse(next.isDone());
        assertTrue(d.entries().get(0).inUse());
        lease.close();
        lease.close();
        try (var l = next.get(30, TimeUnit.SECONDS)) {
            assertEquals(d.worker(0).pid(), WorkerTest.pid(l.model()));
        }
        assertThrows(IllegalStateException.class, lease::model);
    }
}
