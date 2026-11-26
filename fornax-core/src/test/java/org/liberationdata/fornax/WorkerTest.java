package org.liberationdata.fornax;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The worker process with {@link TestService}: calls, streaming, shared memory, cancel, and recovery from real native
 * crashes (abort, segfault), a hang and the parent's death. No engine, model or GPU.
 */
class WorkerTest {
    static final Worker.Options FAST = Worker.Options.DEFAULT.withBackoff(Duration.ofMillis(50), Duration.ofSeconds(1));
    static final String SERVICE = TestService.class.getName();

    final List<Worker> workers = new ArrayList<>();

    Worker worker(Worker.Options o) {
        Worker w = new Worker("test", o);
        workers.add(w);
        return w;
    }

    @AfterEach
    void closeWorkers() { workers.forEach(Worker::close); }

    static byte[] str(String s) { return new Wire.Out().string(s).toBytes(); }
    static byte[] num(int n) { return new Wire.Out().integer(n).toBytes(); }
    static long pid(Worker.Instance i) { return new Wire.In(i.invoke("pid", null)).longValue(); }

    @Test
    void startsLazilyAndCallsTheService() {
        Worker w = worker(FAST);
        assertFalse(w.isRunning());
        var svc = w.open(SERVICE, null);
        assertTrue(w.isRunning());
        assertEquals("hello", new Wire.In(svc.invoke("echo", str("hello"))).string());
        assertEquals(w.pid(), pid(svc));
        assertNotEquals(ProcessHandle.current().pid(), w.pid());
        assertTrue(w.ping().toMillis() < 1000);
    }

    @Test
    void theWorkersExceptionArrivesAsTheSameClassWithItsStack() {
        var svc = worker(FAST).open(SERVICE, null);
        var e = assertThrows(IllegalArgumentException.class, () -> svc.invoke("throw", str("x")));
        assertEquals("bad argument: x", e.getMessage());
        assertEquals(TestService.class.getName(), e.getStackTrace()[0].getClassName());
        assertThrows(UnsupportedOperationException.class, () -> svc.invoke("nope", null));
        assertEquals("ok", new Wire.In(svc.invoke("echo", str("ok"))).string(), "the service is fine after it threw");
    }

    @Test
    void aServiceThatFailsToOpenFailsOpen() {
        Worker w = worker(FAST);
        var e = assertThrows(IllegalStateException.class, () -> w.open(SERVICE, str("fail")));
        assertEquals("bad config", e.getMessage());
        assertEquals("ok", new Wire.In(w.open(SERVICE, null).invoke("echo", str("ok"))).string());
    }

    @Test
    void eventsStreamInOrderBeforeTheResult() {
        var svc = worker(FAST).open(SERVICE, null);
        List<Integer> got = new ArrayList<>();
        byte[] r = svc.invoke("stream", num(1000), null, b -> got.add(new Wire.In(b).integer()));
        assertEquals("done", new Wire.In(r).string());
        assertEquals(1000, got.size());
        for (int i = 0; i < 1000; i++) assertEquals(i, got.get(i));
    }

    @Test
    void aConsumerThatThrowsStopsTheCallAndItsExceptionSurfaces() {
        var svc = worker(FAST).open(SERVICE, null);
        var boom = new IllegalStateException("consumer failed");
        var e = assertThrows(IllegalStateException.class, () -> svc.invoke("spin", null, null, b -> { throw boom; }));
        assertSame(boom, e);
        assertEquals("ok", new Wire.In(svc.invoke("echo", str("ok"))).string());
    }

    @Test
    void largePayloadsGoThroughSharedMemoryBothWays() throws Exception {
        Worker w = worker(FAST.withSharedMemoryThreshold(1024));
        var svc = w.open(SERVICE, null);
        long sentBefore = Frames.SHARED_PAYLOADS.get();
        byte[] big = TestService.pattern(8 << 20);
        assertArrayEquals(big, svc.invoke("echo", big));                       // 8 MB there and back
        assertArrayEquals(TestService.pattern(3 << 20), svc.invoke("big", num(3 << 20)));
        assertEquals(sentBefore + 1, Frames.SHARED_PAYLOADS.get(), "the parent sent one payload through shared memory");
        assertEquals(2, new Wire.In(svc.invoke("sharedPayloads", null)).longValue(), "and the worker two");
        for (int i = 0; i < 20; i++) assertArrayEquals(big, svc.invoke("echo", big));
        try (var files = Files.list(w.sharedDir())) {
            assertEquals(2, files.count(), "one slot each way, reused for every payload");
        }
        assertEquals("small", new Wire.In(svc.invoke("echo", str("small"))).string());
    }

    /**
     * A reply whose write fails with something other than an IOException, as writing a shared-memory slot does on a
     * full /dev/shm (the fault arrives as an InternalError): the call fails at once rather than waiting forever.
     */
    @Test
    void aReplyThatFailsToWriteFailsTheCallInsteadOfHanging() throws Exception {
        Worker w = worker(FAST.withSharedMemoryThreshold(1024));
        var svc = w.open(SERVICE, null);
        assertArrayEquals(TestService.pattern(512 << 10), svc.invoke("big", num(512 << 10)));  // maps the worker's slot
        try (var files = Files.list(w.sharedDir())) {
            for (var slot : files.filter(p -> p.getFileName().toString().startsWith("w")).toList()) {
                try (var fc = java.nio.channels.FileChannel.open(slot, java.nio.file.StandardOpenOption.WRITE)) {
                    fc.truncate(4096);  // the slot's state survives; writing the payload past it faults
                }
            }
        }
        var e = assertTimeoutPreemptively(Duration.ofSeconds(20),
                () -> assertThrows(RuntimeException.class, () -> svc.invoke("big", num(512 << 10))));
        System.out.println(e);
        assertTrue(e.getMessage().contains("InternalError"), "the failure travels back as an error: " + e);
        assertEquals("ok", new Wire.In(svc.invoke("echo", str("ok"))).string(), "the worker still answers");
        assertEquals(1, w.starts());
    }

    @Test
    void concurrentLargePayloadsUseSeveralSlotsAndFilesOfTheirOwn() throws Exception {
        Worker w = worker(FAST.withSharedMemoryThreshold(1024));
        List<Worker.Instance> svcs = new ArrayList<>();
        for (int i = 0; i < 8; i++) svcs.add(w.open(SERVICE, null));
        List<CompletableFuture<Void>> all = new ArrayList<>();
        for (int i = 0; i < svcs.size(); i++) {
            var svc = svcs.get(i);
            byte[] mine = TestService.pattern((1 << 20) * (i + 1) + i);   // sizes that grow, so slots get replaced
            all.add(CompletableFuture.runAsync(() -> {
                for (int j = 0; j < 10; j++) assertArrayEquals(mine, svc.invoke("echo", mine));
            }));
        }
        for (var f : all) f.get(120, TimeUnit.SECONDS);
        try (var files = Files.list(w.sharedDir())) {
            long n = files.count();
            assertTrue(n <= 2 * Frames.MAX_SLOTS, n + " files left: the one-off ones should be deleted");
        }
    }

    @Test
    void cancelStopsTheCallInTheWorker() {
        var svc = worker(FAST).open(SERVICE, null);
        byte[] r = svc.invoke("spin", null, null, started -> svc.cancel());
        assertEquals("stopped", new Wire.In(r).string());
    }

    /** What an isolated model does: a parent ModelThread whose native cancel is the instance's. */
    @Test
    void throughAModelThreadCancelAndFuturesWorkAsInProcess() throws Exception {
        var svc = worker(FAST).open(SERVICE, null);
        try (var model = new ModelThread("isolated", svc::close, svc::cancel, null)) {
            var started = new CompletableFuture<Void>();
            CompletableFuture<byte[]> f = model.submit(c -> svc.invoke("spin", null, c, b -> started.complete(null)));
            started.get(30, TimeUnit.SECONDS);
            model.cancel();
            assertEquals("stopped", new Wire.In(f.get(30, TimeUnit.SECONDS)).string());

            var started2 = new CompletableFuture<Void>();
            CompletableFuture<byte[]> g = model.submit(c -> svc.invoke("spin", null, c, b -> started2.complete(null)));
            started2.get(30, TimeUnit.SECONDS);
            g.cancel(true);
            assertThrows(CancellationException.class, g::join);
            // the worker's call stopped too: the next one runs at once
            assertEquals("next", new Wire.In(model.submit(c -> svc.invoke("echo", str("next"), c, null)).get(30, TimeUnit.SECONDS)).string());
        }
    }

    /**
     * A cancel that reaches the worker before its call has started there (it was cancelled as it was sent) still
     * stops the call the way the engine's own cancel does, as in-process: the result, not a CancellationException.
     */
    @Test
    void aCancelThatArrivesBeforeTheWorkerStartsTheCallStillReturnsItsResult() {
        var svc = worker(FAST).open(SERVICE, null);
        try (var model = new ModelThread("isolated", svc::close, svc::cancel, null)) {
            for (int i = 0; i < 20; i++) {
                byte[] r = model.run(c -> { c.cancel(); return svc.invoke("spin", null, c, null); });
                assertEquals("stopped", new Wire.In(r).string(), "round " + i);
            }
        }
    }

    // ---- crashes ----

    @Test
    void anAbortFailsTheCallAndTheNextCallRestartsTheWorker() {
        Worker w = worker(FAST);
        var svc = w.open(SERVICE, null);
        long first = pid(svc);
        var e = assertThrows(WorkerCrashedException.class, () -> svc.invoke("abort", null));
        System.out.println(e.getMessage());
        assertEquals(first, e.pid());
        assertEquals(6, e.signal().orElseThrow(), "SIGABRT");
        assertTrue(e.getMessage().contains("SIGABRT"), e.getMessage());
        assertFalse(w.isRunning());
        assertFalse(ProcessHandle.of(first).map(ProcessHandle::isAlive).orElse(false), "the crashed worker is gone");

        assertEquals("again", new Wire.In(svc.invoke("echo", str("again"))).string());  // reopened in a new worker
        assertNotEquals(first, pid(svc));
        assertEquals(2, w.starts());
    }

    @Test
    void aSegfaultInNativeCodeFailsTheCallAndTheNextCallRestartsTheWorker() {
        Worker w = worker(FAST);
        var svc = w.open(SERVICE, null);
        long first = pid(svc);
        var e = assertThrows(WorkerCrashedException.class, () -> svc.invoke("segv", null));
        System.out.println(e.getMessage());
        assertEquals("SIGSEGV", e.fatalSignal(), e.getMessage());
        assertNotEquals(0, e.exitCode(), "the JVM's crash handler ends the process (exit 1, as core dumps are off)");
        assertTrue(e.getMessage().contains("SIGSEGV"), e.getMessage());
        assertFalse(e.lastOutput().isEmpty());

        assertEquals("again", new Wire.In(svc.invoke("echo", str("again"))).string());
        assertNotEquals(first, pid(svc));
    }

    @Test
    void anExitIsACrashToo() {
        var svc = worker(FAST).open(SERVICE, null);
        var e = assertThrows(WorkerCrashedException.class, () -> svc.invoke("exit", null));
        assertEquals(3, e.exitCode());
        assertTrue(e.signal().isEmpty());
        assertEquals("ok", new Wire.In(svc.invoke("echo", str("ok"))).string());
    }

    @Test
    void aCrashFailsEveryCallInFlightAndEveryInstanceReloads() throws Exception {
        Worker w = worker(FAST);
        var spinner = w.open(SERVICE, null);
        var crasher = w.open(SERVICE, null);
        var started = new CompletableFuture<Void>();
        var inFlight = CompletableFuture.supplyAsync(() -> spinner.invoke("spin", null, null, b -> started.complete(null)));
        started.get(30, TimeUnit.SECONDS);
        assertThrows(WorkerCrashedException.class, () -> crasher.invoke("abort", null));
        var e = assertThrows(java.util.concurrent.ExecutionException.class, () -> inFlight.get(30, TimeUnit.SECONDS));
        assertInstanceOf(WorkerCrashedException.class, e.getCause());

        assertEquals("a", new Wire.In(spinner.invoke("echo", str("a"))).string());
        assertEquals("b", new Wire.In(crasher.invoke("echo", str("b"))).string());
        assertEquals(pid(spinner), pid(crasher), "both reopened in the one new worker");
        assertEquals(2, w.starts());
    }

    @Test
    void aServiceThatCrashesAsItOpensFailsOpenWithoutTakingThisJvm() {
        Worker w = worker(FAST);
        var e = assertThrows(WorkerCrashedException.class, () -> w.open(SERVICE, str("abort")));
        assertEquals(6, e.signal().orElseThrow());
        assertEquals("ok", new Wire.In(w.open(SERVICE, null).invoke("echo", str("ok"))).string());
    }

    @Test
    void restartsBackOff() {
        Worker w = worker(Worker.Options.DEFAULT.withBackoff(Duration.ofMillis(400), Duration.ofSeconds(5)));
        var svc = w.open(SERVICE, null);
        assertThrows(WorkerCrashedException.class, () -> svc.invoke("abort", null));
        long start = System.nanoTime();
        assertThrows(WorkerCrashedException.class, () -> svc.invoke("abort", null));   // waited 400 ms first
        long firstRestart = (System.nanoTime() - start) / 1_000_000;
        start = System.nanoTime();
        svc.invoke("echo", null);                                                     // waited 800 ms: two in a row
        long secondRestart = (System.nanoTime() - start) / 1_000_000;
        assertTrue(firstRestart >= 380, "restarted after " + firstRestart + " ms");
        assertTrue(secondRestart >= 780, "restarted after " + secondRestart + " ms");
        start = System.nanoTime();
        assertThrows(WorkerCrashedException.class, () -> svc.invoke("abort", null));
        svc.invoke("echo", null);                                                     // a success reset it: 400 ms
        long afterSuccess = (System.nanoTime() - start) / 1_000_000;
        // Against the first restart, which has the same parts (a worker crashing, a 400 ms wait, a worker starting),
        // rather than a fixed bound that a slow machine misses. Without a reset this would wait 1600 ms, not 400.
        assertTrue(afterSuccess < firstRestart + 600, "a success resets the backoff: " + afterSuccess
                + " ms, against " + firstRestart + " ms for the first 400 ms restart");
    }

    @Test
    void aWorkerThatStopsAnsweringIsKilled() throws Exception {
        Worker w = worker(FAST.withHeartbeat(Duration.ofMillis(100), Duration.ofSeconds(1)));
        var svc = w.open(SERVICE, null);
        long pid = pid(svc);
        assertEquals(0, new ProcessBuilder("kill", "-STOP", Long.toString(pid)).start().waitFor());
        var e = assertThrows(WorkerCrashedException.class, () -> svc.invoke("echo", str("hello?")));
        assertTrue(e.getMessage().contains("stopped answering"), e.getMessage());
        assertEquals(9, e.signal().orElseThrow());
        assertEquals("ok", new Wire.In(svc.invoke("echo", str("ok"))).string());
    }

    /** The worker's reader answers pings while a call keeps the service busy for longer than the timeout. */
    @Test
    void aWorkerBusyInALongCallIsNotKilled() {
        Worker w = worker(FAST.withHeartbeat(Duration.ofMillis(100), Duration.ofSeconds(1)));
        var svc = w.open(SERVICE, null);
        long pid = pid(svc);
        byte[] r = svc.invoke("spin", null, null, started ->
                CompletableFuture.delayedExecutor(3, TimeUnit.SECONDS).execute(svc::cancel));
        assertEquals("stopped", new Wire.In(r).string());
        assertEquals(pid, pid(svc), "the same worker, three heartbeat timeouts later");
        assertEquals(1, w.starts());
    }

    @Test
    void closeEndsTheWorkerAndFailsTheCallInFlight() throws Exception {
        Worker w = worker(FAST);
        var svc = w.open(SERVICE, null);
        long pid = w.pid();
        var started = new CompletableFuture<Void>();
        var failure = new AtomicReference<Throwable>();
        Thread t = Thread.ofVirtual().start(() -> {
            try { svc.invoke("spin", null, null, b -> started.complete(null)); } catch (Throwable e) { failure.set(e); }
        });
        started.get(30, TimeUnit.SECONDS);
        w.close();
        t.join(30_000);
        assertInstanceOf(IllegalStateException.class, failure.get());
        assertFalse(failure.get() instanceof WorkerCrashedException);
        ProcessHandle.of(pid).ifPresent(h -> h.onExit().orTimeout(10, TimeUnit.SECONDS).join());
        assertFalse(ProcessHandle.of(pid).map(ProcessHandle::isAlive).orElse(false), "the worker is gone");
        assertThrows(IllegalStateException.class, () -> svc.invoke("echo", null));
        w.close();  // idempotent
    }

    @Test
    void closingAnInstanceFreesItAndKeepsTheWorker() {
        Worker w = worker(FAST);
        var a = w.open(SERVICE, null);
        var b = w.open(SERVICE, null);
        long pid = w.pid();
        a.close();
        a.close();
        assertThrows(IllegalStateException.class, () -> a.invoke("echo", null));
        assertEquals("b", new Wire.In(b.invoke("echo", str("b"))).string());
        assertEquals(pid, w.pid());
    }

    @Test
    void theWorkerDiesWithItsParent() throws Exception {
        var r = ForkedJvm.run(ForkedMains.WorkerParent.class, List.of());
        long pid = ForkedJvm.value(r.output(), "worker");
        var handle = ProcessHandle.of(pid);
        if (handle.isPresent()) handle.get().onExit().get(15, TimeUnit.SECONDS);
        assertFalse(ProcessHandle.of(pid).map(ProcessHandle::isAlive).orElse(false), "worker " + pid + " outlived its parent");
    }

    @Test
    void isolationIsASetting() {
        assertEquals(Isolation.IN_PROCESS, Isolation.parse(null));
        assertEquals(Isolation.IN_PROCESS, Isolation.parse("inprocess"));
        assertEquals(Isolation.IN_PROCESS, Isolation.parse("in-process"));
        assertEquals(Isolation.PROCESS, Isolation.parse("PROCESS"));
        assertThrows(IllegalArgumentException.class, () -> Isolation.parse("thread"));
    }
}
