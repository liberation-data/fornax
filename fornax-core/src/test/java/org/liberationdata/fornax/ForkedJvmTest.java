package org.liberationdata.fornax;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** What only a fresh JVM can show: an escaping upcall exception kills it, and native calls pin virtual threads. */
class ForkedJvmTest {
    static final List<String> ONE_CARRIER = List.of(
            "-Djdk.virtualThreadScheduler.parallelism=1", "-Djdk.virtualThreadScheduler.maxPoolSize=1");

    @Test
    void anExceptionEscapingAnUpcallKillsTheJvm() throws Exception {
        var r = ForkedJvm.run(ForkedMains.ThrowingUpcall.class, List.of(), "raw");
        assertNotEquals(0, r.exitCode());
        assertFalse(r.output().contains("alive=1"), r.output());
        assertTrue(r.output().contains("consumer failed"), r.output());
    }

    @Test
    void aGuardedUpcallSurfacesTheExceptionAfterTheDowncall() throws Exception {
        var r = ForkedJvm.run(ForkedMains.ThrowingUpcall.class, List.of(), "guarded");
        assertEquals(0, r.exitCode(), r.output());
        assertEquals(1, ForkedJvm.value(r.output(), "surfaced"));
        assertEquals(1, ForkedJvm.value(r.output(), "alive"));
    }

    @Test
    void blockingNativeCallsOnVirtualThreadsStallTheCarrier() throws Exception {
        var r = ForkedJvm.run(ForkedMains.Pinning.class, ONE_CARRIER, "blocking");
        assertEquals(0, r.exitCode(), r.output());
        long gap = ForkedJvm.value(r.output(), "maxGapMs");
        assertTrue(gap >= 250, "the heartbeat kept running (longest gap " + gap + " ms): the calls didn't pin");
    }

    @Test
    void asyncCallsLeaveTheCarrierFree() throws Exception {
        var r = ForkedJvm.run(ForkedMains.Pinning.class, ONE_CARRIER, "async");
        assertEquals(0, r.exitCode(), r.output());
        long gap = ForkedJvm.value(r.output(), "maxGapMs"), total = ForkedJvm.value(r.output(), "totalMs");
        assertTrue(total >= 1100, "the four calls should run one after another, " + total + " ms");
        assertTrue(gap < 100, "the heartbeat stalled for " + gap + " ms");
    }
}
