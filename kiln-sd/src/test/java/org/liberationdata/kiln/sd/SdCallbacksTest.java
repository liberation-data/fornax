package org.liberationdata.kiln.sd;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.ModelThread;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Sd's log and progress upcalls, invoked directly as stable-diffusion.cpp would (no model needed, but the native
 * library is: skipped without it). A sink that throws must neither escape into native code nor be lost.
 */
class SdCallbacksTest {

    static void requireNatives() {
        try { Class.forName(Sd.class.getName()); }
        catch (Throwable t) { assumeTrue(false, "libstable-diffusion not found: " + t); }
    }

    @AfterEach
    void quiet() {
        try {
            Sd.setProgressSink((s, n, t) -> {});
            Sd.setLogSink((l, t) -> {});
        } catch (Throwable ignored) {}
    }

    @Test
    void aThrowingProgressSinkFailsTheCallAndStopsIt() {
        requireNatives();
        var boom = new IllegalStateException("progress sink broke");
        Sd.setProgressSink((step, steps, time) -> { throw boom; });
        List<String> cancels = new ArrayList<>();
        try (var t = new ModelThread("sd", () -> {}, () -> cancels.add("cancel"), null)) {
            var thrown = assertThrows(IllegalStateException.class, () -> t.run(call -> {
                Sd.CALL.set(call);
                try {
                    Sd.onProgress(1, 20, 0.5f, MemorySegment.NULL);  // returns normally: nothing escapes
                    assertTrue(call.stopRequested());
                    return null;
                } finally { Sd.CALL.remove(); }
            }));
            assertSame(boom, thrown);
            assertFalse(cancels.isEmpty(), "the engine was asked to stop");
        }
    }

    @Test
    void aThrowingLogSinkFailsTheCall() {
        requireNatives();
        var boom = new IllegalArgumentException("log sink broke");
        Sd.setLogSink((level, text) -> { throw boom; });
        try (var t = new ModelThread("sd", () -> {}); Arena a = Arena.ofConfined()) {
            MemorySegment text = a.allocateFrom("hello\n");
            assertSame(boom, assertThrows(IllegalArgumentException.class, () -> t.run(call -> {
                Sd.CALL.set(call);
                try { Sd.onLog(3, text, MemorySegment.NULL); return null; } finally { Sd.CALL.remove(); }
            })));
        }
    }

    @Test
    void aThrowingSinkOutsideAGenerationIsReportedNotThrown() {
        requireNatives();
        Sd.setProgressSink((step, steps, time) -> { throw new IllegalStateException("nobody to tell"); });
        assertDoesNotThrow(() -> Sd.onProgress(1, 2, 0f, MemorySegment.NULL));
    }

    @Test
    void progressReassertsACancelThatCameBeforeTheRunStarted() {
        requireNatives();
        List<String> cancels = new ArrayList<>();
        try (var t = new ModelThread("sd", () -> {}, () -> cancels.add("cancel"), null)) {
            t.run(call -> {
                Sd.CALL.set(call);
                try {
                    call.cancel();                                  // as if sd.cpp ignored it, not yet executing
                    Sd.onProgress(1, 20, 0.5f, MemorySegment.NULL); // the first step asks again
                    return null;
                } finally { Sd.CALL.remove(); }
            });
        }
        assertEquals(List.of("cancel", "cancel"), cancels);
    }
}
