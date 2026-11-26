package org.liberationdata.kiln.yue2;

import org.liberationdata.kiln.Audio;
import org.liberationdata.kiln.Call;
import org.liberationdata.kiln.ModelThread;
import org.liberationdata.kiln.NativeLibrary;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

import static java.lang.foreign.ValueLayout.*;

/**
 * YuE2 song generation in-process via yue2.cpp (through libkiln_yue2).
 * Request is yue2.cpp's JSON (style, lyrics, duration, cot, seed, lm_seed, steps, ...); output is 48 kHz stereo.
 * YuE2 weights are CC BY-NC 4.0 (no commercial use). Calls run one at a time, from any thread; see
 * {@link #generateAsync} for virtual threads.
 */
public final class Yue2Model implements AutoCloseable {
    private static final NativeLibrary LIB = NativeLibrary.load("kiln_yue2");

    private static MethodHandle fn(String n, FunctionDescriptor fd) { return LIB.function(n, fd); }

    private static final MethodHandle LAST_ERROR = fn("kiln_yue2_last_error", FunctionDescriptor.of(ADDRESS));
    private static final MethodHandle CREATE = fn("kiln_yue2_create", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT));
    private static final MethodHandle GENERATE = fn("kiln_yue2_generate", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle FREE = fn("kiln_yue2_free", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle DESTROY = fn("kiln_yue2_destroy", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle SET_CANCEL = fn("kiln_yue2_set_cancel", FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT));
    private static final int CANCELLED = 3;  // KILN_YUE2_CANCELLED

    /** abcScore is the ABC-notation score YuE2 writes before rendering audio, when present. */
    public record Result(Audio audio, String resolvedRequestJson, String abcScore) {}

    private static String str(MemorySegment s) { return s.equals(MemorySegment.NULL) ? null : s.reinterpret(Long.MAX_VALUE).getString(0); }

    private static String err() {
        try { return str((MemorySegment) LAST_ERROR.invokeExact()); } catch (Throwable t) { return t.toString(); }
    }

    private final MemorySegment handle;
    private final ModelThread thread;

    public Yue2Model(String modelPath, String vaePath, boolean keepLoaded, int maxSeq) {
        try (Arena a = Arena.ofConfined()) {
            handle = (MemorySegment) CREATE.invokeExact(a.allocateFrom(modelPath), a.allocateFrom(vaePath), keepLoaded ? 1 : 0, maxSeq);
        } catch (Throwable t) { throw new RuntimeException(t); }
        if (handle.equals(MemorySegment.NULL)) throw new IllegalStateException("kiln_yue2_create: " + err());
        MemorySegment h = handle;
        thread = new ModelThread("yue2", () -> destroy(h), () -> setCancel(h, 1), () -> setCancel(h, 0));
    }

    private static void destroy(MemorySegment h) {
        try { DESTROY.invokeExact(h); } catch (Throwable t) { throw new RuntimeException(t); }
    }

    private static void setCancel(MemorySegment h, int on) {
        try { SET_CANCEL.invokeExact(h, on); } catch (Throwable t) { throw new RuntimeException(t); }
    }

    /**
     * Stops the generation in progress, from any thread: yue2.cpp checks between LM tokens, diffusion steps and VAE
     * tiles. That call throws a {@link CancellationException}. A call that hasn't started isn't affected: cancel its
     * future instead.
     */
    public void cancel() { thread.cancel(); }

    /** {@link #generate}, on this model's own platform thread. Cancelling the future stops the generation. */
    public CompletableFuture<Result> generateAsync(String requestJson) { return thread.submit(c -> generateOn(c, requestJson)); }

    /** Generates on the calling thread (pinning a virtual thread's carrier meanwhile: see {@link #generateAsync}). */
    public Result generate(String requestJson) { return thread.run(c -> generateOn(c, requestJson)); }

    private Result generateOn(Call call, String requestJson) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment pcmOut = a.allocate(ADDRESS), nOut = a.allocate(JAVA_INT), jsonOut = a.allocate(ADDRESS), scoreOut = a.allocate(ADDRESS);
            int rc = (int) GENERATE.invokeExact(handle, a.allocateFrom(requestJson), pcmOut, nOut, jsonOut, scoreOut);
            if (rc == CANCELLED) throw new CancellationException("yue2 generation was cancelled");
            if (rc != 0) throw new IllegalStateException("kiln_yue2_generate rc=" + rc + ": " + err());
            int n = nOut.get(JAVA_INT, 0);
            MemorySegment pcm = pcmOut.get(ADDRESS, 0).reinterpret((long) n * 8);
            float[] l = pcm.asSlice(0, (long) n * 4).toArray(JAVA_FLOAT);
            float[] r = pcm.asSlice((long) n * 4, (long) n * 4).toArray(JAVA_FLOAT);
            MemorySegment js = jsonOut.get(ADDRESS, 0), sc = scoreOut.get(ADDRESS, 0);
            String json = str(js), score = str(sc);
            FREE.invokeExact(pcm);
            if (!js.equals(MemorySegment.NULL)) FREE.invokeExact(js);
            if (!sc.equals(MemorySegment.NULL)) FREE.invokeExact(sc);
            return new Result(new Audio(l, r, 48000), json, score);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    /**
     * Frees the model. Idempotent, and safe while a call is in flight on another thread: it cancels that call, fails
     * queued async calls with a CancellationException, waits for the call to return, then frees.
     */
    @Override public void close() { thread.close(); }
}
