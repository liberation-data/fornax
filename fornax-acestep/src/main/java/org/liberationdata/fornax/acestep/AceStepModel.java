package org.liberationdata.fornax.acestep;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.Call;
import org.liberationdata.fornax.ModelThread;
import org.liberationdata.fornax.NativeLibrary;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

import static java.lang.foreign.ValueLayout.*;

/**
 * ACE-Step 1.5 song generation in-process via acestep.cpp (through libfornax_acestep).
 * Request is acestep.cpp's JSON (caption, lyrics, bpm, duration, keyscale, seed, ...); output is 48 kHz stereo.
 * Calls run one at a time, from any thread; see {@link #generateAsync} for virtual threads.
 */
public final class AceStepModel implements AutoCloseable {
    private static final NativeLibrary LIB = NativeLibrary.load("fornax_acestep");

    private static MethodHandle fn(String n, FunctionDescriptor fd) { return LIB.function(n, fd); }

    private static final MethodHandle LAST_ERROR = fn("fornax_acestep_last_error", FunctionDescriptor.of(ADDRESS));
    private static final MethodHandle CREATE = fn("fornax_acestep_create", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT));
    private static final MethodHandle GENERATE = fn("fornax_acestep_generate", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle FREE = fn("fornax_acestep_free", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle DESTROY = fn("fornax_acestep_destroy", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle SET_CANCEL = fn("fornax_acestep_set_cancel", FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT));
    private static final int CANCELLED = 5;  // FORNAX_ACESTEP_CANCELLED

    /** resolvedRequestJson is the request after the LM filled in metadata, lyrics and seeds. */
    public record Result(Audio audio, String resolvedRequestJson) {}

    private static String err() {
        try { return ((MemorySegment) LAST_ERROR.invokeExact()).reinterpret(Long.MAX_VALUE).getString(0); }
        catch (Throwable t) { return t.toString(); }
    }

    private final MemorySegment handle;
    private final ModelThread thread;

    /** lmPath may be null to skip the 5 Hz LM (DiT-only: caption + lyrics drive generation directly). */
    public AceStepModel(String lmPath, String textEncPath, String ditPath, String vaePath, boolean keepLoaded) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment lm = lmPath == null ? MemorySegment.NULL : a.allocateFrom(lmPath);
            handle = (MemorySegment) CREATE.invokeExact(lm,
                    a.allocateFrom(textEncPath), a.allocateFrom(ditPath), a.allocateFrom(vaePath), keepLoaded ? 1 : 0);
        } catch (Throwable t) { throw new RuntimeException(t); }
        if (handle.equals(MemorySegment.NULL)) throw new IllegalStateException("fornax_acestep_create: " + err());
        MemorySegment h = handle;
        thread = new ModelThread("acestep", () -> destroy(h), () -> setCancel(h, 1), () -> setCancel(h, 0));
    }

    private static void destroy(MemorySegment h) {
        try { DESTROY.invokeExact(h); } catch (Throwable t) { throw new RuntimeException(t); }
    }

    private static void setCancel(MemorySegment h, int on) {
        try { SET_CANCEL.invokeExact(h, on); } catch (Throwable t) { throw new RuntimeException(t); }
    }

    /**
     * Stops the generation in progress, from any thread: acestep.cpp checks between LM tokens, diffusion steps and VAE
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
            MemorySegment pcmOut = a.allocate(ADDRESS), nOut = a.allocate(JAVA_INT), jsonOut = a.allocate(ADDRESS);
            int rc = (int) GENERATE.invokeExact(handle, a.allocateFrom(requestJson), pcmOut, nOut, jsonOut);
            if (rc == CANCELLED) throw new CancellationException("acestep generation was cancelled");
            if (rc != 0) throw new IllegalStateException("fornax_acestep_generate rc=" + rc + ": " + err());
            int n = nOut.get(JAVA_INT, 0);
            MemorySegment pcm = pcmOut.get(ADDRESS, 0).reinterpret((long) n * 2 * 4);
            MemorySegment js = jsonOut.get(ADDRESS, 0);
            float[] l = pcm.asSlice(0, (long) n * 4).toArray(JAVA_FLOAT);
            float[] r = pcm.asSlice((long) n * 4, (long) n * 4).toArray(JAVA_FLOAT);
            String resolved = js.equals(MemorySegment.NULL) ? null : js.reinterpret(Long.MAX_VALUE).getString(0);
            FREE.invokeExact(pcm);
            if (!js.equals(MemorySegment.NULL)) FREE.invokeExact(js);
            return new Result(new Audio(l, r, 48000), resolved);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    /**
     * Frees the model. Idempotent, and safe while a call is in flight on another thread: it cancels that call, fails
     * queued async calls with a CancellationException, waits for the call to return, then frees.
     */
    @Override public void close() { thread.close(); }
}
