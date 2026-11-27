package org.liberationdata.fornax.yue2;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.Call;
import org.liberationdata.fornax.ModelThread;
import org.liberationdata.fornax.NativeLibrary;
import org.liberationdata.fornax.hub.ModelReferences;
import org.liberationdata.fornax.hub.ModelSet;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

import static java.lang.foreign.ValueLayout.*;

/**
 * YuE2 song generation in-process via yue2.cpp (through libfornax_yue2).
 * Request is yue2.cpp's JSON (style, lyrics, duration, cot, seed, lm_seed, steps, ...); output is 48 kHz stereo.
 * YuE2 weights are CC BY-NC 4.0 (no commercial use). Calls run one at a time, from any thread; see
 * {@link #generateAsync} for virtual threads.
 */
public final class Yue2Model implements AutoCloseable {
    private static final NativeLibrary LIB = NativeLibrary.load("fornax_yue2");

    private static MethodHandle fn(String n, FunctionDescriptor fd) { return LIB.function(n, fd); }

    private static final MethodHandle LAST_ERROR = fn("fornax_yue2_last_error", FunctionDescriptor.of(ADDRESS));
    private static final MethodHandle CREATE = fn("fornax_yue2_create", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT));
    private static final MethodHandle GENERATE = fn("fornax_yue2_generate", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle FREE = fn("fornax_yue2_free", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle DESTROY = fn("fornax_yue2_destroy", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle SET_CANCEL = fn("fornax_yue2_set_cancel", FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT));
    private static final int CANCELLED = 3;  // FORNAX_YUE2_CANCELLED

    /** abcScore is the ABC-notation score YuE2 writes before rendering audio, when present. */
    public record Result(Audio audio, String resolvedRequestJson, String abcScore) {}

    private static String str(MemorySegment s) { return s.equals(MemorySegment.NULL) ? null : s.reinterpret(Long.MAX_VALUE).getString(0); }

    private static String err() {
        try { return str((MemorySegment) LAST_ERROR.invokeExact()); } catch (Throwable t) { return t.toString(); }
    }

    private final MemorySegment handle;
    private final ModelThread thread;

    private static final ModelReferences.Engine SET = new ModelReferences.Engine("Yue2Model", "YuE2 song models",
            Set.of(ModelSet.Kind.SONG), Set.of(), "a YuE2 set");
    private static final ModelReferences.Engine FILE = new ModelReferences.Engine("Yue2Model",
            "one file per path here (pass a set's name alone for the whole set)", Set.of(), Set.of("gguf"), "a GGUF");

    /** The catalogue set with this name ({@code "yue2"}), kept loaded with the engine's own sequence limit, downloaded on first use. */
    public Yue2Model(String model) { this(model, true, 0); }

    /** {@link #Yue2Model(String)} with the loading options of {@link #Yue2Model(String, String, boolean, int)}. */
    public Yue2Model(String model, boolean keepLoaded, int maxSeq) { this(ModelReferences.set(model, SET, "yue2.cpp"), keepLoaded, maxSeq); }

    private Yue2Model(Map<String, Path> files, boolean keepLoaded, int maxSeq) {
        this(files.get("model").toString(), files.get("vae").toString(), keepLoaded, maxSeq);
    }

    /**
     * Either path may be a model reference instead ({@code Serveurperso/YuE2-GGUF:YuE2-Vae-F32.gguf}), downloaded on
     * first use.
     *
     * @param keepLoaded keep every stage's weights loaded between generations
     * @param maxSeq the LM's sequence limit, or 0 for the engine's own
     */
    public Yue2Model(String modelPath, String vaePath, boolean keepLoaded, int maxSeq) {
        ModelReferences.check(Objects.requireNonNull(modelPath, "modelPath"), FILE);   // both refused before any download
        ModelReferences.check(Objects.requireNonNull(vaePath, "vaePath"), FILE);
        modelPath = ModelReferences.path(modelPath, FILE);
        vaePath = ModelReferences.path(vaePath, FILE);
        try (Arena a = Arena.ofConfined()) {
            handle = (MemorySegment) CREATE.invokeExact(a.allocateFrom(modelPath), a.allocateFrom(vaePath), keepLoaded ? 1 : 0, maxSeq);
        } catch (Throwable t) { throw new RuntimeException(t); }
        if (handle.equals(MemorySegment.NULL)) throw new IllegalStateException("fornax_yue2_create: " + err());
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
            if (rc != 0) throw new IllegalStateException("fornax_yue2_generate rc=" + rc + ": " + err());
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
