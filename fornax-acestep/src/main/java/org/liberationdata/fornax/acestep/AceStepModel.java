package org.liberationdata.fornax.acestep;

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

    private static final ModelReferences.Engine SET = new ModelReferences.Engine("AceStepModel", "ACE-Step song models",
            Set.of(ModelSet.Kind.SONG), Set.of(), "an ACE-Step set");
    private static final ModelReferences.Engine FILE = new ModelReferences.Engine("AceStepModel",
            "one file per path here (pass a set's name alone for the whole set)", Set.of(), Set.of("gguf"), "a GGUF");

    /**
     * The catalogue set with this name, kept loaded, downloaded on first use: {@code "ace-step-turbo"} (the DiT alone,
     * with caption and lyrics), {@code "ace-step"} (turbo with the 4B lyric LM), {@code "ace-step-xl-turbo"} or
     * {@code "ace-step-xl"}.
     */
    public AceStepModel(String model) { this(model, true); }

    /** {@link #AceStepModel(String)}; keepLoaded false frees each stage's weights between generations. */
    public AceStepModel(String model, boolean keepLoaded) { this(ModelReferences.set(model, SET, "acestep.cpp"), keepLoaded); }

    private AceStepModel(Map<String, Path> files, boolean keepLoaded) {
        this(files.containsKey("lm") ? files.get("lm").toString() : null, files.get("text_encoder").toString(),
                files.get("dit").toString(), files.get("vae").toString(), keepLoaded);
    }

    /**
     * Each path may be a model reference instead ({@code Serveurperso/ACE-Step-1.5-GGUF:vae-BF16.gguf}), downloaded on
     * first use. lmPath may be null to skip the 5 Hz LM (DiT-only: caption + lyrics drive generation directly).
     */
    public AceStepModel(String lmPath, String textEncPath, String ditPath, String vaePath, boolean keepLoaded) {
        if (lmPath != null) ModelReferences.check(lmPath, FILE);   // every argument refused before any download
        for (String p : new String[]{textEncPath, ditPath, vaePath}) ModelReferences.check(Objects.requireNonNull(p, "path"), FILE);
        lmPath = lmPath == null ? null : ModelReferences.path(lmPath, FILE);
        textEncPath = ModelReferences.path(textEncPath, FILE);
        ditPath = ModelReferences.path(ditPath, FILE);
        vaePath = ModelReferences.path(vaePath, FILE);
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
