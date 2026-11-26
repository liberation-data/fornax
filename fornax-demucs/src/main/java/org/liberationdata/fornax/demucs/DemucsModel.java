package org.liberationdata.fornax.demucs;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.Call;
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.ModelThread;
import org.liberationdata.fornax.NativeLibrary;
import org.liberationdata.fornax.hub.ModelReferences;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Splits music into stems (vocals, drums, bass, other, ...) in-process with Meta's HTDemucs, through stems.cpp (via
 * libfornax_demucs).
 *
 * <pre>{@code
 * try (var demucs = new DemucsModel(DemucsModel.Weights.HTDEMUCS)) {
 *     Stems s = demucs.separate(Audio.readWav(new File("song.wav")));
 *     s.vocals().writeWav(new File("song-vocals.wav"));
 *     s.backing().writeWav(new File("song-backing.wav"));
 *     Audio drums = s.get("drums");
 * }
 * }</pre>
 *
 * Audio at any sample rate goes in; the model runs at 44.1 kHz stereo, and the stems come back at the input's rate
 * and length. The mix is normalised by its mono mean and deviation first and the stems scaled back, as demucs does,
 * and split into overlapping 7.8 s segments that are cross-faded back together. Weights stay loaded until
 * {@link #close()}.
 *
 * <p>Threads: calls run one at a time, in arrival order, from any thread. {@link #separate} runs on the calling
 * thread, progress callback included; {@link #separateAsync} runs on this model's own platform thread and returns a
 * {@link CompletableFuture}, so a virtual thread awaiting it doesn't pin its carrier. {@link #cancel()} and
 * {@link #close()} are safe from any thread.
 */
public final class DemucsModel implements AutoCloseable {
    private static final NativeLibrary LIB = NativeLibrary.load("fornax_demucs");
    private static final MethodHandle LAST_ERROR = LIB.function("fornax_demucs_last_error", FunctionDescriptor.of(ADDRESS));
    private static final MethodHandle CREATE = LIB.function("fornax_demucs_create",
            FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_INT));
    private static final MethodHandle NAME = LIB.function("fornax_demucs_name", FunctionDescriptor.of(ADDRESS, ADDRESS));
    private static final MethodHandle BACKEND = LIB.function("fornax_demucs_backend", FunctionDescriptor.of(ADDRESS, ADDRESS));
    private static final MethodHandle N_SOURCES = LIB.function("fornax_demucs_n_sources", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    private static final MethodHandle SOURCE = LIB.function("fornax_demucs_source", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT));
    private static final MethodHandle SAMPLE_RATE = LIB.function("fornax_demucs_sample_rate", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    private static final MethodHandle CHANNELS = LIB.function("fornax_demucs_channels", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    private static final MethodHandle SEPARATE = LIB.function("fornax_demucs_separate", FunctionDescriptor.of(JAVA_INT,
            ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_FLOAT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle SET_CANCEL = LIB.function("fornax_demucs_set_cancel", FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT));
    private static final MethodHandle FREE = LIB.function("fornax_demucs_free", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle DESTROY = LIB.function("fornax_demucs_destroy", FunctionDescriptor.ofVoid(ADDRESS));
    private static final int CANCELLED = 3;         // FORNAX_DEMUCS_CANCELLED
    private static final int INVALID_ARGUMENT = 1;  // FORNAX_DEMUCS_INVALID_ARGUMENT

    private static final FunctionDescriptor PROGRESS_CB = FunctionDescriptor.ofVoid(JAVA_INT, JAVA_INT, ADDRESS);
    private static final MethodHandle ON_PROGRESS;
    static {
        try {
            ON_PROGRESS = MethodHandles.lookup().findStatic(DemucsModel.class, "onProgress",
                    MethodType.methodType(void.class, Sink.class, int.class, int.class, MemorySegment.class));
        } catch (ReflectiveOperationException e) { throw new ExceptionInInitializerError(e); }
    }
    private record Sink(Call call, Progress progress) {}

    /**
     * The progress upcall, bound to its call's sink in that call's own stub (not a thread-local, which a call to
     * another DemucsModel from the callback, on the same thread, would replace and clear). Nothing may escape it: an
     * exception thrown into the engine's frames kills the JVM. So a listener's exception is recorded in the call,
     * which cancels the separation, and {@link #separate} rethrows it.
     */
    private static void onProgress(Sink sink, int done, int total, MemorySegment user) {
        if (sink.call().stopRequested()) return;
        try {
            sink.progress().update(done, total);
        } catch (Throwable t) {
            sink.call().fail(t);
        }
    }

    /**
     * Stem separation weights on Hugging Face that Fornax knows by name: stems.cpp's conversions of Meta's demucs
     * checkpoints (MIT, code and weights). Fetch one with the {@code hf download} command that {@link #path()} prints
     * when it's missing.
     */
    public enum Weights {
        /** HTDemucs v4 (42M parameters, 160 MB): drums, bass, other, vocals. demucs' default model. MIT. */
        HTDEMUCS("htdemucs-42M-v1.0-F32.gguf"),
        /** HTDemucs v4 with F16 weights (100 MB). MIT. */
        HTDEMUCS_F16("htdemucs-42M-v1.0-F16.gguf"),
        /** HTDemucs v4, six stems (27M, 104 MB): drums, bass, other, vocals, guitar, piano. MIT. */
        HTDEMUCS_6S("htdemucs_6s-27M-v1.0-F32.gguf"),
        /**
         * HTDemucs v4 fine-tuned (a bag of four 42M models, one per stem, 640 MB): the best of the four, at four times
         * the time. MIT.
         */
        HTDEMUCS_FT("htdemucs_ft-4x42M-v1.0-F32.gguf");

        public static final String REPO = "thepatch/htdemucs-GGUF";

        public final String repo = REPO, file;

        Weights(String file) { this.file = file; }

        public String path() { return HfCache.require(repo, file).toString(); }
        public boolean downloaded() { return HfCache.find(repo, file).isPresent(); }

        /** By name, ignoring case and '-' or '_': "htdemucs", "htdemucs-6s", "htdemucs_ft". */
        public static Weights named(String name) { return valueOf(name.toUpperCase().replace('-', '_')); }
    }

    /**
     * How to separate. {@code shifts} averages that many randomly time-shifted passes (demucs' {@code --shifts}; a
     * little cleaner, at {@code shifts} times the time), 0 for one deterministic pass. {@code overlap} between segments, in [0, 1);
     * negative is the model's own (0.25 for HTDemucs). {@code seed} picks the shifts.
     */
    public record Options(int shifts, float overlap, int seed) {
        public static final Options DEFAULT = new Options(0, -1, 0);

        public Options {
            if (shifts < 0 || shifts > 10) throw new IllegalArgumentException("shifts must be 0 to 10: " + shifts);
            if (!(overlap < 1)) throw new IllegalArgumentException("overlap must be below 1: " + overlap);
        }

        public Options withShifts(int n) { return new Options(n, overlap, seed); }
        public Options withOverlap(float o) { return new Options(shifts, o, seed); }
        public Options withSeed(int s) { return new Options(shifts, overlap, s); }
    }

    /** Hears how far a separation has got: {@code done} of {@code total} segment passes, on the separating thread. */
    @FunctionalInterface
    public interface Progress {
        void update(int done, int total);
    }

    private final MemorySegment handle;
    private final ModelThread thread;
    private final String name, backend;
    private final List<String> sources;
    private final int sampleRate, channels;

    private static final ModelReferences.Engine ENGINE = new ModelReferences.Engine("DemucsModel", "stem-separation models",
            Set.of(), Set.of("gguf"), "a stems.cpp GGUF");

    /**
     * @param modelPath a stems.cpp GGUF, or a model reference to one, downloaded on first use:
     *                  {@code thepatch/htdemucs-GGUF:htdemucs-42M-v1.0-F32.gguf}
     * @param gpu false runs on the CPU (on a machine with no GPU, also set {@code -Dfornax.allowCpu=true}: Fornax refuses
     *            to load its GPU build there otherwise)
     * @param cpuThreads threads for the CPU backend; 0 for one per physical core
     */
    public DemucsModel(String modelPath, boolean gpu, int cpuThreads) {
        modelPath = ModelReferences.locate(modelPath, ENGINE).map(f -> ModelReferences.file(f, ENGINE).toString()).orElse(modelPath);
        try (Arena a = Arena.ofConfined()) {
            MemorySegment device = gpu ? MemorySegment.NULL : a.allocateFrom("cpu");
            handle = (MemorySegment) CREATE.invokeExact(a.allocateFrom(modelPath), device, cpuThreads);
            if (handle.equals(MemorySegment.NULL)) throw new IllegalStateException("fornax_demucs_create: " + error());
            name = NativeLibrary.string((MemorySegment) NAME.invokeExact(handle));
            backend = NativeLibrary.string((MemorySegment) BACKEND.invokeExact(handle));
            int n = (int) N_SOURCES.invokeExact(handle);
            List<String> s = new ArrayList<>(n);
            for (int i = 0; i < n; i++) s.add(NativeLibrary.string((MemorySegment) SOURCE.invokeExact(handle, i)));
            sources = List.copyOf(s);
            sampleRate = (int) SAMPLE_RATE.invokeExact(handle);
            channels = (int) CHANNELS.invokeExact(handle);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        MemorySegment h = handle;
        thread = new ModelThread(name, () -> destroy(h), () -> setCancel(h, 1), () -> setCancel(h, 0));
    }

    /** A stems.cpp GGUF or a reference to one, on the GPU. */
    public DemucsModel(String modelPath) { this(modelPath, true, 0); }

    public DemucsModel(Weights w) { this(w.path(), true, 0); }

    /** The checkpoint: "htdemucs", "htdemucs_6s" or "htdemucs_ft". */
    public String name() { return name; }

    /** The ggml backend it runs on, e.g. "Vulkan0", or "CPU". */
    public String backend() { return backend; }

    /** The stems it makes, in order: "drums", "bass", "other", "vocals" (then "guitar", "piano" for htdemucs_6s). */
    public List<String> sources() { return sources; }

    /** The rate the model runs at (44100); input at another rate is resampled to it and the stems back. */
    public int sampleRate() { return sampleRate; }

    public Stems separate(Audio audio) { return separate(audio, Options.DEFAULT, null); }

    public Stems separate(Audio audio, Options options) { return separate(audio, options, null); }

    /**
     * Separates on the calling thread (pinning a virtual thread's carrier meanwhile: see {@link #separateAsync}).
     * progress may be null.
     *
     * @throws CancellationException if {@link #cancel()} stopped it
     */
    public Stems separate(Audio audio, Options options, Progress progress) {
        return thread.run(c -> separateOn(c, audio, options, progress));
    }

    /** {@link #separate}, on this model's own platform thread. Cancelling the future stops the separation. */
    public CompletableFuture<Stems> separateAsync(Audio audio) { return separateAsync(audio, Options.DEFAULT, null); }

    /** {@link #separate}, on this model's own platform thread. Cancelling the future stops the separation. */
    public CompletableFuture<Stems> separateAsync(Audio audio, Options options, Progress progress) {
        return thread.submit(c -> separateOn(c, audio, options, progress));
    }

    /**
     * Stops the separation in progress, from any thread: stems.cpp checks before each segment (about 7.8 s of
     * audio). That call throws a {@link CancellationException}. A call that hasn't started isn't affected: cancel its
     * future instead.
     */
    public void cancel() { thread.cancel(); }

    private Stems separateOn(Call call, Audio audio, Options o, Progress progress) {
        int n = audio.left().length;
        if (n == 0) throw new IllegalArgumentException("no audio to separate");
        try (Arena a = Arena.ofConfined()) {
            MemorySegment pcm = a.allocate(JAVA_FLOAT, 2L * n);
            MemorySegment.copy(audio.left(), 0, pcm, JAVA_FLOAT, 0, n);
            MemorySegment.copy(audio.right(), 0, pcm, JAVA_FLOAT, (long) n * Float.BYTES, n);
            MemorySegment cb = MemorySegment.NULL;
            if (progress != null)
                cb = NativeLibrary.linker().upcallStub(MethodHandles.insertArguments(ON_PROGRESS, 0, new Sink(call, progress)), PROGRESS_CB, a);
            MemorySegment out = a.allocate(ADDRESS);
            int rc = (int) SEPARATE.invokeExact(handle, pcm, (long) n, 2, audio.sampleRate(), o.shifts(), o.overlap(),
                    o.seed(), cb, MemorySegment.NULL, out);
            // A callback that fails at the last segment comes after the engine's last cancel check, so the engine
            // returns its stems anyway: free them before rethrowing.
            if (rc == 0 && call.failure() != null) FREE.invokeExact(out.get(ADDRESS, 0));
            call.rethrow();
            if (rc == CANCELLED) throw new CancellationException("separation cancelled");
            if (rc == INVALID_ARGUMENT) throw new IllegalArgumentException(error());
            if (rc != 0) throw new IllegalStateException("separation failed (rc=" + rc + "): " + error());
            MemorySegment p = out.get(ADDRESS, 0).reinterpret((long) sources.size() * channels * n * Float.BYTES);
            List<Audio> stems = new ArrayList<>(sources.size());
            try {
                for (int s = 0; s < sources.size(); s++) {
                    long base = (long) s * channels * n;
                    float[] l = p.asSlice(base * Float.BYTES, (long) n * Float.BYTES).toArray(JAVA_FLOAT);
                    float[] r = channels > 1 ? p.asSlice((base + n) * Float.BYTES, (long) n * Float.BYTES).toArray(JAVA_FLOAT) : l.clone();
                    stems.add(new Audio(l, r, audio.sampleRate()));
                }
            } finally {
                FREE.invokeExact(p);
            }
            return Stems.of(sources, stems);
        } catch (RuntimeException | Error e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    private static void destroy(MemorySegment h) {
        try { DESTROY.invokeExact(h); } catch (Throwable t) { throw new RuntimeException(t); }
    }

    private static void setCancel(MemorySegment h, int on) {
        try { SET_CANCEL.invokeExact(h, on); } catch (Throwable t) { throw new RuntimeException(t); }
    }

    private static String error() {
        try { return NativeLibrary.string((MemorySegment) LAST_ERROR.invokeExact()); }
        catch (Throwable t) { return t.toString(); }
    }

    /**
     * Frees the model. Idempotent, and safe while a call is in flight on another thread: it cancels that call, fails
     * queued async calls with a CancellationException, waits for the call to return, then frees.
     */
    @Override public void close() { thread.close(); }
}
