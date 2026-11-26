package org.liberationdata.kiln.whisper;

import org.liberationdata.kiln.Audio;
import org.liberationdata.kiln.HfCache;
import org.liberationdata.kiln.NativeLibrary;
import org.liberationdata.kiln.whisper.Transcript.Segment;
import org.liberationdata.kiln.whisper.Words.RawSegment;
import org.liberationdata.kiln.whisper.Words.Token;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Speech to text with word timings, in-process through whisper.cpp (via libkiln_whisper).
 *
 * <pre>{@code
 * try (var stt = new WhisperModel(WhisperModel.Weights.LARGE_V3_TURBO)) {
 *     Transcript t = stt.transcribe(Audio.readWav(new File("talk.wav")));
 *     for (var w : t.words()) System.out.printf("%.2f %.2f %s%n", w.start(), w.end(), w.text());
 * }
 * }</pre>
 *
 * Audio at any sample rate and channel count goes in; it is mixed to mono and resampled to whisper's 16 kHz.
 * Word times come from DTW alignment on the model's alignment heads when the model's {@link Alignment} is known
 * (every {@link Weights} entry has one), otherwise from whisper's timestamp tokens, which are rougher.
 * Weights stay loaded until {@link #close()}. Not thread-safe: use one instance per thread.
 */
public final class WhisperModel implements AutoCloseable {
    /** whisper's input rate. */
    public static final int SAMPLE_RATE = 16000;

    private static final NativeLibrary LIB = NativeLibrary.load("kiln_whisper");
    private static final MethodHandle LAST_ERROR = LIB.function("kiln_whisper_last_error", FunctionDescriptor.of(ADDRESS));
    private static final MethodHandle CREATE = LIB.function("kiln_whisper_create",
            FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT));
    private static final MethodHandle IS_MULTILINGUAL = LIB.function("kiln_whisper_is_multilingual", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    private static final MethodHandle MODEL_TYPE = LIB.function("kiln_whisper_model_type", FunctionDescriptor.of(ADDRESS, ADDRESS));
    private static final MethodHandle PARAMS_LAYOUT = LIB.function("kiln_whisper_params_layout",
            FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT));
    private static final MethodHandle TRANSCRIBE = LIB.function("kiln_whisper_transcribe", FunctionDescriptor.of(JAVA_INT,
            ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle N_VAD_SEGMENTS = LIB.function("kiln_whisper_n_vad_segments", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    private static final MethodHandle VAD_SEGMENT_T0 = LIB.function("kiln_whisper_vad_segment_t0", FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_INT));
    private static final MethodHandle VAD_SEGMENT_T1 = LIB.function("kiln_whisper_vad_segment_t1", FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_INT));
    private static final MethodHandle CANCEL = LIB.function("kiln_whisper_cancel", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle LANGUAGE = LIB.function("kiln_whisper_language", FunctionDescriptor.of(ADDRESS, ADDRESS));
    private static final MethodHandle N_SEGMENTS = LIB.function("kiln_whisper_n_segments", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    private static final MethodHandle SEGMENT_T0 = LIB.function("kiln_whisper_segment_t0", FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_INT));
    private static final MethodHandle SEGMENT_T1 = LIB.function("kiln_whisper_segment_t1", FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_INT));
    private static final MethodHandle SEGMENT_TEXT = LIB.function("kiln_whisper_segment_text", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT));
    private static final MethodHandle N_TOKENS = LIB.function("kiln_whisper_n_tokens", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
    private static final MethodHandle TOKEN = LIB.function("kiln_whisper_token", FunctionDescriptor.of(ADDRESS,
            ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle DESTROY = LIB.function("kiln_whisper_destroy", FunctionDescriptor.ofVoid(ADDRESS));

    private static final FunctionDescriptor PROGRESS_CB = FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS);
    private static final FunctionDescriptor SEGMENT_CB = FunctionDescriptor.ofVoid(JAVA_INT, JAVA_INT, ADDRESS);
    private static final MethodHandle ON_PROGRESS, ON_SEGMENT;
    static {
        try {
            var lookup = MethodHandles.lookup();
            ON_PROGRESS = lookup.findVirtual(WhisperModel.class, "onProgress", MethodType.methodType(void.class, int.class, MemorySegment.class));
            ON_SEGMENT = lookup.findVirtual(WhisperModel.class, "onSegment", MethodType.methodType(void.class, int.class, int.class, MemorySegment.class));
        } catch (ReflectiveOperationException e) { throw new ExceptionInInitializerError(e); }
    }

    /**
     * The DTW alignment heads for a whisper model size: the attention heads whose weights line up text with audio.
     * They give word times within a frame or two (20 ms each). {@code NONE} times words from whisper's timestamp
     * tokens instead. The values are whisper.cpp's {@code whisper_alignment_heads_preset}.
     */
    public enum Alignment {
        NONE(0), TINY_EN(3), TINY(4), BASE_EN(5), BASE(6), SMALL_EN(7), SMALL(8), MEDIUM_EN(9), MEDIUM(10),
        LARGE_V1(11), LARGE_V2(12), LARGE_V3(13), LARGE_V3_TURBO(14);

        final int preset;

        Alignment(int preset) { this.preset = preset; }
    }

    /**
     * Whisper weights on Hugging Face that Kiln knows by name, from whisper.cpp's own repository. Fetch one with the
     * {@code hf download} command that {@link #path()} prints when it's missing.
     */
    public enum Weights {
        /** large-v3-turbo, 5-bit (574 MB): 99 languages, close to large-v3's accuracy at a fraction of the cost. MIT. */
        LARGE_V3_TURBO("ggml-large-v3-turbo-q5_0.bin", Alignment.LARGE_V3_TURBO),
        /** large-v3-turbo, 8-bit (874 MB). MIT. */
        LARGE_V3_TURBO_Q8_0("ggml-large-v3-turbo-q8_0.bin", Alignment.LARGE_V3_TURBO),
        /** base.en (148 MB): English only, small and fast; good for tests and short commands. MIT. */
        BASE_EN("ggml-base.en.bin", Alignment.BASE_EN);

        public static final String REPO = "ggerganov/whisper.cpp";

        public final String repo = REPO, file;
        public final Alignment alignment;

        Weights(String file, Alignment alignment) { this.file = file; this.alignment = alignment; }

        public String path() { return HfCache.require(repo, file).toString(); }
        public boolean downloaded() { return HfCache.find(repo, file).isPresent(); }

        /** By name, ignoring case and '-', '_', '.': "large-v3-turbo", "base.en", "large-v3-turbo-q8_0". */
        public static Weights named(String name) {
            return valueOf(name.toUpperCase().replace('-', '_').replace('.', '_'));
        }
    }

    /**
     * Silero VAD weights for whisper.cpp, from {@code ggml-org/whisper-vad} on Hugging Face (under 1 MB each, MIT).
     * Fetch one with the {@code hf download} command that {@link #path()} prints when it's missing.
     */
    public enum VadWeights {
        /** Silero VAD v6.2.0: the version whisper.cpp's README uses. */
        SILERO_V6_2_0("ggml-silero-v6.2.0.bin"),
        /** Silero VAD v5.1.2. */
        SILERO_V5_1_2("ggml-silero-v5.1.2.bin");

        public static final String REPO = "ggml-org/whisper-vad";

        public final String repo = REPO, file;

        VadWeights(String file) { this.file = file; }

        public String path() { return HfCache.require(repo, file).toString(); }
        public boolean downloaded() { return HfCache.find(repo, file).isPresent(); }

        /** By name, ignoring case and '-', '_', '.': "silero-v6.2.0", "silero_v5_1_2". */
        public static VadWeights named(String name) {
            return valueOf(name.toUpperCase().replace('-', '_').replace('.', '_'));
        }
    }

    /**
     * Voice activity detection: Silero VAD finds the speech first, and whisper transcribes only that (the segments
     * joined, with 0.1 s of silence between), so silence and music without a voice never reach the decoder. Times in
     * the {@link Transcript} are on the original audio. The fields are whisper.cpp's {@code whisper_vad_params}, with
     * its defaults in {@link #of(String)}: {@code threshold} is the speech probability above which a 32 ms frame is
     * speech; speech shorter than {@code minSpeechMs} is dropped; silence of {@code minSilenceMs} ends a segment; a
     * segment longer than {@code maxSpeechSeconds} is split; each is padded by {@code speechPadMs} either side and
     * carries {@code overlapSeconds} of the audio after it. VAD runs on the CPU, and whisper.cpp keeps the first VAD
     * model it loads for the model's life.
     */
    public record Vad(String modelPath, float threshold, int minSpeechMs, int minSilenceMs, float maxSpeechSeconds,
                      int speechPadMs, float overlapSeconds) {
        public Vad {
            if (modelPath == null || modelPath.isEmpty()) throw new IllegalArgumentException("a VAD model path is required");
        }
        /** A Silero VAD ggml model file, with whisper.cpp's defaults. */
        public static Vad of(String modelPath) { return new Vad(modelPath, 0.5f, 250, 100, Float.MAX_VALUE, 30, 0.1f); }
        public static Vad of(VadWeights w) { return of(w.path()); }
        public Vad withThreshold(float t) { return new Vad(modelPath, t, minSpeechMs, minSilenceMs, maxSpeechSeconds, speechPadMs, overlapSeconds); }
        public Vad withMinSpeechMs(int ms) { return new Vad(modelPath, threshold, ms, minSilenceMs, maxSpeechSeconds, speechPadMs, overlapSeconds); }
        public Vad withMinSilenceMs(int ms) { return new Vad(modelPath, threshold, minSpeechMs, ms, maxSpeechSeconds, speechPadMs, overlapSeconds); }
        public Vad withMaxSpeechSeconds(float s) { return new Vad(modelPath, threshold, minSpeechMs, minSilenceMs, s, speechPadMs, overlapSeconds); }
        public Vad withSpeechPadMs(int ms) { return new Vad(modelPath, threshold, minSpeechMs, minSilenceMs, maxSpeechSeconds, ms, overlapSeconds); }
        public Vad withOverlapSeconds(float s) { return new Vad(modelPath, threshold, minSpeechMs, minSilenceMs, maxSpeechSeconds, speechPadMs, s); }
    }

    /**
     * How to transcribe. {@code language} null (or "auto") detects it; otherwise a code such as "en" or "de".
     * {@code translate} turns the speech into English text. {@code initialPrompt} (null for none) conditions the first
     * window: names, spellings, style. {@code threads} &lt;= 0 is whisper.cpp's default. {@code beamSize} &lt;= 1 is
     * greedy decoding (fastest); 5 is what whisper's CLIs use. {@code temperature} &lt; 0 is the default (0).
     *
     * <p>Whisper can hallucinate in long silence, or under music with no voice: typically it repeats the last line it
     * heard until the audio ends. The defaults are whisper.cpp's, and these options counter it:
     * <ul>
     *   <li>{@code vad} (null, the default, for none): transcribe only the speech that voice activity detection finds.
     *   The best fix where there is silence or music, and faster, since less audio is decoded.</li>
     *   <li>{@code conditionOnPreviousText} (default true): prompt each 30 s window with the text before it. False
     *   stops a line carrying on into the next window, at some cost to punctuation and casing across windows.</li>
     *   <li>Fallback: a window whose text is repetitive (entropy of its last 32 tokens below {@code entropyThreshold},
     *   default 2.4; ln 32 is about 3.47) or unlikely (average log-probability below {@code logProbThreshold}, default
     *   -1) is decoded again, {@code temperatureIncrement} (default 0.2; 0 never retries) hotter each time, up to 1.
     *   A window more likely than {@code noSpeechThreshold} (default 0.6) to be silence that also fails the
     *   log-probability check is dropped. A higher entropy threshold catches more repeats, and also more of a song's
     *   real refrains.</li>
     *   <li>{@code suppressNonSpeech} (default false): never output non-speech tokens (♪, brackets, quotes, ...). It
     *   shifts the decoding more than that suggests: on a sung vocal it stopped the repeats, but most commas and full
     *   stops went too, and "ten" became "10".</li>
     * </ul>
     * Each window of a transcription can draw on the text before it, but one transcription never draws on another.
     */
    public record Options(String language, boolean translate, String initialPrompt, int threads, int beamSize, float temperature,
                          boolean conditionOnPreviousText, boolean suppressNonSpeech, float temperatureIncrement, float entropyThreshold,
                          float logProbThreshold, float noSpeechThreshold, Vad vad) {
        public static final Options DEFAULT = new Options(null, false, null, 0, 5, -1);

        /** The first six options, with the defaults for the rest. */
        public Options(String language, boolean translate, String initialPrompt, int threads, int beamSize, float temperature) {
            this(language, translate, initialPrompt, threads, beamSize, temperature, true, false, 0.2f, 2.4f, -1f, 0.6f, null);
        }

        public Options withLanguage(String l) { return new Options(l, translate, initialPrompt, threads, beamSize, temperature, conditionOnPreviousText, suppressNonSpeech, temperatureIncrement, entropyThreshold, logProbThreshold, noSpeechThreshold, vad); }
        public Options withTranslate(boolean t) { return new Options(language, t, initialPrompt, threads, beamSize, temperature, conditionOnPreviousText, suppressNonSpeech, temperatureIncrement, entropyThreshold, logProbThreshold, noSpeechThreshold, vad); }
        public Options withInitialPrompt(String p) { return new Options(language, translate, p, threads, beamSize, temperature, conditionOnPreviousText, suppressNonSpeech, temperatureIncrement, entropyThreshold, logProbThreshold, noSpeechThreshold, vad); }
        public Options withThreads(int n) { return new Options(language, translate, initialPrompt, n, beamSize, temperature, conditionOnPreviousText, suppressNonSpeech, temperatureIncrement, entropyThreshold, logProbThreshold, noSpeechThreshold, vad); }
        public Options withBeamSize(int n) { return new Options(language, translate, initialPrompt, threads, n, temperature, conditionOnPreviousText, suppressNonSpeech, temperatureIncrement, entropyThreshold, logProbThreshold, noSpeechThreshold, vad); }
        public Options withTemperature(float t) { return new Options(language, translate, initialPrompt, threads, beamSize, t, conditionOnPreviousText, suppressNonSpeech, temperatureIncrement, entropyThreshold, logProbThreshold, noSpeechThreshold, vad); }
        public Options withConditionOnPreviousText(boolean b) { return new Options(language, translate, initialPrompt, threads, beamSize, temperature, b, suppressNonSpeech, temperatureIncrement, entropyThreshold, logProbThreshold, noSpeechThreshold, vad); }
        public Options withSuppressNonSpeech(boolean b) { return new Options(language, translate, initialPrompt, threads, beamSize, temperature, conditionOnPreviousText, b, temperatureIncrement, entropyThreshold, logProbThreshold, noSpeechThreshold, vad); }
        public Options withTemperatureIncrement(float t) { return new Options(language, translate, initialPrompt, threads, beamSize, temperature, conditionOnPreviousText, suppressNonSpeech, t, entropyThreshold, logProbThreshold, noSpeechThreshold, vad); }
        public Options withEntropyThreshold(float t) { return new Options(language, translate, initialPrompt, threads, beamSize, temperature, conditionOnPreviousText, suppressNonSpeech, temperatureIncrement, t, logProbThreshold, noSpeechThreshold, vad); }
        public Options withLogProbThreshold(float t) { return new Options(language, translate, initialPrompt, threads, beamSize, temperature, conditionOnPreviousText, suppressNonSpeech, temperatureIncrement, entropyThreshold, t, noSpeechThreshold, vad); }
        public Options withNoSpeechThreshold(float t) { return new Options(language, translate, initialPrompt, threads, beamSize, temperature, conditionOnPreviousText, suppressNonSpeech, temperatureIncrement, entropyThreshold, logProbThreshold, t, vad); }
        public Options withVad(Vad v) { return new Options(language, translate, initialPrompt, threads, beamSize, temperature, conditionOnPreviousText, suppressNonSpeech, temperatureIncrement, entropyThreshold, logProbThreshold, noSpeechThreshold, v); }
        public Options withVad(VadWeights w) { return withVad(Vad.of(w)); }
    }

    /** The native kiln_whisper_params: its size, then each field's offset. */
    static long[] nativeParamsLayout() {
        try (Arena a = Arena.ofConfined()) {
            int max = 64;
            MemorySegment size = a.allocate(JAVA_LONG), offsets = a.allocate(JAVA_LONG, max);
            int n = (int) PARAMS_LAYOUT.invokeExact(size, offsets, max);
            long[] out = new long[n + 1];
            out[0] = size.get(JAVA_LONG, 0);
            for (int i = 0; i < n; i++) out[i + 1] = offsets.getAtIndex(JAVA_LONG, i);
            return out;
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    /**
     * Hears about a transcription as it runs, on the calling thread. A segment arrives once whisper has finished it;
     * the last word of a streamed segment may end slightly later in the returned {@link Transcript}, which times each
     * word against the one after it.
     */
    public interface Listener {
        default void progress(int percent) {}
        default void segment(Segment segment) {}
    }

    private final MemorySegment handle;
    private final boolean multilingual;
    private final String type;
    private Listener listener;
    private float[] pcm;
    private Vad vad;
    private Words.VadTimeline timeline;
    private Throwable callbackError;

    /**
     * @param alignment the model's DTW alignment heads, or {@link Alignment#NONE}
     * @param gpu false runs on the CPU (on a machine with no GPU, also set {@code -Dkiln.allowCpu=true}: Kiln refuses
     *            to load its GPU build there otherwise)
     */
    public WhisperModel(String modelPath, Alignment alignment, boolean gpu) {
        try (Arena a = Arena.ofConfined()) {
            handle = (MemorySegment) CREATE.invokeExact(a.allocateFrom(modelPath), gpu ? 1 : 0, 0, 1, alignment.preset);
            if (handle.equals(MemorySegment.NULL)) throw new IllegalStateException("kiln_whisper_create: " + error());
            multilingual = (int) IS_MULTILINGUAL.invokeExact(handle) != 0;
            type = NativeLibrary.string((MemorySegment) MODEL_TYPE.invokeExact(handle));
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    public WhisperModel(String modelPath, Alignment alignment) { this(modelPath, alignment, true); }

    /** A model file without DTW: word times come from whisper's timestamp tokens. */
    public WhisperModel(String modelPath) { this(modelPath, Alignment.NONE, true); }

    public WhisperModel(Weights w) { this(w.path(), w.alignment, true); }

    /** False for an English-only model (".en"), which always transcribes English. */
    public boolean multilingual() { return multilingual; }

    /** "tiny", "base", "small", "medium" or "large". */
    public String type() { return type; }

    /** Stops the transcription in progress (from any thread); it then throws {@link java.util.concurrent.CancellationException}. */
    public void cancel() {
        try { CANCEL.invokeExact(handle); } catch (Throwable t) { throw new RuntimeException(t); }
    }

    public Transcript transcribe(Audio audio) { return transcribe(audio, Options.DEFAULT, null); }

    public Transcript transcribe(Audio audio, Options options) { return transcribe(audio, options, null); }

    /** Transcribes audio at any rate: it is mixed to mono and resampled to 16 kHz first. */
    public Transcript transcribe(Audio audio, Options options, Listener listener) {
        return transcribe(Audio.resample(audio.mono(), audio.sampleRate(), SAMPLE_RATE), options, listener);
    }

    /**
     * Transcribes 16 kHz mono float PCM in [-1, 1]. listener may be null. Audio shorter than a second is padded with
     * silence to one: whisper.cpp skips input under 100 ms without clearing its spectrogram, so it would otherwise
     * transcribe the previous call's audio again.
     *
     * @throws IllegalArgumentException for a language whisper doesn't know
     */
    public Transcript transcribe(float[] pcm16k, Options o, Listener listener) {
        if (pcm16k.length < SAMPLE_RATE) pcm16k = Arrays.copyOf(pcm16k, SAMPLE_RATE);
        this.listener = listener;
        this.pcm = pcm16k;
        this.callbackError = null;
        this.vad = o.vad();
        this.timeline = null;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment samples = a.allocateFrom(JAVA_FLOAT, pcm16k);
            MemorySegment progress = MemorySegment.NULL, segment = MemorySegment.NULL;
            if (listener != null) {
                progress = NativeLibrary.linker().upcallStub(ON_PROGRESS.bindTo(this), PROGRESS_CB, a);
                segment = NativeLibrary.linker().upcallStub(ON_SEGMENT.bindTo(this), SEGMENT_CB, a);
            }
            int rc = (int) TRANSCRIBE.invokeExact(handle, samples, pcm16k.length, WhisperParams.of(a, o), progress, segment, MemorySegment.NULL);
            if (callbackError != null) throw new RuntimeException("listener failed", callbackError);
            if (rc == -100) throw new java.util.concurrent.CancellationException("transcription cancelled");
            if (rc == -101) throw new IllegalArgumentException(error());
            if (rc != 0) throw new IllegalStateException("transcription failed (rc=" + rc + "): " + error());
            List<RawSegment> raw = read(a, 0, (int) N_SEGMENTS.invokeExact(handle));
            List<Segment> segments = Words.fitToWords(Words.snapToVoice(Words.assemble(raw), pcm16k, SAMPLE_RATE));
            StringBuilder text = new StringBuilder();
            for (RawSegment s : raw) text.append(s.text());
            String language = vad != null && (int) N_VAD_SEGMENTS.invokeExact(handle) == 0
                    ? WhisperParams.languageWithoutSpeech(o.language(), multilingual)
                    : NativeLibrary.string((MemorySegment) LANGUAGE.invokeExact(handle));
            return new Transcript(text.toString().strip(), language, segments);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { this.listener = null; this.pcm = null; this.vad = null; this.timeline = null; }
    }

    /** Where VAD put this transcription's speech (known once whisper starts decoding); NONE without VAD. */
    private Words.VadTimeline timeline() throws Throwable {
        if (timeline == null) {
            int n = vad == null ? 0 : (int) N_VAD_SEGMENTS.invokeExact(handle);
            long[] t0 = new long[n], t1 = new long[n];
            for (int i = 0; i < n; i++) {
                t0[i] = (long) VAD_SEGMENT_T0.invokeExact(handle, i);
                t1[i] = (long) VAD_SEGMENT_T1.invokeExact(handle, i);
            }
            timeline = Words.VadTimeline.of(t0, t1, pcm.length, vad == null ? 0 : vad.overlapSeconds());
        }
        return timeline;
    }

    private List<RawSegment> read(Arena a, int first, int n) throws Throwable {
        MemorySegment special = a.allocate(JAVA_INT), p = a.allocate(JAVA_FLOAT),
                t0 = a.allocate(JAVA_LONG), t1 = a.allocate(JAVA_LONG), dtw = a.allocate(JAVA_LONG);
        Words.VadTimeline vt = n > 0 ? timeline() : Words.VadTimeline.NONE;
        List<RawSegment> out = new ArrayList<>(n);
        for (int s = first; s < first + n; s++) {
            int nt = (int) N_TOKENS.invokeExact(handle, s);
            List<Token> tokens = new ArrayList<>(nt);
            for (int t = 0; t < nt; t++) {
                MemorySegment text = (MemorySegment) TOKEN.invokeExact(handle, s, t, special, p, t0, t1, dtw);
                tokens.add(new Token(bytes(text), special.get(JAVA_INT, 0) != 0, p.get(JAVA_FLOAT, 0),
                        t0.get(JAVA_LONG, 0), t1.get(JAVA_LONG, 0), vt.map(dtw.get(JAVA_LONG, 0))));
            }
            out.add(new RawSegment((long) SEGMENT_T0.invokeExact(handle, s), (long) SEGMENT_T1.invokeExact(handle, s),
                    NativeLibrary.string((MemorySegment) SEGMENT_TEXT.invokeExact(handle, s)), tokens));
        }
        return out;
    }

    /** A C string's bytes, without decoding them (a token can end part-way through a UTF-8 character). */
    private static byte[] bytes(MemorySegment s) {
        if (s.equals(MemorySegment.NULL)) return new byte[0];
        MemorySegment r = s.reinterpret(Long.MAX_VALUE);
        long n = 0;
        while (r.get(JAVA_BYTE, n) != 0) n++;
        return r.asSlice(0, n).toArray(JAVA_BYTE);
    }

    private void onProgress(int percent, MemorySegment user) {
        if (callbackError != null) return;
        try { listener.progress(percent); } catch (Throwable t) { callbackError = t; cancel(); }
    }

    private void onSegment(int first, int n, MemorySegment user) {
        if (callbackError != null) return;
        try (Arena a = Arena.ofConfined()) {
            for (Segment s : Words.fitToWords(Words.snapToVoice(Words.assemble(read(a, first, n)), pcm, SAMPLE_RATE))) listener.segment(s);
        } catch (Throwable t) { callbackError = t; cancel(); }
    }

    private static String error() {
        try { return NativeLibrary.string((MemorySegment) LAST_ERROR.invokeExact()); }
        catch (Throwable t) { return t.toString(); }
    }

    @Override public void close() {
        try { DESTROY.invokeExact(handle); } catch (Throwable t) { throw new RuntimeException(t); }
    }
}
