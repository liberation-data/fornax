package org.liberationdata.kiln.tts;

import org.liberationdata.kiln.Audio;
import org.liberationdata.kiln.HfCache;
import org.liberationdata.kiln.NativeLibrary;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * A text-to-speech model running in-process through llama.cpp's audio generation (via libkiln_tts). One engine runs
 * every TTS model llama.cpp supports: a backbone GGUF plus its mmproj GGUF (the audio decoder and the model's
 * predefined voices). See {@link Weights} for the ones Kiln knows by name.
 * Weights stay loaded until {@link #close()}. Not thread-safe: use one instance per thread.
 */
public final class TtsModel implements AutoCloseable {
    private static final NativeLibrary LIB = NativeLibrary.load("kiln_tts");
    private static final MethodHandle LAST_ERROR = LIB.function("kiln_tts_last_error", FunctionDescriptor.of(ADDRESS));
    private static final MethodHandle CREATE = LIB.function("kiln_tts_create",
            FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT));
    private static final MethodHandle INFO = LIB.function("kiln_tts_info",
            FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle VOICE_NAME = LIB.function("kiln_tts_voice_name",
            FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT));
    private static final MethodHandle SPEAK = LIB.function("kiln_tts_speak", FunctionDescriptor.of(JAVA_INT,
            ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_FLOAT, JAVA_INT, JAVA_FLOAT, JAVA_INT,
            ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle FREE = LIB.function("kiln_tts_free", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle DESTROY = LIB.function("kiln_tts_destroy", FunctionDescriptor.ofVoid(ADDRESS));

    private static final FunctionDescriptor AUDIO_CB = FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS);
    private static final MethodHandle ON_AUDIO;
    static {
        try {
            ON_AUDIO = MethodHandles.lookup().findStatic(TtsModel.class, "onAudio",
                    MethodType.methodType(int.class, MemorySegment.class, long.class, int.class, MemorySegment.class));
        } catch (ReflectiveOperationException e) { throw new ExceptionInInitializerError(e); }
    }
    private record Sink(TtsModel model, Consumer<Audio> consumer) {}

    private static final ThreadLocal<Sink> SINK = new ThreadLocal<>();

    private static int onAudio(MemorySegment pcm, long frames, int channels, MemorySegment user) {
        Sink sink = SINK.get();
        if (sink == null) return 0;
        float[] f = pcm.reinterpret(frames * channels * Float.BYTES).toArray(JAVA_FLOAT);
        sink.consumer().accept(Pcm.toAudio(f, channels, sink.model().sampleRate));
        return sink.model().cancelled ? 1 : 0;
    }

    /**
     * TTS weights on Hugging Face that Kiln knows by name (all run through this one engine). Fetch one with the
     * {@code hf download} command that {@link #model()} prints when it's missing.
     */
    public enum Weights {
        /** Kokoro-82M: 54 voices (American and British English, plus 7 other languages). Apache-2.0. */
        KOKORO("ggml-org/Kokoro-82M-GGUF", "Kokoro-82M-F32.gguf", "mmproj-Kokoro-82M-F16.gguf"),
        /** Zyphra ZONOS3-Mini: 27 voices (American, British and Irish English, plus 9 other languages). Apache-2.0. */
        ZONOS3("ggml-org/ZONOS3-Mini-GGUF", "ZONOS3-Mini-Q8_0.gguf", "mmproj-ZONOS3-Mini-F32.gguf"),
        /** LiquidAI LFM3-Audio-2B: voices "female" (default) and "male". LFM Open License. */
        LFM3_AUDIO("LiquidAI/LFM3-Audio-2B-GGUF", "LFM3-Audio-2B-Q8_0.gguf", "mmproj-LFM3-Audio-2B-F16.gguf"),
        /** tiny-voca (4.5M parameters): voices M2 (default) and F2. Apache-2.0. */
        TINY_VOCA("ggml-org/tiny-voca-GGUF", "tiny-voca-F32.gguf", "mmproj-tiny-voca-F32.gguf"),
        /** Chatterbox Turbo: one voice, "default". MIT. */
        CHATTERBOX_TURBO("ggml-org/chatterbox-turbo-GGUF", "chatterbox-turbo-Q8_0.gguf", "mmproj-chatterbox-turbo-F16.gguf"),
        /** Soprano-1.1-80M: one fixed voice (American English, female). Apache-2.0. */
        SOPRANO("ggml-org/Soprano-1.1-80M-GGUF", "Soprano-1.1-80M-F32.gguf", "mmproj-Soprano-1.1-80M-F16.gguf"),
        /**
         * OpenBMB VoxCPM2 (2B): no predefined voices; a description in parentheses at the start of the text designs
         * one, e.g. "(A calm male narrator)Hello". Apache-2.0.
         */
        VOXCPM2("ggml-org/VoxCPM2-GGUF", "VoxCPM2-Q8_0.gguf", "mmproj-VoxCPM2-BF16.gguf");

        public final String repo, modelFile, mmprojFile;

        Weights(String repo, String modelFile, String mmprojFile) {
            this.repo = repo; this.modelFile = modelFile; this.mmprojFile = mmprojFile;
        }

        public String model() { return HfCache.require(repo, modelFile).toString(); }
        public String mmproj() { return HfCache.require(repo, mmprojFile).toString(); }
        public boolean downloaded() { return HfCache.find(repo, modelFile).isPresent() && HfCache.find(repo, mmprojFile).isPresent(); }

        /** By name, ignoring case and '-' vs '_': "kokoro", "zonos3", "lfm3-audio", "tiny-voca", "chatterbox-turbo", ... */
        public static Weights named(String name) {
            return valueOf(name.toUpperCase().replace('-', '_'));
        }
    }

    /**
     * lang null = the model's default. temperature &lt; 0 = the model's default. seed -1 = random.
     * speed &lt;= 0 = the model's own pace (most models take no other value). maxFrames &lt;= 0 = the model's limit.
     */
    public record Options(String lang, float temperature, int seed, float speed, int maxFrames) {
        public static final Options DEFAULT = new Options(null, -1, -1, 0, 0);
        public Options withSeed(int s) { return new Options(lang, temperature, s, speed, maxFrames); }
        public Options withLang(String l) { return new Options(l, temperature, seed, speed, maxFrames); }
        public Options withTemperature(float t) { return new Options(lang, t, seed, speed, maxFrames); }
        public Options withSpeed(float s) { return new Options(lang, temperature, seed, s, maxFrames); }
        public Options withMaxFrames(int n) { return new Options(lang, temperature, seed, speed, n); }
    }

    private final MemorySegment handle;
    private final int sampleRate, channels;
    private final List<String> voices;
    private volatile boolean cancelled;

    /** @param gpuLayers -1 offloads every layer. @param contextSize 0 uses llama.cpp's default for the model. */
    public TtsModel(String modelPath, String mmprojPath, int gpuLayers, int contextSize) {
        try (Arena a = Arena.ofConfined()) {
            handle = (MemorySegment) CREATE.invokeExact(a.allocateFrom(modelPath), a.allocateFrom(mmprojPath), gpuLayers, contextSize);
            if (handle.equals(MemorySegment.NULL)) throw new IllegalStateException("kiln_tts_create: " + error());
            MemorySegment rate = a.allocate(JAVA_INT), ch = a.allocate(JAVA_INT), nv = a.allocate(JAVA_INT);
            INFO.invokeExact(handle, rate, ch, nv);
            sampleRate = rate.get(JAVA_INT, 0);
            channels = Math.max(1, ch.get(JAVA_INT, 0));
            List<String> v = new ArrayList<>();
            for (int i = 0; i < nv.get(JAVA_INT, 0); i++) v.add(NativeLibrary.string((MemorySegment) VOICE_NAME.invokeExact(handle, i)));
            voices = List.copyOf(v);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    public TtsModel(String modelPath, String mmprojPath) { this(modelPath, mmprojPath, -1, 0); }

    public TtsModel(Weights w) { this(w.model(), w.mmproj()); }

    public int sampleRate() { return sampleRate; }

    /** The model's predefined voices, the default first. Empty for a model without any (Soprano, VoxCPM2). */
    public List<String> voices() { return voices; }

    /** Stops the speech in progress (from any thread); the call returns the audio so far. Only for streaming calls. */
    public void cancel() { cancelled = true; }

    /** Speaks text with a voice (null for the model's default). */
    public Audio speak(String text, String voice) { return speak(text, voice, Options.DEFAULT, null); }

    /** Speaks text, streaming the audio to onChunk as it's decoded; returns all of it. */
    public Audio speak(String text, String voice, Consumer<Audio> onChunk) { return speak(text, voice, Options.DEFAULT, onChunk); }

    /**
     * Speaks text with a voice (null for the model's default). onChunk (optional) streams the audio as it's decoded;
     * the returned Audio holds all of it either way.
     */
    public Audio speak(String text, String voice, Options o, Consumer<Audio> onChunk) {
        cancelled = false;
        try (Arena a = Arena.ofConfined()) {
            MemorySegment cb = MemorySegment.NULL;
            if (onChunk != null) {
                SINK.set(new Sink(this, onChunk));
                cb = NativeLibrary.linker().upcallStub(ON_AUDIO, AUDIO_CB, a);
            }
            MemorySegment pcm = a.allocate(ADDRESS), nFrames = a.allocate(JAVA_LONG), nGen = a.allocate(JAVA_INT);
            int rc = (int) SPEAK.invokeExact(handle, a.allocateFrom(text), cstr(a, voice), cstr(a, o.lang()), o.temperature(),
                    o.seed(), o.speed(), o.maxFrames(), cb, MemorySegment.NULL, pcm, nFrames, nGen);
            MemorySegment p = pcm.get(ADDRESS, 0);
            float[] f = new float[0];
            if (!p.equals(MemorySegment.NULL)) {
                f = p.reinterpret(nFrames.get(JAVA_LONG, 0) * channels * Float.BYTES).toArray(JAVA_FLOAT);
                FREE.invokeExact(p);
            }
            if (rc != 0) throw new IllegalStateException("tts failed (rc=" + rc + "): " + error());
            return Pcm.toAudio(f, channels, sampleRate);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { SINK.remove(); }
    }

    private static MemorySegment cstr(Arena a, String s) { return s == null ? MemorySegment.NULL : a.allocateFrom(s); }

    private static String error() {
        try { return NativeLibrary.string((MemorySegment) LAST_ERROR.invokeExact()); }
        catch (Throwable t) { return t.toString(); }
    }

    @Override public void close() {
        try { DESTROY.invokeExact(handle); } catch (Throwable t) { throw new RuntimeException(t); }
    }
}
