package org.liberationdata.kiln.whisper;

import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.whisper.WhisperModel.Options;
import org.liberationdata.kiln.whisper.WhisperModel.Vad;
import org.liberationdata.kiln.whisper.WhisperModel.VadWeights;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Options as kiln_whisper_params: the Java layout against the C struct (through the native library, skipped without
 * it, and against the LP64 offsets without it), and every option landing in its field.
 */
class WhisperParamsTest {
    static final List<String> FIELDS = List.of("language", "initial_prompt", "vad_model_path", "translate", "n_threads",
            "beam_size", "temperature", "temperature_inc", "entropy_thold", "logprob_thold", "no_speech_thold",
            "max_text_ctx", "suppress_nst", "vad_threshold", "vad_min_speech_ms", "vad_min_silence_ms",
            "vad_max_speech_s", "vad_speech_pad_ms", "vad_samples_overlap");

    static long[] javaLayout() {
        long[] out = new long[FIELDS.size() + 1];
        out[0] = WhisperParams.LAYOUT.byteSize();
        for (int i = 0; i < FIELDS.size(); i++) out[i + 1] = WhisperParams.offset(FIELDS.get(i));
        return out;
    }

    @Test
    void layoutListsTheFieldsInOrder() {
        assertEquals(FIELDS, WhisperParams.LAYOUT.memberLayouts().stream().map(l -> l.name().orElseThrow()).toList());
    }

    @Test
    void layoutIsTheLp64One() {
        // three pointers, then sixteen 4-byte fields, no padding
        long[] expected = {88, 0, 8, 16, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84};
        assertArrayEquals(expected, javaLayout());
    }

    @Test
    void layoutMatchesTheNativeStruct() {
        String allowCpu = System.setProperty("kiln.allowCpu", "true");  // no model runs; CI runners have no GPU
        long[] nativeLayout;
        try { nativeLayout = WhisperModel.nativeParamsLayout(); }
        catch (Throwable t) { assumeTrue(false, "libkiln_whisper not found: " + t); return; }
        finally {
            if (allowCpu == null) System.clearProperty("kiln.allowCpu");
            else System.setProperty("kiln.allowCpu", allowCpu);
        }
        assertArrayEquals(javaLayout(), nativeLayout);
    }

    @Test
    void defaultsAreWhisperCpps() {
        Options o = Options.DEFAULT;
        assertTrue(o.conditionOnPreviousText());
        assertFalse(o.suppressNonSpeech());
        assertEquals(0.2f, o.temperatureIncrement());
        assertEquals(2.4f, o.entropyThreshold());
        assertEquals(-1f, o.logProbThreshold());
        assertEquals(0.6f, o.noSpeechThreshold());
        assertNull(o.vad());
        assertEquals(5, o.beamSize());
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = WhisperParams.of(a, o);
            assertEquals(MemorySegment.NULL, p.get(ADDRESS, WhisperParams.offset("language")));
            assertEquals(MemorySegment.NULL, p.get(ADDRESS, WhisperParams.offset("vad_model_path")));
            assertEquals(-1, p.get(JAVA_INT, WhisperParams.offset("max_text_ctx")));  // whisper.cpp's default
            assertEquals(-1f, p.get(JAVA_FLOAT, WhisperParams.offset("temperature")));
        }
    }

    @Test
    void everyOptionLandsInItsField() {
        Vad vad = Vad.of("/models/vad.bin").withThreshold(0.4f).withMinSpeechMs(200).withMinSilenceMs(300)
                .withMaxSpeechSeconds(20f).withSpeechPadMs(50).withOverlapSeconds(0.2f);
        Options o = Options.DEFAULT.withLanguage("de").withTranslate(true).withInitialPrompt("Kiln").withThreads(6)
                .withBeamSize(1).withTemperature(0.1f).withTemperatureIncrement(0.3f).withEntropyThreshold(2.8f)
                .withLogProbThreshold(-0.8f).withNoSpeechThreshold(0.5f).withConditionOnPreviousText(false)
                .withSuppressNonSpeech(true).withVad(vad);
        try (Arena a = Arena.ofConfined()) {
            MemorySegment p = WhisperParams.of(a, o);
            assertEquals("de", string(p, "language"));
            assertEquals("Kiln", string(p, "initial_prompt"));
            assertEquals("/models/vad.bin", string(p, "vad_model_path"));
            assertEquals(1, i(p, "translate"));
            assertEquals(6, i(p, "n_threads"));
            assertEquals(1, i(p, "beam_size"));
            assertEquals(0.1f, f(p, "temperature"));
            assertEquals(0.3f, f(p, "temperature_inc"));
            assertEquals(2.8f, f(p, "entropy_thold"));
            assertEquals(-0.8f, f(p, "logprob_thold"));
            assertEquals(0.5f, f(p, "no_speech_thold"));
            assertEquals(0, i(p, "max_text_ctx"));
            assertEquals(1, i(p, "suppress_nst"));
            assertEquals(0.4f, f(p, "vad_threshold"));
            assertEquals(200, i(p, "vad_min_speech_ms"));
            assertEquals(300, i(p, "vad_min_silence_ms"));
            assertEquals(20f, f(p, "vad_max_speech_s"));
            assertEquals(50, i(p, "vad_speech_pad_ms"));
            assertEquals(0.2f, f(p, "vad_samples_overlap"));
        }
    }

    @Test
    void eachWitherChangesOnlyItsOwnOption() {
        Options o = Options.DEFAULT.withEntropyThreshold(2.8f);
        assertEquals(Options.DEFAULT, o.withEntropyThreshold(2.4f));
        assertEquals(new Options(null, false, null, 0, 5, -1, false, false, 0.2f, 2.4f, -1f, 0.6f, null),
                Options.DEFAULT.withConditionOnPreviousText(false));
        assertEquals(Options.DEFAULT, new Options(null, false, null, 0, 5, -1));
    }

    @Test
    void vadDefaultsAreWhisperCpps() {
        Vad v = Vad.of("vad.bin");
        assertEquals(new Vad("vad.bin", 0.5f, 250, 100, Float.MAX_VALUE, 30, 0.1f), v);
        assertThrows(IllegalArgumentException.class, () -> Vad.of((String) null));
        assertThrows(IllegalArgumentException.class, () -> Vad.of(""));
    }

    @Test
    void vadWeightsByName() {
        assertEquals(VadWeights.SILERO_V6_2_0, VadWeights.named("silero-v6.2.0"));
        assertEquals(VadWeights.SILERO_V5_1_2, VadWeights.named("silero_v5_1_2"));
        assertEquals("ggml-org/whisper-vad", VadWeights.SILERO_V6_2_0.repo);
        assertEquals("ggml-silero-v6.2.0.bin", VadWeights.SILERO_V6_2_0.file);
    }

    static String string(MemorySegment p, String field) {
        return p.get(ADDRESS, WhisperParams.offset(field)).reinterpret(Long.MAX_VALUE).getString(0);
    }
    static int i(MemorySegment p, String field) { return p.get(JAVA_INT, WhisperParams.offset(field)); }
    static float f(MemorySegment p, String field) { return p.get(JAVA_FLOAT, WhisperParams.offset(field)); }
}
