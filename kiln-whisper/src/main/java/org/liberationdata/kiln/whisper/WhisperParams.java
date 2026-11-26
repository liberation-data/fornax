package org.liberationdata.kiln.whisper;

import org.liberationdata.kiln.whisper.WhisperModel.Options;
import org.liberationdata.kiln.whisper.WhisperModel.Vad;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * {@link Options} as the native {@code kiln_whisper_params} (kiln_whisper.h), which the library copies into
 * whisper.cpp's own {@code whisper_full_params}. Kept apart from {@link WhisperModel} so it can be tested without the
 * native library; {@code WhisperParamsTest} checks the layout against the library's when it's there.
 */
final class WhisperParams {
    private WhisperParams() {}

    /** kiln_whisper_params, field for field, in declaration order. */
    static final StructLayout LAYOUT = MemoryLayout.structLayout(
            ADDRESS.withName("language"), ADDRESS.withName("initial_prompt"), ADDRESS.withName("vad_model_path"),
            JAVA_INT.withName("translate"), JAVA_INT.withName("n_threads"), JAVA_INT.withName("beam_size"),
            JAVA_FLOAT.withName("temperature"), JAVA_FLOAT.withName("temperature_inc"), JAVA_FLOAT.withName("entropy_thold"),
            JAVA_FLOAT.withName("logprob_thold"), JAVA_FLOAT.withName("no_speech_thold"),
            JAVA_INT.withName("max_text_ctx"), JAVA_INT.withName("suppress_nst"),
            JAVA_FLOAT.withName("vad_threshold"), JAVA_INT.withName("vad_min_speech_ms"), JAVA_INT.withName("vad_min_silence_ms"),
            JAVA_FLOAT.withName("vad_max_speech_s"), JAVA_INT.withName("vad_speech_pad_ms"), JAVA_FLOAT.withName("vad_samples_overlap"));

    /** The options in arena a. */
    static MemorySegment of(Arena a, Options o) {
        MemorySegment p = a.allocate(LAYOUT);
        p.set(ADDRESS, offset("language"), cstr(a, o.language()));
        p.set(ADDRESS, offset("initial_prompt"), cstr(a, o.initialPrompt()));
        p.set(JAVA_INT, offset("translate"), o.translate() ? 1 : 0);
        p.set(JAVA_INT, offset("n_threads"), o.threads());
        p.set(JAVA_INT, offset("beam_size"), o.beamSize());
        p.set(JAVA_FLOAT, offset("temperature"), o.temperature());
        p.set(JAVA_FLOAT, offset("temperature_inc"), o.temperatureIncrement());
        p.set(JAVA_FLOAT, offset("entropy_thold"), o.entropyThreshold());
        p.set(JAVA_FLOAT, offset("logprob_thold"), o.logProbThreshold());
        p.set(JAVA_FLOAT, offset("no_speech_thold"), o.noSpeechThreshold());
        p.set(JAVA_INT, offset("max_text_ctx"), o.conditionOnPreviousText() ? -1 : 0);
        p.set(JAVA_INT, offset("suppress_nst"), o.suppressNonSpeech() ? 1 : 0);
        Vad v = o.vad();
        p.set(ADDRESS, offset("vad_model_path"), v == null ? MemorySegment.NULL : a.allocateFrom(v.modelPath()));
        if (v != null) {
            p.set(JAVA_FLOAT, offset("vad_threshold"), v.threshold());
            p.set(JAVA_INT, offset("vad_min_speech_ms"), v.minSpeechMs());
            p.set(JAVA_INT, offset("vad_min_silence_ms"), v.minSilenceMs());
            p.set(JAVA_FLOAT, offset("vad_max_speech_s"), v.maxSpeechSeconds());
            p.set(JAVA_INT, offset("vad_speech_pad_ms"), v.speechPadMs());
            p.set(JAVA_FLOAT, offset("vad_samples_overlap"), v.overlapSeconds());
        }
        return p;
    }

    /**
     * When VAD finds no speech, whisper.cpp returns before it detects or sets the language, so its own is the previous
     * call's: report the one asked for instead, or null when it was to be detected.
     */
    static String languageWithoutSpeech(String requested, boolean multilingual) {
        if (!multilingual) return "en";
        return requested == null || requested.isEmpty() || requested.equals("auto") ? null : requested;
    }

    static long offset(String field) { return LAYOUT.byteOffset(MemoryLayout.PathElement.groupElement(field)); }

    static MemorySegment cstr(Arena a, String s) { return s == null ? MemorySegment.NULL : a.allocateFrom(s); }
}
