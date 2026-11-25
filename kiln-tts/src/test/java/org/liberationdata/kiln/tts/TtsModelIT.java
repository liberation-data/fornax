package org.liberationdata.kiln.tts;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.Audio;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Runs against Kokoro-82M when it's in the Hugging Face cache and the native library is built; skipped otherwise. */
class TtsModelIT {
    static final TtsModel.Weights W = TtsModel.Weights.KOKORO;
    static final String TEXT = "Kiln runs open generative models inside your Java application.";
    static TtsModel model;

    @BeforeAll
    static void load() {
        assumeTrue(W.downloaded(), "model not downloaded: hf download " + W.repo);
        model = new TtsModel(W);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    static double rms(float[] x) {
        double s = 0;
        for (float v : x) s += v * v;
        return Math.sqrt(s / Math.max(1, x.length));
    }

    @Test
    void listsItsVoices() {
        assertTrue(model.voices().contains("am_michael"), model.voices().toString());
        assertEquals("af_heart", model.voices().getFirst());
        assertEquals(24000, model.sampleRate());
    }

    @Test
    void speaksASentence() {
        Audio a = model.speak(TEXT, "bm_george");
        // 9 words at a narrator's 2-3.5 words a second
        assertTrue(a.seconds() > 2 && a.seconds() < 8, "seconds: " + a.seconds());
        assertTrue(rms(a.left()) > 0.01, "rms: " + rms(a.left()));
    }

    @Test
    void streamsTheSameAudio() {
        List<Audio> chunks = new ArrayList<>();
        Audio a = model.speak(TEXT, "af_heart", TtsModel.Options.DEFAULT.withSeed(1), chunks::add);
        assertFalse(chunks.isEmpty());
        assertEquals(a.left().length, chunks.stream().mapToInt(c -> c.left().length).sum());
        assertTrue(rms(a.left()) > 0.01);
    }

    @Test
    void rejectsAnUnknownVoice() {
        var e = assertThrows(IllegalStateException.class, () -> model.speak(TEXT, "nobody"));
        assertTrue(e.getMessage().contains("am_michael"), e.getMessage());
    }
}
