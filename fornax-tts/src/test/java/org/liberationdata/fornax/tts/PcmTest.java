package org.liberationdata.fornax.tts;

import org.junit.jupiter.api.Test;

import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;

class PcmTest {
    @Test
    void monoPlaysOnBothSides() {
        var a = Pcm.toAudio(new float[] {0.1f, -0.2f, 0.3f}, 1, 24000);
        assertArrayEquals(new float[] {0.1f, -0.2f, 0.3f}, a.left());
        assertArrayEquals(a.left(), a.right());
        assertNotSame(a.left(), a.right());
        assertEquals(24000, a.sampleRate());
        assertEquals(3 / 24000.0, a.seconds(), 1e-12);
    }

    @Test
    void stereoIsDeinterleaved() {
        var a = Pcm.toAudio(new float[] {1, -1, 2, -2}, 2, 48000);
        assertArrayEquals(new float[] {1, 2}, a.left());
        assertArrayEquals(new float[] {-1, -2}, a.right());
    }

    @Test
    void weightsByName() {
        assertEquals(TtsModel.Weights.LFM3_AUDIO, TtsModel.Weights.named("lfm3-audio"));
        assertEquals(TtsModel.Weights.KOKORO, TtsModel.Weights.named("Kokoro"));
    }

    @Test
    void weightsByNameInATurkishLocale() {
        Locale saved = Locale.getDefault();
        try {
            Locale.setDefault(Locale.forLanguageTag("tr-TR"));
            assertEquals(TtsModel.Weights.LFM3_AUDIO, TtsModel.Weights.named("lfm3-audio"));
            assertEquals(TtsModel.Weights.TINY_VOCA, TtsModel.Weights.named("tiny-voca"));
        } finally {
            Locale.setDefault(saved);
        }
    }
}
