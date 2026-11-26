package org.liberationdata.kiln.demucs;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** DemucsModel's nested types, which load without the native library. */
class OptionsTest {
    @Test
    void optionsRefuseWhatStemsCppWould() {
        var o = DemucsModel.Options.DEFAULT;
        assertThrows(IllegalArgumentException.class, () -> o.withShifts(-1));
        assertThrows(IllegalArgumentException.class, () -> o.withShifts(11));
        assertThrows(IllegalArgumentException.class, () -> o.withOverlap(1));
        assertThrows(IllegalArgumentException.class, () -> o.withOverlap(Float.NaN));
        var ok = o.withShifts(10).withOverlap(0.5f).withSeed(7);
        assertEquals(new DemucsModel.Options(10, 0.5f, 7), ok);
    }

    @Test
    void weightsByName() {
        assertEquals(DemucsModel.Weights.HTDEMUCS, DemucsModel.Weights.named("htdemucs"));
        assertEquals(DemucsModel.Weights.HTDEMUCS_6S, DemucsModel.Weights.named("htdemucs-6s"));
        assertEquals(DemucsModel.Weights.HTDEMUCS_FT, DemucsModel.Weights.named("HTDEMUCS_ft"));
        assertEquals(DemucsModel.Weights.HTDEMUCS_F16, DemucsModel.Weights.named("htdemucs-f16"));
        assertThrows(IllegalArgumentException.class, () -> DemucsModel.Weights.named("mdx"));
    }
}
