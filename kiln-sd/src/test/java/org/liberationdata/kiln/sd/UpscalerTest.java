package org.liberationdata.kiln.sd;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/** How an upscale factor maps to passes of the model. No native library needed. */
class UpscalerTest {

    @Test
    void powersOfTheScaleBecomePasses() {
        assertEquals(1, Upscaler.repeats(4, 4));
        assertEquals(2, Upscaler.repeats(4, 16));
        assertEquals(3, Upscaler.repeats(2, 8));
    }

    @Test
    void otherFactorsAreRefusedNotApproximated() {
        for (int f : new int[] {0, 1, 2, 3, 8, 12, 32, -4, Integer.MAX_VALUE})
            assertThrows(IllegalArgumentException.class, () -> Upscaler.repeats(4, f), "factor " + f);
    }

    @Test
    void aModelThatReportsNoScaleIsAnError() {
        assertThrows(IllegalStateException.class, () -> Upscaler.repeats(1, 4));
    }
}
