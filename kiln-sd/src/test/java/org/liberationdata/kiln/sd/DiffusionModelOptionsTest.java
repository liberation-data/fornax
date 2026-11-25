package org.liberationdata.kiln.sd;

import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Option checks that need no native library: these run on any machine. */
class DiffusionModelOptionsTest {

    @Test
    void aHiresModelPathSelectsTheModelUpscaler() {
        Map<String, Object> o = new HashMap<>(Map.of("hires.enabled", true, "hires.model_path", "/m/upscaler.safetensors"));
        DiffusionModel.checkHires(o);
        assertEquals("model", o.get("hires.upscaler"));
    }

    @Test
    void anExplicitModelUpscalerIsKept() {
        Map<String, Object> o = new HashMap<>(Map.of("hires.upscaler", "model", "hires.model_path", "/m/u.safetensors"));
        DiffusionModel.checkHires(o);
        assertEquals("model", o.get("hires.upscaler"));
    }

    @Test
    void aModelPathWithAnInterpolatingUpscalerIsRefused() {
        Map<String, Object> o = new HashMap<>(Map.of("hires.upscaler", "latent", "hires.model_path", "/m/u.safetensors"));
        var e = assertThrows(IllegalArgumentException.class, () -> DiffusionModel.checkHires(o));
        assertTrue(e.getMessage().contains("ignores the model"), e.getMessage());
    }

    @Test
    void interpolationWithoutAModelIsLeftAlone() {
        Map<String, Object> o = new HashMap<>(Map.of("hires.enabled", true, "hires.upscaler", "latent"));
        DiffusionModel.checkHires(o);
        assertEquals("latent", o.get("hires.upscaler"));
        assertFalse(o.containsKey("hires.model_path"));
    }
}
