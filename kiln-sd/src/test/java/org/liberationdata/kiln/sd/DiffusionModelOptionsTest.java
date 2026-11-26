package org.liberationdata.kiln.sd;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.kiln.Audio;

import java.nio.file.Path;
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

    private static DiffusionModel.VideoRequest clip() {
        return new DiffusionModel.VideoRequest("a singer", 336, 224, 49, 24, 8, 1, 7);
    }

    @Test
    void withAudioCutsAClipLengthSectionFromTheStart() {
        float[] l = new float[48000 * 6], r = new float[48000 * 6];
        for (int i = 0; i < l.length; i++) { l[i] = i / (float) l.length; r[i] = -l[i]; }
        var req = clip().withAudio(new Audio(l, r, 48000), 2.0);
        Audio a = req.refAudio();
        assertEquals(48000, a.sampleRate());
        assertEquals(98000, a.left().length);                // 49 frames at 24 fps = 2.0417 s
        assertEquals(l[96000], a.left()[0]);
        assertEquals(r[96000 + 500], a.right()[500]);
        assertTrue(a.seconds() >= req.seconds());
    }

    @Test
    void withAudioPadsWithSilencePastTheEnd() {
        var req = clip().withAudio(Audio.ofMono(new float[] {0.5f, 0.5f, 0.5f, 0.5f}, 4), 0);
        assertEquals(9, req.refAudio().left().length);       // ceil(49 / 24 * 4)
        assertEquals(0.5f, req.refAudio().left()[3]);
        assertEquals(0f, req.refAudio().left()[4]);
    }

    @Test
    void withAudioReadsAWav(@TempDir Path dir) throws Exception {
        Path wav = dir.resolve("song.wav");
        Audio.ofMono(new float[16000 * 3], 16000).writeWav(wav.toFile());
        var req = clip().withAudio(wav, 0.5);
        assertEquals(16000, req.refAudio().sampleRate());
        assertEquals((int) Math.ceil(49 / 24.0 * 16000), req.refAudio().left().length);
    }

    @Test
    void theRequestKeepsItsAudioThroughTheOtherWithers() {
        var req = clip().withAudio(Audio.ofMono(new float[48000 * 3], 48000), 0);
        var later = req.withExtra(Map.of("strength", 0.7)).withInitImage(null);
        assertSame(req.refAudio(), later.refAudio());
    }

    @Test
    void videoOptionsMapTheRequestAndLetExtraOverride() {
        var req = clip().withExtra(Map.of("sample.steps", 3, "sample.method", "euler"));
        var o = DiffusionModel.videoOptions(req);
        assertEquals("a singer", o.get("prompt"));
        assertEquals(336, o.get("width"));
        assertEquals(224, o.get("height"));
        assertEquals(49, o.get("video_frames"));
        assertEquals(24, o.get("fps"));
        assertEquals(3, o.get("sample.steps"));
        assertEquals("euler", o.get("sample.method"));
        assertEquals(7L, o.get("seed"));
        assertFalse(o.containsKey("ref_audios"), "audio goes in as a tensor, not an option");
    }

    @Test
    void audioIsInterleavedChannelFirst() {
        float[] pcm = DiffusionModel.interleave(new Audio(new float[] {1, 2}, new float[] {-1, -2}, 8000));
        assertArrayEquals(new float[] {1, -1, 2, -2}, pcm);
    }

    @Test
    void interpolationWithoutAModelIsLeftAlone() {
        Map<String, Object> o = new HashMap<>(Map.of("hires.enabled", true, "hires.upscaler", "latent"));
        DiffusionModel.checkHires(o);
        assertEquals("latent", o.get("hires.upscaler"));
        assertFalse(o.containsKey("hires.model_path"));
    }
}
