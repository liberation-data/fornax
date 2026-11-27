package org.liberationdata.fornax.snippets;

// --8<-- [start:ltx]
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.sd.DiffusionModel;

import java.util.Map;
// --8<-- [end:ltx]
// --8<-- [start:keyframes]
import javax.imageio.ImageIO;
import java.io.File;
// --8<-- [end:keyframes]
// --8<-- [start:audio]
import java.nio.file.Path;
// --8<-- [end:audio]
// --8<-- [start:hires]
import java.util.HashMap;
// --8<-- [end:hires]

/** The snippets of website/src/content/docs/docs/guides/video.md: compiled, never run. */
class Video {
    void ltx() {
        // --8<-- [start:ltx]
        String ltx = "Lightricks/LTX-2.5";
        var config = DiffusionModel.Config.builder()
                .diffusionModel(HfCache.require(ltx, "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors").toString())
                .llm(HfCache.require(ltx, "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors").toString())
                .vae(HfCache.require(ltx, "vae/ltx-2.5-video-vae-conv-bf16.safetensors").toString())
                .set("audio_vae_path", HfCache.require(ltx, "vae/ltx-2.5-audio-vae-bf16.safetensors").toString())
                .build();

        try (var model = new DiffusionModel(config)) {
            //                                                    prompt, width, height, frames, fps, steps, cfg, seed
            var request = new DiffusionModel.VideoRequest("a potter at a wheel, warm light, the hum of the wheel",
                    768, 448, 49, 24, 8, 1.0, 7)
                    .withExtra(Map.of("sample.method", "euler", "sample.scheduler", "ltx2"));
            DiffusionModel.Video video = model.generateVideo(request);
            System.out.printf("%d frames at %d fps (%.1f s)%s%n", video.frames().size(), video.fps(), video.seconds(),
                    video.audio() != null ? ", with sound" : "");
        }
        // --8<-- [end:ltx]
    }

    void keyframes() throws Exception {
        // --8<-- [start:keyframes]
        var request = new DiffusionModel.VideoRequest("the vase turns slowly on the wheel", 768, 448, 49, 24, 8, 1.0, 7)
                .withExtra(Map.of("sample.method", "euler", "sample.scheduler", "ltx2"))
                .withInitImage(ImageIO.read(new File("first.png")))
                .withEndImage(ImageIO.read(new File("last.png")));
        // --8<-- [end:keyframes]
    }

    void audio(DiffusionModel model) throws Exception {
        // --8<-- [start:audio]
        var request = new DiffusionModel.VideoRequest(
                "close-up of a singer at a vintage microphone, his jaw opening and closing in time with the words. "
                        + "Audio: a man singing a country song.", 768, 448, 97, 24, 8, 1.0, 7)
                .withExtra(Map.of("sample.method", "euler", "sample.scheduler", "ltx2"))
                .withInitImage(ImageIO.read(new File("singer.png")))
                .withAudio(Path.of("song.wav"), 25.9);   // 97 frames at 24 fps: 25.9 s to 29.9 s
        var video = model.generateVideo(request);     // video.audio() is the song section
        // --8<-- [end:audio]
    }

    void hires(String ltx) throws Exception {
        // --8<-- [start:hires]
        var extra = new HashMap<String, Object>();
        extra.put("sample.method", "euler_a");
        extra.put("sample.scheduler", "ltx2");
        extra.put("sample.sigmas", new float[] {1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f, 0.0f});
        extra.put("strength", 0.7);
        extra.put("hires.enabled", true);
        extra.put("hires.upscaler", "model");
        extra.put("hires.model_path", HfCache.require(ltx, "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors").toString());
        extra.put("hires.sigmas", new float[] {0.85f, 0.725f, 0.4219f, 0.0f});
        extra.put("vae_tiling.enabled", true);    // short GPU dispatches stay under the driver's compute timeout

        var request = new DiffusionModel.VideoRequest("a potter at a wheel, warm light", 672, 448, 121, 24, 8, 1.0, 7)
                .withInitImage(ImageIO.read(new File("first.png")))
                .withExtra(extra);
        // --8<-- [end:hires]
    }
}
