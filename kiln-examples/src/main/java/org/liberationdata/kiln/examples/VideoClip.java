package org.liberationdata.kiln.examples;

import org.liberationdata.kiln.HfCache;
import org.liberationdata.kiln.sd.DiffusionModel;

import javax.imageio.ImageIO;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/**
 * A video clip with sound from LTX-2.5 through stable-diffusion.cpp.
 * Usage: VideoClip <out.mp4> "<prompt>" [first-frame.png] [last-frame.png]
 *   [-Dwidth=768 -Dheight=448 -Dframes=49 -Dfps=24 -Dsteps=8 -Dcfg=1 -Dseed=7 -Dmodel=distilled|dev -Drecipe=two-stage]
 *   [-Dsong=song.wav -Dsong.start=48]
 * -Drecipe=two-stage is the ComfyUI "Image to Video (LTX-2.5)" recipe Endless Country used: a draft at width x height
 * with 8 manual sigmas and the first frame at strength 0.7, the x2 latent upscaler, then a refine at twice the size.
 * Files: from -Dltx.dir=<folder with diffusion_models/ text_encoders/ vae/> when there, else the Hugging Face cache.
 * Needs ffmpeg on the PATH to write the .mp4 (frames and audio are kept beside it either way).
 */
public class VideoClip {
    static final String BF16_TO_F32 = ".*\\.bias=f32,.*_norm\\.weight=f32,.*to_gate_logits\\.weight=f32,"
            + ".*adaln_single.*\\.weight=f32,.*(patchify_proj|proj_out)\\.weight=f32,"
            + ".*(learnable_registers|keyframes_abs_pos_embedding)=f32";

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: VideoClip <out.mp4> \"<prompt>\" [first-frame.png] [last-frame.png]");
            System.exit(2);
        }
        Path out = Path.of(args[0]);
        String prompt = args[1];
        String variant = System.getProperty("model", "distilled");
        String dir = System.getProperty("ltx.dir");
        var cfg = DiffusionModel.Config.builder()
                .diffusionModel(file(dir, "diffusion_models/ltx-2.5-22b-" + variant + "-transformer-comfy-int8-convrot.safetensors"))
                .llm(file(dir, "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"))
                .vae(file(dir, "vae/ltx-2.5-video-vae-conv-bf16.safetensors"))
                // -Daudio=false skips LTX's own soundtrack (Endless Country lays the song over every clip anyway).
                .set("audio_vae_path", Boolean.parseBoolean(System.getProperty("audio", "true"))
                        ? file(dir, "vae/ltx-2.5-audio-vae-bf16.safetensors") : "")
                // ComfyUI's int8 LTX keeps biases, norms and embedders in bf16; ggml's Vulkan and CPU backends can't
                // add bf16 to f32, so widen those (~0.9 GB) to f32 at load. The int8 linears are untouched.
                .set("tensor_type_rules", System.getProperty("ltx.rules", BF16_TO_F32))
                .build();
        int w = Integer.getInteger("width", 768), h = Integer.getInteger("height", 448);
        int frames = Integer.getInteger("frames", 49), fps = Integer.getInteger("fps", 24);
        int steps = Integer.getInteger("steps", variant.equals("distilled") ? 8 : 20);
        double guidance = Double.parseDouble(System.getProperty("cfg", variant.equals("distilled") ? "1" : "3"));
        var req = new DiffusionModel.VideoRequest(prompt, w, h, frames, fps, steps, guidance, Long.getLong("seed", 7))
                .withExtra(Map.of("sample.method", "euler", "sample.scheduler", "ltx2"));
        if ("two-stage".equals(System.getProperty("recipe"))) {
            var extra = new java.util.HashMap<String, Object>(req.extra());
            extra.put("sample.method", "euler_a");
            extra.put("sample.sigmas", new float[] {1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f, 0.0f});
            extra.put("strength", Double.parseDouble(System.getProperty("strength", "0.7")));
            // -Dhires=false renders the draft stage alone, to tell its artifacts from the refine's.
            extra.put("hires.enabled", Boolean.parseBoolean(System.getProperty("hires", "true")));
            // -Dupscaler=interpolate swaps the learned x2 latent upscaler for plain interpolation (a diagnostic).
            // sd.cpp only runs the learned upscaler when hires.upscaler is "model"; otherwise it silently interpolates the
            // latent bilinearly and ignores hires.model_path, and refining an interpolated latent ghosts badly.
            if (!"interpolate".equals(System.getProperty("upscaler"))) {
                extra.put("hires.upscaler", "model");
                extra.put("hires.model_path", file(dir, "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"));
            }
            // -Drefine=scheduler lets sd.cpp's scheduler pick 3 refine steps at strength 0.85 instead of the manual sigmas.
            if ("scheduler".equals(System.getProperty("refine"))) {
                extra.put("hires.steps", 3);
                extra.put("hires.denoising_strength", 0.85);
            } else {
                // -Drefine=renoise starts the refine from pure noise (sigma 1), a diagnostic.
                extra.put("hires.sigmas", "renoise".equals(System.getProperty("refine"))
                        ? new float[] {1.0f, 0.725f, 0.4219f, 0.0f} : new float[] {0.85f, 0.725f, 0.4219f, 0.0f});
            }
            // Tiled decode, as the ComfyUI workflow does: short GPU dispatches stay under the driver's compute timeout.
            extra.put("vae_tiling.enabled", true);
            req = req.withExtra(extra);
        }
        // LTX was trained on compressed frames; ComfyUI's LTXVPreprocess round-trips each keyframe through H.264.
        int crf = Integer.getInteger("preprocess", "two-stage".equals(System.getProperty("recipe")) ? 18 : 0);
        if (args.length > 2) req = req.withInitImage(keyframe(Path.of(args[2]), crf));
        if (args.length > 3) req = req.withEndImage(keyframe(Path.of(args[3]), crf));
        // -Dsong=song.wav -Dsong.start=48: audio to video. The clip-length section from song.start is held as the
        // soundtrack and the picture is generated to it (a singer's lips, a strum).
        String song = System.getProperty("song");
        if (song != null) req = req.withAudio(Path.of(song), Double.parseDouble(System.getProperty("song.start", "0")));

        DiffusionModel.onProgress((s, n, t) -> System.out.printf("  step %d/%d (%.2fs/it)%n", s, n, t));
        long t0 = System.nanoTime();
        try (var model = new DiffusionModel(cfg)) {
            System.out.printf("[kiln] loaded %s in %.1fs%n", model.modelVersion(), (System.nanoTime() - t0) / 1e9);
            long g0 = System.nanoTime();
            var video = model.generateVideo(req);
            System.out.printf("[kiln] %d frames at %d fps (%.1fs)%s in %.1fs%n", video.frames().size(), video.fps(),
                    video.seconds(), video.audio() != null ? " with audio" : "", (System.nanoTime() - g0) / 1e9);
            write(video, out);
        }
    }

    /** The keyframe, optionally through one H.264 frame at the given CRF (0 = as is), as LTXVPreprocess does. */
    static java.awt.image.BufferedImage keyframe(Path image, int crf) throws Exception {
        if (crf <= 0) return ImageIO.read(image.toFile());
        Path tmp = Files.createTempDirectory("kiln-ltx-preprocess");
        Path mp4 = tmp.resolve("frame.mp4"), png = tmp.resolve("frame.png");
        run("ffmpeg", "-y", "-loglevel", "error", "-i", image.toString(), "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", String.valueOf(crf), "-pix_fmt", "yuv420p", mp4.toString());
        run("ffmpeg", "-y", "-loglevel", "error", "-i", mp4.toString(), "-frames:v", "1", png.toString());
        return ImageIO.read(png.toFile());
    }

    private static void run(String... cmd) throws Exception {
        if (new ProcessBuilder(cmd).inheritIO().start().waitFor() != 0) throw new IllegalStateException("failed: " + String.join(" ", cmd));
    }

    static void write(DiffusionModel.Video video, Path out) throws Exception {
        String base = out.toString().replaceAll("\\.mp4$", "");
        Path frames = Files.createDirectories(Path.of(base + "-frames"));
        for (int i = 0; i < video.frames().size(); i++)
            ImageIO.write(video.frames().get(i), "png", frames.resolve(String.format("%05d.png", i)).toFile());
        Path wav = video.audio() != null ? Path.of(base + ".wav") : null;
        if (wav != null) video.audio().writeWav(wav.toFile());
        var cmd = new java.util.ArrayList<>(java.util.List.of("ffmpeg", "-y", "-loglevel", "error",
                "-framerate", String.valueOf(video.fps()), "-i", frames.resolve("%05d.png").toString()));
        if (wav != null) cmd.addAll(java.util.List.of("-i", wav.toString(), "-c:a", "aac", "-shortest"));
        cmd.addAll(java.util.List.of("-c:v", "libx264", "-pix_fmt", "yuv420p", out.toString()));
        try {
            int rc = new ProcessBuilder(cmd).inheritIO().start().waitFor();
            System.out.println(rc == 0 ? "[kiln] wrote " + out : "[kiln] ffmpeg failed; frames are in " + frames);
        } catch (java.io.IOException e) {
            System.out.println("[kiln] no ffmpeg; frames are in " + frames + (wav != null ? ", audio in " + wav : ""));
        }
    }

    private static String file(String dir, String rel) {
        if (dir != null && Files.exists(Path.of(dir, rel))) return Path.of(dir, rel).toString();
        return HfCache.require("Lightricks/LTX-2.5", rel).toString();
    }
}
