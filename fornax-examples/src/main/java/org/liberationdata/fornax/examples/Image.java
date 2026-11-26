package org.liberationdata.fornax.examples;

import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.sd.DiffusionModel;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;
import java.util.Map;

/**
 * Text-to-image (or edit, with a reference image) with stable-diffusion.cpp.
 * Usage: Image <klein|qwen-image|qwen-edit> <out.png> "<prompt>" [ref.png]
 *   [-Dsteps=N -Dwidth=W -Dheight=H -Dseed=S -Drepeat=N]
 * With -Drepeat=2 the second image reuses the loaded weights.
 */
public class Image {
    /** A model's files and the steps and CFG Fornax's examples use with it. */
    record Preset(DiffusionModel.Config.Builder config, int steps, double cfg) {}

    static Preset preset(String name) {
        var cfg = DiffusionModel.Config.builder();
        return switch (name) {
            case "klein" -> {  // FLUX.2 [klein] base 4B
                cfg.diffusionModel(path("black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors"))
                   .vae(path("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"))
                   .llm(path("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors"));
                yield new Preset(cfg, 20, 4.0);
            }
            case "qwen-image" -> {  // Qwen-Image 2512
                cfg.diffusionModel(path("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"))
                   .vae(path("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"))
                   .llm(qwenVl());
                yield new Preset(cfg, 20, 2.5);
            }
            case "qwen-edit" -> {  // Qwen-Image-Edit 2511 (needs a reference image)
                cfg.diffusionModel(path("Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors"))
                   .vae(path("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"))
                   .llm(qwenVl());
                yield new Preset(cfg, 20, 2.5);
            }
            case "sd15" -> {  // Stable Diffusion 1.5: small, and the family most ControlNets are for
                cfg.set("model_path", path("Comfy-Org/stable-diffusion-v1-5-archive", "v1-5-pruned-emaonly-fp16.safetensors"));
                yield new Preset(cfg, 20, 7.0);
            }
            default -> throw new IllegalArgumentException("unknown preset " + name + " (klein, qwen-image, qwen-edit, sd15)");
        };
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("usage: Image <klein|qwen-image|qwen-edit> <out.png> \"<prompt>\" [ref.png]");
            System.exit(2);
        }
        Preset preset = preset(args[0]);
        File out = new File(args[1]);
        String prompt = args[2];
        List<BufferedImage> refs = args.length > 3 ? List.of(ImageIO.read(new File(args[3]))) : List.of();
        int w = Integer.getInteger("width", refs.isEmpty() ? 1024 : refs.get(0).getWidth() / 16 * 16);
        int h = Integer.getInteger("height", refs.isEmpty() ? 1024 : refs.get(0).getHeight() / 16 * 16);
        int steps = Integer.getInteger("steps", preset.steps());

        DiffusionModel.onProgress((s, n, t) -> System.out.printf("  step %d/%d (%.2fs/it)%n", s, n, t));
        long t0 = System.nanoTime();
        try (var model = new DiffusionModel(preset.config().build())) {
            System.out.printf("[fornax] loaded %s in %.1fs%n", model.modelVersion(), (System.nanoTime() - t0) / 1e9);
            int repeat = Integer.getInteger("repeat", 1);
            long seed = Long.getLong("seed", 7);
            for (int i = 0; i < repeat; i++) {
                long g0 = System.nanoTime();
                var images = model.generate(new DiffusionModel.Request(prompt, w, h, steps, preset.cfg(), seed + i, refs, Map.of()));
                File f = repeat == 1 ? out : new File(out.getPath().replace(".png", "-" + i + ".png"));
                ImageIO.write(images.get(0), "png", f);
                System.out.printf("[fornax] image %d: %dx%d, %d steps in %.1fs -> %s%n", i, w, h, steps, (System.nanoTime() - g0) / 1e9, f);
            }
        }
    }

    static String path(String repo, String file) { return HfCache.require(repo, file).toString(); }

    private static String qwenVl() {
        String f = "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors";
        return HfCache.requireAny("Comfy-Org/Qwen-Image_ComfyUI", f, "Comfy-Org/HunyuanVideo_1.5_repackaged", f).toString();
    }
}
