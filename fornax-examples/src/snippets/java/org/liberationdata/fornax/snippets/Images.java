package org.liberationdata.fornax.snippets;

// --8<-- [start:flux]
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.sd.DiffusionModel;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
// --8<-- [end:flux]
// --8<-- [start:edit]
import java.util.List;
import java.util.Map;
// --8<-- [end:edit]
// --8<-- [start:upscale]
import org.liberationdata.fornax.sd.Upscaler;
// --8<-- [end:upscale]

/** The snippets of website/src/content/docs/docs/guides/images.md: compiled, never run. */
class Images {
    void flux() throws Exception {
        // --8<-- [start:flux]
        var config = DiffusionModel.Config.builder()
                .diffusionModel(HfCache.require("black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors").toString())
                .vae(HfCache.require("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors").toString())
                .llm(HfCache.require("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors").toString())
                .build();

        try (var images = new DiffusionModel(config)) {
            //                                              prompt, width, height, steps, cfg, seed
            var request = new DiffusionModel.Request("a neon sign that says FORNAX", 768, 768, 12, 4.0, 7);
            BufferedImage img = images.generate(request).get(0);
            ImageIO.write(img, "png", new File("fornax.png"));
        }
        // --8<-- [end:flux]
    }

    void fluxByName() {
        // --8<-- [start:flux-by-name]
        try (var images = new DiffusionModel(DiffusionModel.Config.builder().model("flux2-klein").build())) {
            BufferedImage img = images.generate(new DiffusionModel.Request("a neon sign that says FORNAX", 768, 768, 12, 4.0, 7)).get(0);
        }
        // --8<-- [end:flux-by-name]
    }

    void qwenImage() {
        // --8<-- [start:qwen-image]
        String repo = "Comfy-Org/Qwen-Image_ComfyUI";
        var config = DiffusionModel.Config.builder()
                .diffusionModel(HfCache.require(repo, "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors").toString())
                .vae(HfCache.require(repo, "split_files/vae/qwen_image_vae.safetensors").toString())
                .llm(HfCache.require(repo, "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors").toString())
                .build();

        try (var images = new DiffusionModel(config)) {
            BufferedImage img = images.generate(new DiffusionModel.Request(
                    "a hand-painted shop sign reading 'Fornax & Co. Pottery'", 1024, 1024, 20, 2.5, 7)).get(0);
        }
        // --8<-- [end:qwen-image]
    }

    void edit(String repo) throws Exception {
        // --8<-- [start:edit]
        var config = DiffusionModel.Config.builder()
                .diffusionModel(HfCache.require("Comfy-Org/Qwen-Image-Edit_ComfyUI",
                        "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors").toString())
                .vae(HfCache.require(repo, "split_files/vae/qwen_image_vae.safetensors").toString())
                .llm(HfCache.require(repo, "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors").toString())
                .build();

        BufferedImage photo = ImageIO.read(new File("vase.png"));
        int w = photo.getWidth() / 16 * 16, h = photo.getHeight() / 16 * 16;   // a multiple of 16, near the original

        try (var editor = new DiffusionModel(config)) {
            BufferedImage edited = editor.generate(new DiffusionModel.Request(
                    "make the glaze deep cobalt blue", w, h, 20, 2.5, 7, List.of(photo), Map.of())).get(0);
        }
        // --8<-- [end:edit]
    }

    void img2img(DiffusionModel images) throws Exception {
        // --8<-- [start:img2img]
        BufferedImage shop = ImageIO.read(new File("shop.png"));
        var request = new DiffusionModel.Request("the same shop at night in heavy snow", 512, 512, 20, 4.0, 7)
                .withInitImage(shop, 0.6);
        BufferedImage snowy = images.generate(request).get(0);
        // --8<-- [end:img2img]
    }

    void inpaint(BufferedImage shop) throws Exception {
        // --8<-- [start:inpaint]
        BufferedImage mask = ImageIO.read(new File("mask.png"));     // white where the bicycle was, black elsewhere
        var request = new DiffusionModel.Request("a large terracotta pot", 512, 512, 20, 4.0, 7)
                .withInitImage(shop, 1.0)
                .withMaskImage(mask);
        // --8<-- [end:inpaint]
    }

    void controlNet(String sdxl, String vae, BufferedImage cannyEdges) {
        // --8<-- [start:controlnet]
        var config = DiffusionModel.Config.builder()
                .diffusionModel(sdxl).vae(vae)
                .controlNet("controlnet-canny-sdxl.safetensors")
                .build();

        var request = new DiffusionModel.Request("a glass teapot", 1024, 1024, 25, 6.0, 7)
                .withControlImage(cannyEdges, 0.8);     // strength defaults to 0.9
        // --8<-- [end:controlnet]
    }

    void lora() {
        // --8<-- [start:lora]
        var pixelArt = new DiffusionModel.Lora(
                HfCache.require("adirik/pixel-art-lora-flux.2-klein-4B", "pixelart_lora.safetensors"), 1.0f);

        var request = new DiffusionModel.Request("PXART4. A pixel art of a small pottery shop", 512, 512, 20, 4.0, 7)
                .withLoras(pixelArt);
        // --8<-- [end:lora]
    }

    void editLightning(int w, int h, BufferedImage photo) {
        // --8<-- [start:edit-lightning]
        var lightning = new DiffusionModel.Lora(HfCache.require("lightx2v/Qwen-Image-Edit-2511-Lightning",
                "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"), 1.0f);

        var request = new DiffusionModel.Request("make the sign say FORNAX", w, h, 4, 1.0, 1)
                .withRefImages(List.of(photo))
                .withLoras(lightning)
                .withExtra(Map.of("sample.flow_shift", 3.1, "sample.method", "euler", "sample.scheduler", "simple"));
        // --8<-- [end:edit-lightning]
    }

    void upscale(BufferedImage img) {
        // --8<-- [start:upscale]
        try (var up = new Upscaler(HfCache.require("Comfy-Org/Real-ESRGAN", "RealESRGAN_x4plus.safetensors"))) {
            BufferedImage big = up.upscale(img, 4);      // 512×512 -> 2048×2048
        }
        // --8<-- [end:upscale]
    }

    void zImage() {
        // --8<-- [start:z-image]
        String z = "Comfy-Org/z_image_turbo";
        var config = DiffusionModel.Config.builder()
                .diffusionModel(HfCache.require(z, "split_files/diffusion_models/z_image_turbo_bf16.safetensors").toString())
                .vae(HfCache.require(z, "split_files/vae/ae.safetensors").toString())
                .llm(HfCache.require(z, "split_files/text_encoders/qwen_3_4b.safetensors").toString())
                .build();

        try (var images = new DiffusionModel(config)) {
            BufferedImage img = images.generate(new DiffusionModel.Request(
                    "a rain-slicked street at night, neon reflections, a sign that says FORNAX", 1024, 1024, 8, 1.0, 7)).get(0);
        }
        // --8<-- [end:z-image]
    }

    void extra(DiffusionModel images) {
        // --8<-- [start:extra]
        var request = new DiffusionModel.Request("a neon sign", 1024, 1024, 20, 4.0, 7, List.of(), Map.of(
                "negative_prompt", "blurry, low quality",
                "sample.method", "euler",
                "batch_count", 2,                  // two images from one call
                "vae_tiling.enabled", true));      // decode in tiles, to use less memory
        List<BufferedImage> two = images.generate(request);
        // --8<-- [end:extra]
    }

    void configSet(String flux, String vae, String textEncoder) {
        // --8<-- [start:config-set]
        var config = DiffusionModel.Config.builder()
                .diffusionModel(flux).vae(vae).llm(textEncoder)
                .set("flash_attn", true)
                .build();
        // --8<-- [end:config-set]
    }

    void progress() {
        // --8<-- [start:progress]
        DiffusionModel.onProgress((step, steps, seconds) -> System.out.printf("step %d/%d (%.2f s/it)%n", step, steps, seconds));
        DiffusionModel.onLog((level, text) -> System.err.print(text));    // levels: 0 debug, 1 info, 2 warn, 3 error
        // --8<-- [end:progress]
    }
}
