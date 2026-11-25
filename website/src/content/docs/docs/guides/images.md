---
title: Images
description: Generate and edit images from Java with stable-diffusion.cpp - FLUX.2 [klein], Qwen-Image, Qwen-Image-Edit and Z-Image - and pass any engine option through.
---

`kiln-sd` runs image (and [video](/docs/guides/video/)) diffusion models through
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp): FLUX.1 and FLUX.2, Qwen-Image and
Qwen-Image-Edit, Z-Image, SD 1.x, 2.x, XL and 3, and more. The class is `DiffusionModel`.

A diffusion model is usually several files: the **diffusion model** itself, a **VAE** that turns latents into pixels,
and a **text encoder** (a language model, for the models on this page). You name each one in a `Config`, and the
model loads all of them in its constructor and keeps them loaded across calls.

## FLUX.2 [klein]

FLUX.2 [klein] base 4B is the model Kiln's demo paints its covers with, and a good first one to try.

```sh
hf download black-forest-labs/FLUX.2-klein-base-4b-fp8 flux-2-klein-base-4b-fp8.safetensors
hf download Comfy-Org/flux2-dev split_files/vae/flux2-vae.safetensors
hf download Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b.safetensors
```

```java
import org.liberationdata.kiln.HfCache;
import org.liberationdata.kiln.sd.DiffusionModel;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

var config = DiffusionModel.Config.builder()
        .diffusionModel(HfCache.require("black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors").toString())
        .vae(HfCache.require("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors").toString())
        .llm(HfCache.require("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors").toString())
        .build();

try (var images = new DiffusionModel(config)) {
    //                                              prompt, width, height, steps, cfg, seed
    var request = new DiffusionModel.Request("a neon sign that says KILN", 768, 768, 12, 4.0, 7);
    BufferedImage img = images.generate(request).get(0);
    ImageIO.write(img, "png", new File("kiln.png"));
}
```

`generate` returns a `List<BufferedImage>`, one image unless you ask for more (below). Kiln's examples use 20 steps
at CFG 4.0 for klein; at 768×768 and 12 steps it took about 130 s on the Radeon 8060S with Vulkan.

The text encoder here is Qwen3 4B, from Z-Image Turbo's repository: FLUX.2 [klein] 4B and Z-Image use the same one.

## Qwen-Image

Qwen-Image 2512 is a much larger model, and slower. Its text encoder is Qwen2.5-VL 7B.

```sh
hf download Comfy-Org/Qwen-Image_ComfyUI split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors \
    split_files/vae/qwen_image_vae.safetensors split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
```

```java
String repo = "Comfy-Org/Qwen-Image_ComfyUI";
var config = DiffusionModel.Config.builder()
        .diffusionModel(HfCache.require(repo, "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors").toString())
        .vae(HfCache.require(repo, "split_files/vae/qwen_image_vae.safetensors").toString())
        .llm(HfCache.require(repo, "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors").toString())
        .build();

try (var images = new DiffusionModel(config)) {
    BufferedImage img = images.generate(new DiffusionModel.Request(
            "a hand-painted shop sign reading 'Kiln & Co. Pottery'", 1024, 1024, 20, 2.5, 7)).get(0);
}
```

Kiln's examples use 20 steps at CFG 2.5. At 1024×1024 that took about 13 minutes on the 8060S with Vulkan.

### Editing an image: Qwen-Image-Edit

Qwen-Image-Edit 2511 takes a reference image and an instruction. It uses Qwen-Image's VAE and text encoder:

```sh
hf download Comfy-Org/Qwen-Image-Edit_ComfyUI split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors
```

```java
import java.util.List;
import java.util.Map;

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
```

The full `Request` constructor takes the reference images and a map of extra options; the short one leaves both
empty.

## Z-Image

Z-Image Turbo is a distilled model: it needs only a few steps, at CFG 1. It shares Qwen3 4B with FLUX.2 [klein] and
uses the FLUX.1 VAE.

```sh
hf download Comfy-Org/z_image_turbo split_files/diffusion_models/z_image_turbo_bf16.safetensors \
    split_files/vae/ae.safetensors split_files/text_encoders/qwen_3_4b.safetensors
```

```java
String z = "Comfy-Org/z_image_turbo";
var config = DiffusionModel.Config.builder()
        .diffusionModel(HfCache.require(z, "split_files/diffusion_models/z_image_turbo_bf16.safetensors").toString())
        .vae(HfCache.require(z, "split_files/vae/ae.safetensors").toString())
        .llm(HfCache.require(z, "split_files/text_encoders/qwen_3_4b.safetensors").toString())
        .build();

try (var images = new DiffusionModel(config)) {
    BufferedImage img = images.generate(new DiffusionModel.Request(
            "a rain-slicked street at night, neon reflections, a sign that says KILN", 1024, 1024, 8, 1.0, 7)).get(0);
}
```

:::caution[Not yet in Kiln's examples]
Kiln's `image` example has no Z-Image preset yet. The settings above (8 steps, CFG 1.0, the FLUX.1 VAE) are
stable-diffusion.cpp's own for Z-Image Turbo. If you try it, a report of how it went is welcome.
:::

## Options

`Request` sets what most models need: the prompt, the size, the number of steps, the CFG scale and the seed. Anything
else that stable-diffusion.cpp accepts goes in the request's `extra` map, under the engine's own key names (from its
[C API guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/guides/c_api.md)). Values can be
`Boolean`, `Integer` or `Long` (set as an integer), any other `Number` (set as a float), `String` or `float[]`.
The engine doesn't convert between them, so a float option needs a floating-point value: `"strength", 1.0`, not
`"strength", 1`.

```java
var request = new DiffusionModel.Request("a neon sign", 1024, 1024, 20, 4.0, 7, List.of(), Map.of(
        "negative_prompt", "blurry, low quality",
        "sample.method", "euler",
        "batch_count", 2,                  // two images from one call
        "vae_tiling.enabled", true));      // decode in tiles, to use less memory
List<BufferedImage> two = images.generate(request);
```

Options for the model as a whole, such as the paths of extra files or how weights are stored, go in the `Config`,
with `set(key, value)`:

```java
var config = DiffusionModel.Config.builder()
        .diffusionModel(flux).vae(vae).llm(textEncoder)
        .set("flash_attn", true)
        .build();
```

An unknown key, or a value of the wrong type, fails with an `Sd.SdException` that names the key.

## Progress and logs

The engine reports each sampling step. `onProgress` and `onLog` are process-wide.

```java
DiffusionModel.onProgress((step, steps, seconds) -> System.out.printf("step %d/%d (%.2f s/it)%n", step, steps, seconds));
DiffusionModel.onLog((level, text) -> System.err.print(text));    // levels: 0 debug, 1 info, 2 warn, 3 error
```

`modelVersion()` names the model stable-diffusion.cpp detected, which is a quick check that the files are the ones
you meant.

The engine is quiet by default, apart from errors. `KILN_VERBOSE=1` shows its info messages, and
`KILN_VERBOSE=debug` everything, including the program it compiles for each generation.

## Try it

```sh
./gradlew image --args='klein out.png "a neon sign that says KILN"' -Psteps=12 -Pwidth=768 -Pheight=768
./gradlew image --args='qwen-image out.png "a hand-painted shop sign"'
./gradlew image --args='qwen-edit out.png "make the glaze cobalt blue" vase.png'
./gradlew image --args='klein out.png "a kiln at night"' -Prepeat=2 -Pseed=3     # the second image reuses the loaded weights
```
