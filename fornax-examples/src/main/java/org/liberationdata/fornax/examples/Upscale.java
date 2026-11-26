package org.liberationdata.fornax.examples;

import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.sd.Upscaler;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;

/**
 * Upscale an image with an ESRGAN model through stable-diffusion.cpp.
 * Usage: Upscale <in.png> <out.png> [-Dfactor=4] [-Dtile=128] [-Dmodel=path/to/model.safetensors]
 * The model defaults to RealESRGAN_x4plus (BSD-3-Clause) from the Hugging Face cache:
 *   hf download Comfy-Org/Real-ESRGAN RealESRGAN_x4plus.safetensors
 */
public class Upscale {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: Upscale <in.png> <out.png> [-Pfactor=4] [-Ptile=128] [-Pmodel=esrgan.safetensors]");
            System.exit(2);
        }
        BufferedImage in = ImageIO.read(new File(args[0]));
        String model = System.getProperty("model");
        Path weights = model != null ? Path.of(model) : HfCache.require("Comfy-Org/Real-ESRGAN", "RealESRGAN_x4plus.safetensors");
        long t0 = System.nanoTime();
        try (var up = new Upscaler(weights, Integer.getInteger("tile", Upscaler.DEFAULT_TILE_SIZE))) {
            System.out.printf("[fornax] loaded a %dx upscaler in %.1fs%n", up.scale(), (System.nanoTime() - t0) / 1e9);
            int factor = Integer.getInteger("factor", up.scale());
            long u0 = System.nanoTime();
            BufferedImage big = up.upscale(in, factor);
            ImageIO.write(big, "png", new File(args[1]));
            System.out.printf("[fornax] %dx%d -> %dx%d in %.1fs -> %s%n", in.getWidth(), in.getHeight(),
                    big.getWidth(), big.getHeight(), (System.nanoTime() - u0) / 1e9, args[1]);
        }
    }
}
