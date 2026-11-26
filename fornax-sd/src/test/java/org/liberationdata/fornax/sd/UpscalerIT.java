package org.liberationdata.fornax.sd;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.HfCache;

import java.awt.image.BufferedImage;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Upscaler on its model thread, with RealESRGAN_x4plus on a 16x16 image when the weights are in the Hugging Face
 * cache; skipped otherwise.
 */
class UpscalerIT {

    @Test
    void upscalesOnItsThreadAndClosesOnce() {
        var esrgan = HfCache.find("Comfy-Org/Real-ESRGAN", "RealESRGAN_x4plus.safetensors");
        assumeTrue(esrgan.isPresent(), "RealESRGAN_x4plus not downloaded");
        var img = new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB);
        var up = new Upscaler(esrgan.get(), 16);
        try {
            assertEquals(4, up.scale());
            CompletableFuture<BufferedImage> first = up.upscaleAsync(img);
            CompletableFuture<BufferedImage> queued = up.upscaleAsync(img);
            queued.cancel(true);                                   // still queued behind first: never starts
            assertEquals(64, first.join().getWidth());
            assertThrows(CancellationException.class, queued::join);
            assertEquals(64, up.upscale(img).getHeight());         // blocking, on this thread
            assertThrows(IllegalArgumentException.class, () -> up.upscaleAsync(img, 8));
        } finally {
            up.close();
            up.close();  // idempotent
        }
        assertThrows(IllegalStateException.class, () -> up.upscale(img));
        var after = assertThrows(CompletionException.class, () -> up.upscaleAsync(img).join());
        assertInstanceOf(IllegalStateException.class, after.getCause());
    }
}
