package org.liberationdata.kiln.sd;

import java.awt.image.BufferedImage;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Map;

import static java.lang.foreign.ValueLayout.ADDRESS;

/**
 * An in-process image upscaler backed by stable-diffusion.cpp: ESRGAN-family models such as RealESRGAN_x4plus, and
 * compatible MoSR models. The model loads in the constructor and stays resident across calls. Its scale (2x, 4x) is
 * read from the weights. Not thread-safe: use one instance per thread.
 *
 * <pre>{@code
 * try (var up = new Upscaler(esrgan)) {
 *     BufferedImage big = up.upscale(img, 4);
 * }
 * }</pre>
 */
public final class Upscaler implements AutoCloseable {

    /** stable-diffusion.cpp's default tile edge, in input pixels. */
    public static final int DEFAULT_TILE_SIZE = 128;

    private final MemorySegment upscaler;
    private final int scale;

    public Upscaler(Path model) { this(model, DEFAULT_TILE_SIZE); }

    /**
     * @param tileSize the model runs over tiles of this many input pixels a side; smaller tiles use less GPU memory
     *                 and keep each GPU dispatch short
     */
    public Upscaler(Path model, int tileSize) {
        if (tileSize < 1) throw new IllegalArgumentException("tileSize must be at least 1, was " + tileSize);
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment opt = (MemorySegment) Sd.upscaler_opt_create.invokeExact();
            try {
                DiffusionModel.applyAll(opt, Map.of("model_path", model.toAbsolutePath().toString(), "tile_size", tileSize), tmp);
                MemorySegment out = tmp.allocate(ADDRESS);
                Sd.check((int) Sd.upscaler_create.invokeExact(opt, out), "sd_upscaler_create");
                upscaler = out.get(ADDRESS, 0);
            } finally {
                Sd.opt_free.invokeExact(opt);
            }
            scale = (int) Sd.upscaler_get_scale.invokeExact(upscaler);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    /** How much one pass enlarges an image: 4 for RealESRGAN_x4plus. */
    public int scale() { return scale; }

    /** One pass of the model: {@link #scale()} times larger. */
    public BufferedImage upscale(BufferedImage image) { return upscale(image, scale); }

    /**
     * Enlarges {@code image} by {@code factor}, which must be a power of the model's scale (4 or 16 for a 4x model):
     * the model runs once per power. The engine refuses results over 16384 pixels a side or 64 megapixels.
     */
    public BufferedImage upscale(BufferedImage image, int factor) {
        int repeats = repeats(scale, factor);
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment opt = (MemorySegment) Sd.upscale_opt_create.invokeExact();
            MemorySegment tensor = DiffusionModel.toTensor(image, tmp);
            try {
                DiffusionModel.applyAll(opt, Map.of("repeats", repeats), tmp);
                Sd.check((int) Sd.opt_set_tensor.invokeExact(opt, tmp.allocateFrom("image"), tensor), "set image");
                MemorySegment out = tmp.allocate(ADDRESS);
                Sd.check((int) Sd.upscale.invokeExact(upscaler, opt, out), "sd_upscale");
                MemorySegment result = out.get(ADDRESS, 0);
                try {
                    return DiffusionModel.fromTensor((MemorySegment) Sd.result_get_image.invokeExact(result, 0L));
                } finally {
                    Sd.result_free.invokeExact(result);
                }
            } finally {
                Sd.tensor_free.invokeExact(tensor);
                Sd.opt_free.invokeExact(opt);
            }
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    /** How many passes of a {@code scale}x model give {@code factor}x; anything else is refused, not approximated. */
    static int repeats(int scale, int factor) {
        if (scale < 2) throw new IllegalStateException("upscaler reports scale " + scale);
        int repeats = 0;
        for (long f = 1; f < factor; f *= scale) repeats++;
        long reached = 1;
        for (int i = 0; i < repeats; i++) reached *= scale;
        if (factor < scale || reached != factor)
            throw new IllegalArgumentException("a " + scale + "x model upscales by powers of " + scale + " (" + scale + ", "
                    + scale * scale + ", ...), not " + factor + "; resize the result for other sizes");
        return repeats;
    }

    @Override public void close() {
        try { Sd.upscaler_free.invokeExact(upscaler); } catch (Throwable t) { throw new RuntimeException(t); }
    }
}
