package org.liberationdata.kiln.sd;

import java.awt.image.BufferedImage;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static java.lang.foreign.ValueLayout.*;

/**
 * An in-process image (and video) diffusion model backed by stable-diffusion.cpp: SD 1.x/2.x/XL/3, FLUX.1/2,
 * Qwen-Image, Wan, LTX and more. Weights load in the constructor and stay resident across calls.
 * Not thread-safe: use one instance per thread.
 */
public final class DiffusionModel implements AutoCloseable {

    /** Context-level options: model paths and runtime flags (keys from docs/guides/c_api.md). */
    public record Config(Map<String, Object> options) {
        public static Builder builder() { return new Builder(); }
        public static final class Builder {
            private final Map<String, Object> o = new LinkedHashMap<>();
            public Builder diffusionModel(String p) { o.put("diffusion_model_path", p); return this; }
            public Builder vae(String p) { o.put("vae_path", p); return this; }
            public Builder llm(String p) { o.put("llm_path", p); return this; }
            public Builder set(String key, Object value) { o.put(key, value); return this; }
            public Config build() { return new Config(Map.copyOf(o)); }
        }
    }

    /** Per-request options. */
    public record Request(String prompt, int width, int height, int steps, double cfg, long seed,
                          List<BufferedImage> refImages, Map<String, Object> extra) {
        public Request(String prompt, int width, int height, int steps, double cfg, long seed) {
            this(prompt, width, height, steps, cfg, seed, List.of(), Map.of());
        }
    }

    private final MemorySegment ctx;
    private final Arena arena = Arena.ofShared();

    public DiffusionModel(Config config) {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment opt = (MemorySegment) Sd.ctx_opt_create.invokeExact();
            try {
                applyAll(opt, config.options(), tmp);
                MemorySegment out = tmp.allocate(ADDRESS);
                Sd.check((int) Sd.ctx_create.invokeExact(opt, out), "sd_ctx_create");
                ctx = out.get(ADDRESS, 0);
            } finally {
                Sd.opt_free.invokeExact(opt);
            }
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    /** Sd log and progress callbacks (process-wide). */
    public static void onLog(Sd.LogSink s) { Sd.setLogSink(s); }
    public static void onProgress(Sd.ProgressSink s) { Sd.setProgressSink(s); }

    public String modelVersion() {
        try {
            return ((MemorySegment) Sd.get_model_version_name.invokeExact(ctx)).reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) { throw new RuntimeException(t); }
    }

    public List<BufferedImage> generate(Request r) {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment opt = (MemorySegment) Sd.img_gen_opt_create.invokeExact();
            MemorySegment[] refs = new MemorySegment[r.refImages().size()];
            try {
                Map<String, Object> o = new LinkedHashMap<>();
                o.put("prompt", r.prompt());
                o.put("width", r.width());
                o.put("height", r.height());
                o.put("sample.steps", r.steps());
                o.put("sample.guidance.txt_cfg", r.cfg());
                o.put("seed", r.seed());
                o.putAll(r.extra());
                applyAll(opt, o, tmp);
                if (refs.length > 0) {
                    for (int i = 0; i < refs.length; i++) refs[i] = toTensor(r.refImages().get(i), tmp);
                    MemorySegment arr = tmp.allocate(ADDRESS, refs.length);
                    for (int i = 0; i < refs.length; i++) arr.setAtIndex(ADDRESS, i, refs[i]);
                    Sd.check((int) Sd.opt_set_tensors.invokeExact(opt, tmp.allocateFrom("ref_images"), arr, (long) refs.length), "set ref_images");
                }
                MemorySegment out = tmp.allocate(ADDRESS);
                Sd.check((int) Sd.generate_image.invokeExact(ctx, opt, out), "sd_generate_image");
                MemorySegment result = out.get(ADDRESS, 0);
                try {
                    long n = (long) Sd.result_get_image_count.invokeExact(result);
                    var images = new java.util.ArrayList<BufferedImage>();
                    for (long i = 0; i < n; i++) images.add(fromTensor((MemorySegment) Sd.result_get_image.invokeExact(result, i)));
                    return images;
                } finally {
                    Sd.result_free.invokeExact(result);
                }
            } finally {
                for (MemorySegment t : refs) if (t != null) Sd.tensor_free.invokeExact(t);
                Sd.opt_free.invokeExact(opt);
            }
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    static void applyAll(MemorySegment opt, Map<String, Object> options, Arena a) throws Throwable {
        for (var e : options.entrySet()) {
            MemorySegment key = a.allocateFrom(e.getKey());
            Object v = e.getValue();
            int st = switch (v) {
                case Boolean b -> (int) Sd.opt_set_bool.invokeExact(opt, key, (boolean) b);
                case Integer i -> (int) Sd.opt_set_int.invokeExact(opt, key, (long) i);
                case Long l -> (int) Sd.opt_set_int.invokeExact(opt, key, (long) l);
                case Number n -> (int) Sd.opt_set_float.invokeExact(opt, key, n.doubleValue());
                case String s -> (int) Sd.opt_set_string.invokeExact(opt, key, a.allocateFrom(s));
                default -> throw new IllegalArgumentException("unsupported option type for " + e.getKey());
            };
            Sd.check(st, "set option '" + e.getKey() + "'");
        }
    }

    /** sd tensors are U8 [C,W,H], contiguous (channel) dimension first — i.e. interleaved RGB rows. */
    static BufferedImage fromTensor(MemorySegment t) throws Throwable {
        long c = (long) Sd.tensor_dim.invokeExact(t, 0L), w = (long) Sd.tensor_dim.invokeExact(t, 1L), h = (long) Sd.tensor_dim.invokeExact(t, 2L);
        long size = (long) Sd.tensor_byte_size.invokeExact(t);
        MemorySegment data = ((MemorySegment) Sd.tensor_data.invokeExact(t)).reinterpret(size);
        byte[] px = data.toArray(JAVA_BYTE);
        var img = new BufferedImage((int) w, (int) h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int i = (int) ((y * w + x) * c);
                int r = px[i] & 0xff, g = px[i + (c > 1 ? 1 : 0)] & 0xff, b = px[i + (c > 2 ? 2 : 0)] & 0xff;
                img.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        return img;
    }

    static MemorySegment toTensor(BufferedImage img, Arena a) throws Throwable {
        int w = img.getWidth(), h = img.getHeight();
        byte[] px = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y), i = (y * w + x) * 3;
                px[i] = (byte) (rgb >> 16); px[i + 1] = (byte) (rgb >> 8); px[i + 2] = (byte) rgb;
            }
        MemorySegment shape = a.allocateFrom(JAVA_LONG, 3, w, h);
        MemorySegment data = a.allocateFrom(JAVA_BYTE, px);
        MemorySegment t = (MemorySegment) Sd.tensor_create.invokeExact(Sd.SD_TENSOR_U8, 3L, shape, data, (long) px.length);
        if (t.equals(MemorySegment.NULL)) throw new Sd.SdException("sd_tensor_create: " + Sd.lastError());
        return t;
    }

    @Override public void close() {
        try { Sd.ctx_free.invokeExact(ctx); } catch (Throwable t) { throw new RuntimeException(t); }
        arena.close();
    }
}
