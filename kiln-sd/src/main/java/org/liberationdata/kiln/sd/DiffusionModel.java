package org.liberationdata.kiln.sd;

import org.liberationdata.kiln.Audio;

import java.awt.image.BufferedImage;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
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

    /**
     * A video request. initImage / endImage (optional) pin the first / last frame (LTX, Wan); refAudio (optional)
     * drives speech-to-video models (Wan2.2 S2V, MiniMax-H3). extra takes any other sd.cpp video option key.
     */
    public record VideoRequest(String prompt, int width, int height, int frames, int fps, int steps, double cfg, long seed,
                               BufferedImage initImage, BufferedImage endImage, Audio refAudio, Map<String, Object> extra) {
        public VideoRequest(String prompt, int width, int height, int frames, int fps, int steps, double cfg, long seed) {
            this(prompt, width, height, frames, fps, steps, cfg, seed, null, null, null, Map.of());
        }

        public VideoRequest withInitImage(BufferedImage i) { return new VideoRequest(prompt, width, height, frames, fps, steps, cfg, seed, i, endImage, refAudio, extra); }
        public VideoRequest withEndImage(BufferedImage i) { return new VideoRequest(prompt, width, height, frames, fps, steps, cfg, seed, initImage, i, refAudio, extra); }
        public VideoRequest withRefAudio(Audio a) { return new VideoRequest(prompt, width, height, frames, fps, steps, cfg, seed, initImage, endImage, a, extra); }
        public VideoRequest withExtra(Map<String, Object> e) { return new VideoRequest(prompt, width, height, frames, fps, steps, cfg, seed, initImage, endImage, refAudio, e); }
    }

    /** Generated video: its frames, frame rate, and the soundtrack when the model makes one (LTX with an audio VAE). */
    public record Video(List<BufferedImage> frames, int fps, Audio audio) {
        public double seconds() { return fps == 0 ? 0 : frames.size() / (double) fps; }
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
                checkHires(o);
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

    public Video generateVideo(VideoRequest r) {
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment opt = (MemorySegment) Sd.vid_gen_opt_create.invokeExact();
            List<MemorySegment> owned = new java.util.ArrayList<>();
            try {
                Map<String, Object> o = new LinkedHashMap<>();
                o.put("prompt", r.prompt());
                o.put("width", r.width());
                o.put("height", r.height());
                o.put("video_frames", r.frames());
                o.put("fps", r.fps());
                o.put("sample.steps", r.steps());
                o.put("sample.guidance.txt_cfg", r.cfg());
                o.put("seed", r.seed());
                o.putAll(r.extra());
                checkHires(o);
                applyAll(opt, o, tmp);
                if (r.initImage() != null) setTensor(opt, "init_image", owned, toTensor(r.initImage(), tmp), tmp);
                if (r.endImage() != null) setTensor(opt, "end_image", owned, toTensor(r.endImage(), tmp), tmp);
                if (r.refAudio() != null) {
                    MemorySegment t = audioTensor(r.refAudio(), tmp);
                    owned.add(t);
                    MemorySegment a = tmp.allocate(Sd.AUDIO);
                    a.set(JAVA_INT, 0, r.refAudio().sampleRate());
                    a.set(ADDRESS, Sd.AUDIO.byteOffset(MemoryLayout.PathElement.groupElement("data")), t);
                    Sd.check((int) Sd.opt_set_audios.invokeExact(opt, tmp.allocateFrom("ref_audios"), a, 1L), "set ref_audios");
                }
                MemorySegment out = tmp.allocate(ADDRESS);
                Sd.check((int) Sd.generate_video.invokeExact(ctx, opt, out), "sd_generate_video");
                MemorySegment result = out.get(ADDRESS, 0);
                try {
                    long n = (long) Sd.result_get_image_count.invokeExact(result);
                    var frames = new java.util.ArrayList<BufferedImage>((int) n);
                    for (long i = 0; i < n; i++) frames.add(fromTensor((MemorySegment) Sd.result_get_image.invokeExact(result, i)));
                    int fps = (int) Sd.result_get_fps.invokeExact(result);
                    return new Video(frames, fps, fromAudio((MemorySegment) Sd.result_get_audio.invokeExact(result)));
                } finally {
                    Sd.result_free.invokeExact(result);
                }
            } finally {
                for (MemorySegment t : owned) Sd.tensor_free.invokeExact(t);
                Sd.opt_free.invokeExact(opt);
            }
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    private static void setTensor(MemorySegment opt, String key, List<MemorySegment> owned, MemorySegment t, Arena a) throws Throwable {
        owned.add(t);
        Sd.check((int) Sd.opt_set_tensor.invokeExact(opt, a.allocateFrom(key), t), "set " + key);
    }

    /** sd audio tensors are F32 [C, N], channel first: interleaved samples. */
    static MemorySegment audioTensor(Audio audio, Arena a) throws Throwable {
        int n = audio.left().length;
        float[] interleaved = new float[n * 2];
        for (int i = 0; i < n; i++) { interleaved[2 * i] = audio.left()[i]; interleaved[2 * i + 1] = audio.right()[i]; }
        MemorySegment shape = a.allocateFrom(JAVA_LONG, 2, n);
        MemorySegment data = a.allocateFrom(JAVA_FLOAT, interleaved);
        MemorySegment t = (MemorySegment) Sd.tensor_create.invokeExact(Sd.SD_TENSOR_F32, 2L, shape, data, (long) interleaved.length * 4);
        if (t.equals(MemorySegment.NULL)) throw new Sd.SdException("sd_tensor_create (audio): " + Sd.lastError());
        return t;
    }

    static Audio fromAudio(MemorySegment audio) throws Throwable {
        if (audio.equals(MemorySegment.NULL)) return null;
        audio = audio.reinterpret(Sd.AUDIO.byteSize());
        int rate = audio.get(JAVA_INT, 0);
        MemorySegment t = audio.get(ADDRESS, Sd.AUDIO.byteOffset(MemoryLayout.PathElement.groupElement("data")));
        if (t.equals(MemorySegment.NULL) || rate == 0) return null;
        long channels = (long) Sd.tensor_dim.invokeExact(t, 0L), n = (long) Sd.tensor_dim.invokeExact(t, 1L);
        long size = (long) Sd.tensor_byte_size.invokeExact(t);
        float[] s = ((MemorySegment) Sd.tensor_data.invokeExact(t)).reinterpret(size).toArray(JAVA_FLOAT);
        float[] l = new float[(int) n], rr = new float[(int) n];
        for (int i = 0; i < n; i++) {
            l[i] = s[(int) (i * channels)];
            rr[i] = channels > 1 ? s[(int) (i * channels + 1)] : l[i];
        }
        return new Audio(l, rr, rate);
    }

    /**
     * stable-diffusion.cpp only runs a learned hires upscaler when {@code hires.upscaler} is {@code "model"}. With any
     * other value, or none, it interpolates the latent and silently ignores {@code hires.model_path}, and refining an
     * interpolated video latent ghosts. So a model path with no upscaler selects the model, and a model path with a
     * different upscaler is refused.
     */
    static void checkHires(Map<String, Object> o) {
        Object path = o.get("hires.model_path");
        if (path == null || path.toString().isEmpty()) return;
        Object upscaler = o.get("hires.upscaler");
        if (upscaler == null) o.put("hires.upscaler", "model");
        else if (!"model".equals(upscaler))
            throw new IllegalArgumentException("hires.model_path is set but hires.upscaler is '" + upscaler
                    + "', which ignores the model; use \"model\" or drop hires.model_path");
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
                case float[] f -> (int) Sd.opt_set_floats.invokeExact(opt, key, a.allocateFrom(JAVA_FLOAT, f), (long) f.length);
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
