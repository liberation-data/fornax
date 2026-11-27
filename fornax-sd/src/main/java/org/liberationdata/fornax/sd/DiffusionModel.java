package org.liberationdata.fornax.sd;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.Call;
import org.liberationdata.fornax.ModelThread;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelReferences;
import org.liberationdata.fornax.hub.ModelSet;

import java.awt.image.BufferedImage;
import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;

import static java.lang.foreign.ValueLayout.*;

/**
 * An in-process image (and video) diffusion model backed by stable-diffusion.cpp: SD 1.x/2.x/XL/3, FLUX.1/2,
 * Qwen-Image, Wan, LTX and more. Weights load in the constructor and stay resident across calls.
 *
 * <p>Threads: calls run one at a time, in arrival order, from any thread. {@link #generate} and
 * {@link #generateVideo} run on the calling thread; {@link #generateAsync} and {@link #generateVideoAsync} run on
 * this model's own platform thread and return a {@link CompletableFuture}, so a virtual thread awaiting one doesn't pin
 * its carrier. {@link #cancel()} stops a run between steps (it then throws a {@link CancellationException}), and
 * {@link #close()} is safe from any thread. The log and progress sinks are process-wide, and run on the generating
 * thread.
 */
public final class DiffusionModel implements AutoCloseable {

    /**
     * Context-level options: model paths and runtime flags (keys from docs/guides/c_api.md). A path option (a key
     * ending in {@code _path}) may be a model reference instead ({@code city96/FLUX.1-dev-gguf:Q4_K_S}), downloaded when
     * the model loads; {@link Builder#model} sets every path from a catalogue set.
     */
    public record Config(Map<String, Object> options) {
        /**
         * {@code tensor_type_rules} that widened the bf16 tensors of ComfyUI's int8 LTX checkpoints (about 0.9 GB) to f32
         * at load, for Vulkan drivers without bf16. Not needed since Fornax's stable-diffusion.cpp includes
         * leejet/stable-diffusion.cpp#2432 (0.3.0), and no longer set by {@link Builder#model}.
         * @deprecated not needed; it only costs memory
         */
        @Deprecated(since = "0.3.0", forRemoval = true)
        public static final String LTX_BF16_TO_F32 = ".*\\.bias=f32,.*_norm\\.weight=f32,.*to_gate_logits\\.weight=f32,"
                + ".*adaln_single.*\\.weight=f32,.*(patchify_proj|proj_out)\\.weight=f32,"
                + ".*(learnable_registers|keyframes_abs_pos_embedding)=f32";

        static final ModelReferences.Engine SET = new ModelReferences.Engine("DiffusionModel", "image and video models",
                Set.of(ModelSet.Kind.IMAGE, ModelSet.Kind.VIDEO), Set.of(), "an image or video set");
        static final ModelReferences.Engine FILE = new ModelReferences.Engine("DiffusionModel",
                "one file per path option (Config.builder().model(name) takes a whole set)", Set.of(), Set.of("gguf", "model"), "a model file");

        /** Each sd.cpp context option a catalogue role sets. The latent upscaler isn't one: it is a request's {@code hires.model_path}. */
        static final Map<String, String> ROLE_OPTIONS = Map.of("diffusion_model", "diffusion_model_path", "vae", "vae_path",
                "llm", "llm_path", "audio_vae", "audio_vae_path");

        public static Builder builder() { return new Builder(); }

        /** The options with every path option that is a model reference replaced by its file, downloaded if missing. */
        Map<String, Object> resolved() { return resolved(null); }

        /** {@link #resolved()} with the given hub, or null for the standard one (built only for a reference). */
        Map<String, Object> resolved(FornaxHub hub) {
            List<String> paths = options.entrySet().stream().filter(Config::isPathOption).map(e -> (String) e.getValue()).toList();
            for (String p : paths) {   // every one refused before any download
                if (hub == null) ModelReferences.check(p, FILE);
                else ModelReferences.check(p, FILE, hub);
            }
            Map<String, Object> out = new LinkedHashMap<>(options);
            for (var e : options.entrySet()) {
                if (!isPathOption(e)) continue;
                String v = (String) e.getValue();
                out.put(e.getKey(), hub == null ? ModelReferences.path(v, FILE) : ModelReferences.path(v, FILE, hub));
            }
            return out;
        }

        private static boolean isPathOption(Map.Entry<String, Object> e) {
            return e.getKey().endsWith("_path") && e.getValue() instanceof String v && !v.isBlank();
        }

        public static final class Builder {
            private final Map<String, Object> o = new LinkedHashMap<>();

            /**
             * Every model path of the catalogue set with this name ({@code "flux2-klein"}, {@code "qwen-image"},
             * {@code "ltx-2.5-distilled"}), downloading what's missing now. Options set after this override it.
             *
             * @throws IllegalArgumentException for a name that isn't an image or video set, before any download
             */
            public Builder model(String name) { return model(name, null); }

            /** {@link #model(String)} with the given hub, or null for the standard one. */
            Builder model(String name, FornaxHub hub) {
                Map<String, Path> files = hub == null ? ModelReferences.set(name, SET, "stable-diffusion.cpp")
                        : ModelReferences.set(name, SET, "stable-diffusion.cpp", hub);
                files.forEach((role, file) -> { if (ROLE_OPTIONS.containsKey(role)) o.put(ROLE_OPTIONS.get(role), file.toString()); });
                return this;
            }

            public Builder diffusionModel(String p) { o.put("diffusion_model_path", p); return this; }
            public Builder vae(String p) { o.put("vae_path", p); return this; }
            public Builder llm(String p) { o.put("llm_path", p); return this; }
            /**
             * A ControlNet model, used by requests with a {@link Request#controlImage()}. FLUX.1 Controls and Flex.2
             * take the control image natively and need none.
             */
            public Builder controlNet(String p) { o.put("control_net_path", p); return this; }
            public Builder set(String key, Object value) { o.put(key, value); return this; }
            public Config build() { return new Config(Map.copyOf(o)); }
        }
    }

    /**
     * A LoRA adapter applied to one request, at {@code weight} (1 is full strength; negative weights subtract it).
     *
     * <p>stable-diffusion.cpp reads LoRAs in the common formats (kohya / sd-scripts, diffusers / PEFT, ai-toolkit) and
     * maps their tensor names onto the loaded model's family, so a LoRA works with the family it was trained for: SD
     * 1.x/2.x/XL/3, FLUX.1, FLUX.2 (klein included), Qwen-Image and Qwen-Image-Edit, Z-Image, Wan and others. Fornax
     * has rendered with one on FLUX.2 [klein] base 4B, and compiled the Qwen-Image-Edit Lightning recipe. A LoRA
     * for another family fails or has no effect, so check what it was trained on.
     *
     * <p>The engine applies a LoRA to the diffusion model, the text encoder or both, whichever it has tensors for.
     * How it is applied (merged into the weights, or at run time) is the context option {@code lora_apply_mode}:
     * {@code auto} (the default), {@code immediately} or {@code at_runtime}.
     */
    public record Lora(Path file, float weight) {
        public Lora {
            Objects.requireNonNull(file, "file");
            if (!Float.isFinite(weight)) throw new IllegalArgumentException("LoRA weight must be finite, was " + weight);
        }

        public Lora(Path file) { this(file, 1f); }
    }

    /**
     * Per-request options.
     *
     * <ul>
     *   <li>{@code refImages}: reference images, for models that edit or compose from them (FLUX.2, Qwen-Image-Edit,
     *       FLUX.1 Kontext).</li>
     *   <li>{@code initImage} and {@code strength} (img2img): start from this image instead of noise. {@code strength}
     *       is how much of the schedule is re-run, from 0 (keep the image) to 1 (ignore it); {@code null} means the
     *       engine's 0.75. The image is fitted to {@code width} x {@code height}.</li>
     *   <li>{@code maskImage} (inpainting): only the white parts of the init image are repainted. It needs an init
     *       image of the same size. Any model can inpaint this way (the mask limits the denoising); dedicated inpaint
     *       models (SD inpaint, FLUX.1 Fill) also get the masked image as conditioning.</li>
     *   <li>{@code controlImage} and {@code controlStrength}: a ControlNet hint (edges, depth, pose), with
     *       {@link Config.Builder#controlNet(String)}, fitted to {@code width} x {@code height}. {@code null} strength
     *       means the engine's 0.9. Models that take the hint natively (FLUX.1 Controls, Flex.2) ignore the
     *       strength.</li>
     *   <li>{@code loras}: LoRA adapters for this request.</li>
     *   <li>{@code extra}: any other stable-diffusion.cpp image option, under its own key.</li>
     * </ul>
     *
     * Settings the engine would silently ignore are refused when the request is generated: a strength or mask
     * without an init image, a control strength without a control image, or an option with its own field set in
     * {@code extra} instead.
     */
    public record Request(String prompt, int width, int height, int steps, double cfg, long seed,
                          List<BufferedImage> refImages, Map<String, Object> extra,
                          BufferedImage initImage, Double strength, BufferedImage maskImage,
                          BufferedImage controlImage, Double controlStrength, List<Lora> loras) {
        public Request {
            refImages = refImages == null ? List.of() : List.copyOf(refImages);
            extra = extra == null ? Map.of() : Map.copyOf(extra);
            loras = loras == null ? List.of() : List.copyOf(loras);
            if (strength != null && !(strength >= 0 && strength <= 1))
                throw new IllegalArgumentException("strength must be between 0 and 1, was " + strength);
            if (controlStrength != null && !(controlStrength >= 0 && Double.isFinite(controlStrength)))
                throw new IllegalArgumentException("controlStrength must be finite and at least 0, was " + controlStrength);
        }

        public Request(String prompt, int width, int height, int steps, double cfg, long seed,
                       List<BufferedImage> refImages, Map<String, Object> extra) {
            this(prompt, width, height, steps, cfg, seed, refImages, extra, null, null, null, null, null, List.of());
        }

        public Request(String prompt, int width, int height, int steps, double cfg, long seed) {
            this(prompt, width, height, steps, cfg, seed, List.of(), Map.of());
        }

        public Request withRefImages(List<BufferedImage> r) { return new Request(prompt, width, height, steps, cfg, seed, r, extra, initImage, strength, maskImage, controlImage, controlStrength, loras); }
        public Request withExtra(Map<String, Object> e) { return new Request(prompt, width, height, steps, cfg, seed, refImages, e, initImage, strength, maskImage, controlImage, controlStrength, loras); }
        /** img2img from {@code image}, at the engine's default strength (0.75). */
        public Request withInitImage(BufferedImage image) { return new Request(prompt, width, height, steps, cfg, seed, refImages, extra, image, strength, maskImage, controlImage, controlStrength, loras); }
        /** img2img from {@code image}: {@code strength} 0 keeps it, 1 ignores it. */
        public Request withInitImage(BufferedImage image, double strength) { return new Request(prompt, width, height, steps, cfg, seed, refImages, extra, image, strength, maskImage, controlImage, controlStrength, loras); }
        /** Inpainting: repaint only the white parts of the init image. */
        public Request withMaskImage(BufferedImage mask) { return new Request(prompt, width, height, steps, cfg, seed, refImages, extra, initImage, strength, mask, controlImage, controlStrength, loras); }
        /** A ControlNet hint, at the engine's default strength (0.9). */
        public Request withControlImage(BufferedImage image) { return new Request(prompt, width, height, steps, cfg, seed, refImages, extra, initImage, strength, maskImage, image, controlStrength, loras); }
        public Request withControlImage(BufferedImage image, double controlStrength) { return new Request(prompt, width, height, steps, cfg, seed, refImages, extra, initImage, strength, maskImage, image, controlStrength, loras); }
        public Request withLoras(List<Lora> l) { return new Request(prompt, width, height, steps, cfg, seed, refImages, extra, initImage, strength, maskImage, controlImage, controlStrength, l); }
        public Request withLoras(Lora... l) { return withLoras(List.of(l)); }

        /** Engine keys that have their own Request field: setting them in extra as well is refused. */
        static final List<String> TYPED_KEYS = List.of("strength", "control_strength", "init_image", "mask_image",
                "control_image", "ref_images", "loras");

        /**
         * The scalar engine options for this request (images and LoRAs are set separately), after refusing settings
         * the engine would silently ignore.
         */
        Map<String, Object> options() {
            for (String k : TYPED_KEYS)
                if (extra.containsKey(k))
                    throw new IllegalArgumentException("'" + k + "' has its own Request field; set it there, not in extra");
            if (initImage == null) {
                if (strength != null)
                    throw new IllegalArgumentException("strength only applies to img2img: give an initImage too, or drop strength");
                if (maskImage != null)
                    throw new IllegalArgumentException("maskImage marks what to repaint in the initImage: give an initImage too");
            }
            if (maskImage != null && (maskImage.getWidth() != initImage.getWidth() || maskImage.getHeight() != initImage.getHeight()))
                throw new IllegalArgumentException("maskImage is " + maskImage.getWidth() + "x" + maskImage.getHeight()
                        + " but initImage is " + initImage.getWidth() + "x" + initImage.getHeight() + "; they must match");
            if (controlStrength != null && controlImage == null)
                throw new IllegalArgumentException("controlStrength only applies with a controlImage");
            for (Lora l : loras)
                if (!Files.isRegularFile(l.file())) throw new IllegalArgumentException("LoRA file not found: " + l.file());
            Map<String, Object> o = new LinkedHashMap<>();
            o.put("prompt", prompt);
            o.put("width", width);
            o.put("height", height);
            o.put("sample.steps", steps);
            o.put("sample.guidance.txt_cfg", cfg);
            o.put("seed", seed);
            if (strength != null) o.put("strength", strength);
            if (controlStrength != null) o.put("control_strength", controlStrength);
            o.putAll(extra);
            checkHires(o);
            return o;
        }
    }

    /**
     * A video request. initImage / endImage (optional) pin the first / last frame (LTX, Wan). refAudio (optional)
     * drives speech-to-video models (Wan2.2 S2V, MiniMax-H3); with LTX it is the soundtrack itself (audio to video):
     * it is encoded with LTX's audio VAE and held fixed while the picture is generated to it, so a singer's mouth
     * follows the song. LTX needs it at least as long as the clip ({@link #withAudio} cuts it to fit). extra takes any
     * other sd.cpp video option key.
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

        /** The clip's length in seconds: frames / fps. */
        public double seconds() { return frames / (double) fps; }

        /**
         * Animates to {@code audio} from {@code startSeconds}: the clip-length section from there (padded with
         * silence past the end) becomes {@link #refAudio}. With LTX it is the output's soundtrack.
         */
        public VideoRequest withAudio(Audio audio, double startSeconds) {
            return withRefAudio(audio.section(startSeconds, seconds()));
        }

        /** {@link #withAudio(Audio, double)} from a WAV file. */
        public VideoRequest withAudio(java.nio.file.Path wav, double startSeconds) throws java.io.IOException {
            return withAudio(Audio.readWav(wav.toFile()), startSeconds);
        }
    }

    /** Generated video: its frames, frame rate, and the soundtrack when the model makes one (LTX with an audio VAE). */
    public record Video(List<BufferedImage> frames, int fps, Audio audio) {
        public double seconds() { return fps == 0 ? 0 : frames.size() / (double) fps; }
    }

    private final MemorySegment ctx;
    private final Arena arena = Arena.ofShared();
    private final ModelThread thread;

    public DiffusionModel(Config config) {
        Map<String, Object> options = config.resolved();
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment opt = (MemorySegment) Sd.ctx_opt_create.invokeExact();
            try {
                applyAll(opt, options, tmp);
                MemorySegment out = tmp.allocate(ADDRESS);
                Sd.check((int) Sd.ctx_create.invokeExact(opt, out), "sd_ctx_create");
                ctx = out.get(ADDRESS, 0);
            } finally {
                Sd.opt_free.invokeExact(opt);
            }
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        var file = java.nio.file.Path.of(String.valueOf(options.getOrDefault("diffusion_model_path",
                options.getOrDefault("model_path", "sd")))).getFileName();
        MemorySegment c = ctx;
        thread = new ModelThread(file == null ? "sd" : file.toString(), () -> {
            try { Sd.ctx_free.invokeExact(c); } catch (Throwable t) { throw new RuntimeException(t); }
            arena.close();
        }, () -> {
            try { Sd.cancel_generation.invokeExact(c, Sd.SD_CANCEL_ALL); } catch (Throwable t) { throw new RuntimeException(t); }
        }, null);
    }

    /**
     * Stops the run in progress (from any thread, including a progress sink) at the next step; that call throws a
     * {@link CancellationException}. Works for images and video. A call that hasn't started isn't affected: cancel
     * its future instead.
     */
    public void cancel() { thread.cancel(); }

    /** {@link #generate}, on this model's thread. Cancelling the future stops the run. */
    public CompletableFuture<List<BufferedImage>> generateAsync(Request r) { return thread.submit(c -> generateOn(c, r)); }

    /** {@link #generateVideo}, on this model's thread. Cancelling the future stops the run. */
    public CompletableFuture<Video> generateVideoAsync(VideoRequest r) { return thread.submit(c -> generateVideoOn(c, r)); }

    /** A status from a generate call: a sink's exception first, then cancellation, then any other failure. */
    private static void checkRun(Call call, int status, String what) {
        call.rethrow();
        if (status == Sd.SD_STATUS_CANCELLED) throw new CancellationException(what + " was cancelled");
        Sd.check(status, what);
    }

    /** Sd log and progress callbacks (process-wide). */
    public static void onLog(Sd.LogSink s) { Sd.setLogSink(s); }
    public static void onProgress(Sd.ProgressSink s) { Sd.setProgressSink(s); }

    public String modelVersion() {
        return thread.run(c -> {
            try {
                return ((MemorySegment) Sd.get_model_version_name.invokeExact(ctx)).reinterpret(Long.MAX_VALUE).getString(0);
            } catch (Throwable t) { throw new RuntimeException(t); }
        });
    }

    public List<BufferedImage> generate(Request r) { return thread.run(c -> generateOn(c, r)); }

    private List<BufferedImage> generateOn(Call call, Request r) {
        Map<String, Object> o = r.options();
        Sd.CALL.set(call);
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment opt = (MemorySegment) Sd.img_gen_opt_create.invokeExact();
            List<MemorySegment> owned = new ArrayList<>();
            try {
                applyAll(opt, o, tmp);
                if (!r.refImages().isEmpty()) {
                    MemorySegment arr = tmp.allocate(ADDRESS, r.refImages().size());
                    for (int i = 0; i < r.refImages().size(); i++) {
                        MemorySegment t = toTensor(r.refImages().get(i), tmp);
                        owned.add(t);
                        arr.setAtIndex(ADDRESS, i, t);
                    }
                    Sd.check((int) Sd.opt_set_tensors.invokeExact(opt, tmp.allocateFrom("ref_images"), arr, (long) r.refImages().size()), "set ref_images");
                }
                if (r.initImage() != null) setTensor(opt, "init_image", owned, toTensor(r.initImage(), tmp), tmp);
                if (r.maskImage() != null) setTensor(opt, "mask_image", owned, toMaskTensor(r.maskImage(), tmp), tmp);
                if (r.controlImage() != null) setTensor(opt, "control_image", owned, toTensor(r.controlImage(), tmp), tmp);
                if (!r.loras().isEmpty())
                    Sd.check((int) Sd.opt_set_loras.invokeExact(opt, tmp.allocateFrom("loras"), loraArray(r.loras(), tmp), (long) r.loras().size()), "set loras");
                MemorySegment out = tmp.allocate(ADDRESS);
                out.set(ADDRESS, 0, MemorySegment.NULL);
                int status = (int) Sd.generate_image.invokeExact(ctx, opt, out);
                MemorySegment result = out.get(ADDRESS, 0);
                try {
                    checkRun(call, status, "sd_generate_image");
                    long n = (long) Sd.result_get_image_count.invokeExact(result);
                    var images = new ArrayList<BufferedImage>();
                    for (long i = 0; i < n; i++) images.add(fromTensor((MemorySegment) Sd.result_get_image.invokeExact(result, i)));
                    return images;
                } finally {
                    if (!result.equals(MemorySegment.NULL)) Sd.result_free.invokeExact(result);
                }
            } finally {
                for (MemorySegment t : owned) Sd.tensor_free.invokeExact(t);
                Sd.opt_free.invokeExact(opt);
            }
        } catch (RuntimeException | Error e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { Sd.CALL.remove(); }
    }

    /** sd_lora_t: { bool is_high_noise; float multiplier; const char* path; } */
    static final StructLayout LORA = MemoryLayout.structLayout(
            JAVA_BOOLEAN.withName("is_high_noise"), MemoryLayout.paddingLayout(3),
            JAVA_FLOAT.withName("multiplier"), ADDRESS.withName("path"));

    /** An sd_lora_t array for these LoRAs, with absolute paths, allocated in {@code a}. */
    static MemorySegment loraArray(List<Lora> loras, Arena a) {
        long size = LORA.byteSize();
        long weight = LORA.byteOffset(MemoryLayout.PathElement.groupElement("multiplier"));
        long path = LORA.byteOffset(MemoryLayout.PathElement.groupElement("path"));
        MemorySegment arr = a.allocate(LORA, loras.size());
        for (int i = 0; i < loras.size(); i++) {
            arr.set(JAVA_BOOLEAN, i * size, false);   // high-noise LoRAs are for Wan2.2's second expert; images have none
            arr.set(JAVA_FLOAT, i * size + weight, loras.get(i).weight());
            arr.set(ADDRESS, i * size + path, a.allocateFrom(loras.get(i).file().toAbsolutePath().toString()));
        }
        return arr;
    }

    public Video generateVideo(VideoRequest r) { return thread.run(c -> generateVideoOn(c, r)); }

    private Video generateVideoOn(Call call, VideoRequest r) {
        Sd.CALL.set(call);
        try (Arena tmp = Arena.ofConfined()) {
            MemorySegment opt = (MemorySegment) Sd.vid_gen_opt_create.invokeExact();
            List<MemorySegment> owned = new ArrayList<>();
            try {
                applyAll(opt, videoOptions(r), tmp);
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
                out.set(ADDRESS, 0, MemorySegment.NULL);
                int status = (int) Sd.generate_video.invokeExact(ctx, opt, out);
                MemorySegment result = out.get(ADDRESS, 0);
                try {
                    checkRun(call, status, "sd_generate_video");
                    long n = (long) Sd.result_get_image_count.invokeExact(result);
                    var frames = new ArrayList<BufferedImage>((int) n);
                    for (long i = 0; i < n; i++) frames.add(fromTensor((MemorySegment) Sd.result_get_image.invokeExact(result, i)));
                    int fps = (int) Sd.result_get_fps.invokeExact(result);
                    return new Video(frames, fps, fromAudio((MemorySegment) Sd.result_get_audio.invokeExact(result)));
                } finally {
                    if (!result.equals(MemorySegment.NULL)) Sd.result_free.invokeExact(result);
                }
            } finally {
                for (MemorySegment t : owned) Sd.tensor_free.invokeExact(t);
                Sd.opt_free.invokeExact(opt);
            }
        } catch (RuntimeException | Error e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { Sd.CALL.remove(); }
    }

    /** The sd.cpp video options for a request (its images and audio go in as tensors, not options). */
    static Map<String, Object> videoOptions(VideoRequest r) {
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
        return o;
    }

    /** sd audio tensors are F32 [C, N], channel first: interleaved samples. */
    static float[] interleave(Audio audio) {
        int n = audio.left().length;
        float[] interleaved = new float[n * 2];
        for (int i = 0; i < n; i++) { interleaved[2 * i] = audio.left()[i]; interleaved[2 * i + 1] = audio.right()[i]; }
        return interleaved;
    }

    private static void setTensor(MemorySegment opt, String key, List<MemorySegment> owned, MemorySegment t, Arena a) throws Throwable {
        owned.add(t);
        Sd.check((int) Sd.opt_set_tensor.invokeExact(opt, a.allocateFrom(key), t), "set " + key);
    }

    static MemorySegment audioTensor(Audio audio, Arena a) throws Throwable {
        int n = audio.left().length;
        float[] interleaved = interleave(audio);
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
        return u8Tensor(rgb(img), 3, img.getWidth(), img.getHeight(), a);
    }

    /** A mask as one grey channel, the way sd-cli loads one (stb_image's luma, (77 R + 150 G + 29 B) / 256). */
    static MemorySegment toMaskTensor(BufferedImage img, Arena a) throws Throwable {
        return u8Tensor(grey(img), 1, img.getWidth(), img.getHeight(), a);
    }

    /** Interleaved RGB rows; a grey image's level goes to all three channels. */
    static byte[] rgb(BufferedImage img) {
        int w = img.getWidth(), h = img.getHeight();
        byte[] px = new byte[w * h * 3];
        byte[] levels = greyLevels(img);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int i = (y * w + x) * 3;
                if (levels != null) { px[i] = px[i + 1] = px[i + 2] = levels[y * w + x]; continue; }
                int rgb = img.getRGB(x, y);
                px[i] = (byte) (rgb >> 16); px[i + 1] = (byte) (rgb >> 8); px[i + 2] = (byte) rgb;
            }
        return px;
    }

    static byte[] grey(BufferedImage img) {
        byte[] levels = greyLevels(img);
        if (levels != null) return levels;
        int w = img.getWidth(), h = img.getHeight();
        byte[] px = new byte[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                px[y * w + x] = (byte) ((((rgb >> 16) & 0xff) * 77 + ((rgb >> 8) & 0xff) * 150 + (rgb & 0xff) * 29) >> 8);
            }
        return px;
    }

    /**
     * A grey image's stored levels as 8 bits (16-bit ones keep the high byte, as stb_image does), or null for any
     * other image. Not getRGB: ImageIO reads a greyscale PNG as linear grey, and getRGB converts that to sRGB (100
     * becomes 168), which would move a mask's 0.5 threshold and brighten a depth hint compared with sd-cli.
     */
    static byte[] greyLevels(BufferedImage img) {
        if (img.getColorModel().getColorSpace().getType() != java.awt.color.ColorSpace.TYPE_GRAY) return null;
        int bits = img.getColorModel().getComponentSize(0), w = img.getWidth(), h = img.getHeight();
        var raster = img.getRaster();
        byte[] px = new byte[w * h];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int s = raster.getSample(x, y, 0);
                px[y * w + x] = (byte) (bits >= 8 ? s >> (bits - 8) : s * 255 / ((1 << bits) - 1));
            }
        return px;
    }

    /** A U8 [C, W, H] tensor (channel first in shape, so the bytes are interleaved rows). */
    static MemorySegment u8Tensor(byte[] px, int channels, int w, int h, Arena a) throws Throwable {
        MemorySegment shape = a.allocateFrom(JAVA_LONG, channels, w, h);
        MemorySegment data = a.allocateFrom(JAVA_BYTE, px);
        MemorySegment t = (MemorySegment) Sd.tensor_create.invokeExact(Sd.SD_TENSOR_U8, 3L, shape, data, (long) px.length);
        if (t.equals(MemorySegment.NULL)) throw new Sd.SdException("sd_tensor_create: " + Sd.lastError());
        return t;
    }

    /**
     * Frees the model. Idempotent, and safe while a call is in flight on another thread: it cancels that call, fails
     * queued async calls with a CancellationException, waits for the run to stop, then frees.
     * @throws IllegalStateException if called from inside a log or progress sink during this model's run
     */
    @Override public void close() { thread.close(); }
}
