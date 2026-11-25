package org.liberationdata.kiln.studio;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.liberationdata.kiln.llama.LlamaModel;
import org.liberationdata.kiln.sd.DiffusionModel;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Pattern;

/**
 * The backend Kiln Studio (the desktop app) starts: Kiln's engines behind a small HTTP API on localhost, JDK only.
 *
 * <pre>
 * GET  /api/health                 {"ok": true, ...}
 * GET  /api/models                 the catalogue, what is loaded, and the memory in use
 * POST /api/models/load   {id}     loads a model into the pool
 * POST /api/models/unload {id}     unloads it ({"id": "*"} unloads every model not in use)
 * POST /api/chat                   streams a reply (server-sent events: status, token, done | error)
 * POST /api/chat/tools             the tool-calling demo through chatJson (status, token, assistant, tool, done | error)
 * POST /api/image                  generates an image (status, progress, done | error)
 * POST /api/song                   generates a song, with ACE-Step or YuE2 (status, done | error)
 * POST /api/video                  generates a video clip with sound, with LTX-2.5 (status, progress, done | error)
 * GET  /api/outputs/{file}         an image, song or video made, from the studio's home
 * </pre>
 *
 * Settings, as system properties or the environment: {@code kiln.studio.port} / {@code KILN_STUDIO_PORT} (5191),
 * {@code kiln.studio.budget} / {@code KILN_STUDIO_BUDGET_GB} (the pool's budget in GB, 24),
 * {@code kiln.studio.stub} / {@code KILN_STUDIO_STUB} (kinds answered by stubs, as {@code image,song,video}),
 * {@code kiln.studio.ffmpeg} / {@code KILN_STUDIO_FFMPEG} (the ffmpeg that makes a video's .mp4; else the PATH's),
 * {@code KILN_STUDIO_HOME} (~/.kiln-studio, where outputs are kept), {@code KILN_STUDIO_IDLE_MINUTES} (15), and
 * {@code KILN_STUDIO_PARENT_PID}: the app that started it, whose exit ends the server too, so a model never stays on
 * the GPU after the app has gone.
 */
public final class StudioServer {
    private static final Pattern OUTPUT_NAME = Pattern.compile("[A-Za-z0-9._-]+\\.(png|wav|json|mp4)");

    private final Studio studio;
    private final HttpServer http;
    private final String ffmpeg;

    StudioServer(Studio studio, int port) throws IOException {
        this(studio, port, Mp4.find(setting("kiln.studio.ffmpeg", "KILN_STUDIO_FFMPEG", "")));
    }

    /** ffmpeg: the one that makes a video's .mp4, or null for none (a video then fails, saying where its frames are). */
    StudioServer(Studio studio, int port, String ffmpeg) throws IOException {
        this.studio = studio;
        this.ffmpeg = ffmpeg;
        this.http = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), port), 0);
        // Platform threads: a generation is one long native call, which would pin a virtual thread's carrier throughout.
        http.setExecutor(Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "kiln-studio-http");
            t.setDaemon(true);
            return t;
        }));
        route("/api/health", "GET", this::health);
        route("/api/models", "GET", ex -> json(ex, 200, studio.status()));
        route("/api/models/load", "POST", this::load);
        route("/api/models/unload", "POST", this::unload);
        route("/api/chat", "POST", this::chat);
        route("/api/chat/tools", "POST", this::chatTools);
        route("/api/image", "POST", this::image);
        route("/api/song", "POST", this::song);
        route("/api/video", "POST", this::video);
        http.createContext("/api/outputs/", ex -> handle(ex, "GET", this::output));
    }

    int port() { return http.getAddress().getPort(); }

    void start() { http.start(); }

    void stop() { http.stop(0); }

    public static void main(String[] args) throws Exception {
        int port = Integer.parseInt(setting("kiln.studio.port", "KILN_STUDIO_PORT", "5191"));
        long budget = (long) (Double.parseDouble(setting("kiln.studio.budget", "KILN_STUDIO_BUDGET_GB", "24")) * (1L << 30));
        Set<Catalog.Kind> stubs = EnumSet.noneOf(Catalog.Kind.class);
        for (String k : setting("kiln.studio.stub", "KILN_STUDIO_STUB", "").split(",")) {
            if (!k.isBlank()) stubs.add(Catalog.Kind.valueOf(k.trim()));
        }
        Path home = Path.of(setting("kiln.studio.home", "KILN_STUDIO_HOME", System.getProperty("user.home") + "/.kiln-studio"));
        Duration idle = Duration.ofMinutes(Long.parseLong(setting("kiln.studio.idle", "KILN_STUDIO_IDLE_MINUTES", "15")));

        var studio = new Studio(new Catalog(), budget, idle, stubs, home.resolve("outputs"));
        var server = new StudioServer(studio, port);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            server.stop();
            studio.close();  // frees every model's memory on the GPU
        }, "kiln-studio-shutdown"));
        String parent = System.getenv("KILN_STUDIO_PARENT_PID");
        if (parent != null && !parent.isBlank()) {
            ProcessHandle.of(Long.parseLong(parent.trim())).ifPresentOrElse(
                    p -> p.onExit().thenRun(() -> System.exit(0)),
                    () -> System.exit(0));
        }
        server.start();
        // The line the desktop app waits for (it then asks /api/health, as it does of the Next.js server).
        System.out.println("[kiln-studio] listening on http://127.0.0.1:" + server.port() + " (cache " + studio.catalog().root()
                + ", budget " + budget / (1L << 30) + " GB" + (stubs.isEmpty() ? "" : ", stubs " + stubs) + ")");
    }

    private static String setting(String property, String env, String dflt) {
        String v = System.getProperty(property);
        if (v == null || v.isBlank()) v = System.getenv(env);
        return v == null || v.isBlank() ? dflt : v;
    }

    // ---- routes ----

    private void health(HttpExchange ex) throws IOException {
        json(ex, 200, Json.obj("ok", true, "name", "kiln-studio-server", "java", Runtime.version().toString(),
                "kilnLibraryPath", System.getProperty("kiln.library.path", System.getenv("KILN_LIBRARY_PATH")),
                "cacheRoot", studio.catalog().root().toString(), "stubs", studio.stubs().stream().map(Enum::name).sorted().toList(),
                "ffmpeg", ffmpeg));
    }

    private void load(HttpExchange ex) throws IOException {
        String id = Json.str(body(ex), "id", "");
        var m = studio.catalog().find(id).orElseThrow(() -> new IllegalArgumentException("no model " + id));
        studio.model(id, m.kind());
        try (var lease = studio.lease(m, AutoCloseable.class)) {
            lease.model();
        }
        json(ex, 200, Json.obj("id", id, "loaded", true, "loadSeconds", studio.loadSeconds(id)));
    }

    private void unload(HttpExchange ex) throws IOException {
        String id = Json.str(body(ex), "id", "");
        if (id.equals("*")) {
            studio.unloadAll();
            json(ex, 200, Json.obj("id", id, "unloaded", true));
        } else {
            json(ex, 200, Json.obj("id", id, "unloaded", studio.unload(id)));
        }
    }

    /**
     * {"model": id, "messages": [{"role", "content"}], "maxTokens", "temperature", "seed", "think": false}
     * think false adds Qwen3's /no_think switch to the last user message.
     */
    private void chat(HttpExchange ex) throws IOException {
        Map<String, Object> req = body(ex);
        var m = studio.model(Json.str(req, "model", ""), Catalog.Kind.chat);
        List<LlamaModel.Message> messages = new ArrayList<>();
        for (Object o : Json.list(req, "messages")) {
            @SuppressWarnings("unchecked") var msg = (Map<String, Object>) o;
            messages.add(new LlamaModel.Message(Json.str(msg, "role", "user"), Json.str(msg, "content", "")));
        }
        if (messages.isEmpty()) throw new IllegalArgumentException("no messages");
        if (!Json.bool(req, "think", true)) {
            int last = messages.size() - 1;
            if (messages.get(last).role().equals("user")) {
                messages.set(last, LlamaModel.Message.user(messages.get(last).content() + " /no_think"));
            }
        }
        var options = new LlamaModel.Options((int) Json.num(req, "maxTokens", 1024), (float) Json.dbl(req, "temperature", 0.7),
                (float) Json.dbl(req, "topP", 0.9), (float) Json.dbl(req, "minP", 0.05), (int) Json.num(req, "seed", -1));

        try (var sse = new Sse(ex)) {
            try (var lease = leaseSaying(sse, m, Engines.Chat.class)) {
                var pieces = new AtomicInteger();
                var first = new AtomicLong();
                long t0 = System.nanoTime();
                var r = lease.model().chat(messages, options, piece -> {
                    if (first.get() == 0) first.set(System.nanoTime());
                    pieces.incrementAndGet();
                    sse.send("token", Json.obj("text", piece));
                });
                sse.send("done", timing(t0, first.get(), r.promptTokens(), r.generatedTokens(), Json.obj("text", r.text())));
            } catch (RuntimeException e) {
                sse.send("error", Json.obj("message", message(e)));
            }
        }
    }

    /**
     * The tool-calling demo: {"model", "messages", "maxTokens", "temperature"}. The model is offered {@link Tools}; each
     * call it makes is run here and its answer given back, until it answers in words (at most six rounds).
     */
    @SuppressWarnings("unchecked")
    private void chatTools(HttpExchange ex) throws IOException {
        Map<String, Object> req = body(ex);
        var m = studio.model(Json.str(req, "model", ""), Catalog.Kind.chat);
        List<Object> messages = new ArrayList<>(Json.list(req, "messages"));
        if (messages.isEmpty()) throw new IllegalArgumentException("no messages");

        try (var sse = new Sse(ex)) {
            try (var lease = leaseSaying(sse, m, Engines.Chat.class)) {
                long t0 = System.nanoTime();
                var first = new AtomicLong();
                long prompt = 0, generated = 0;
                for (int round = 0; round < 6; round++) {
                    var call = Json.obj("messages", messages, "tools", Tools.DEFINITIONS, "tool_choice", "auto",
                            "max_tokens", Json.num(req, "maxTokens", 1024), "temperature", Json.dbl(req, "temperature", 0.7));
                    if (req.get("seed") != null) call.put("seed", Json.num(req, "seed", -1));
                    String reply = lease.model().chatJson(Json.write(call), piece -> {
                        if (first.get() == 0) first.set(System.nanoTime());
                        sse.send("token", Json.obj("text", piece));
                    });
                    Map<String, Object> r = Json.object(reply);
                    var usage = r.get("usage") instanceof Map<?, ?> u ? (Map<String, Object>) u : Map.<String, Object>of();
                    prompt += Json.num(usage, "prompt_tokens", 0);
                    generated += Json.num(usage, "completion_tokens", 0);
                    var message = (Map<String, Object>) r.getOrDefault("message", Json.obj("role", "assistant", "content", ""));
                    sse.send("assistant", Json.obj("message", message, "finishReason", r.get("finish_reason")));
                    List<Object> calls = Json.list(message, "tool_calls");
                    messages.add(message);
                    if (calls.isEmpty()) break;
                    for (Object c : calls) {
                        var tc = (Map<String, Object>) c;
                        var fn = (Map<String, Object>) tc.getOrDefault("function", Map.of());
                        String name = Json.str(fn, "name", "");
                        Object rawArgs = fn.get("arguments");
                        Map<String, Object> args;
                        try {
                            args = rawArgs instanceof Map<?, ?> a ? (Map<String, Object>) a
                                    : rawArgs == null || rawArgs.toString().isBlank() ? Map.of() : Json.object(rawArgs.toString());
                        } catch (IllegalArgumentException bad) {
                            args = Map.of();
                        }
                        String result = Tools.call(name, args);
                        sse.send("tool", Json.obj("id", tc.get("id"), "name", name, "arguments", args, "result", result));
                        messages.add(Json.obj("role", "tool", "tool_call_id", Json.str(tc, "id", name), "name", name, "content", result));
                    }
                }
                var done = timing(t0, first.get(), prompt, generated, Json.obj());
                // chatJson streams only the reply's words, not its reasoning or tool calls: over the whole exchange, then.
                done.put("tokensPerSecond", generated / Math.max(1e-9, (double) done.get("seconds")));
                sse.send("done", done);
            } catch (RuntimeException e) {
                sse.send("error", Json.obj("message", message(e)));
            }
        }
    }

    /** {"model", "prompt", "width", "height", "steps", "cfg", "seed"}: the preset's defaults for those left out. */
    private void image(HttpExchange ex) throws IOException {
        Map<String, Object> req = body(ex);
        var m = studio.model(Json.str(req, "model", ""), Catalog.Kind.image);
        var d = m.defaults();
        String prompt = Json.str(req, "prompt", "").trim();
        if (prompt.isEmpty()) throw new IllegalArgumentException("no prompt");
        int width = (int) Json.num(req, "width", Json.num(d, "width", 768)), height = (int) Json.num(req, "height", Json.num(d, "height", 768));
        int steps = (int) Json.num(req, "steps", Json.num(d, "steps", 20));
        double cfg = Json.dbl(req, "cfg", Json.dbl(d, "cfg", 4.0));
        long seed = Json.num(req, "seed", -1);
        if (seed < 0) seed = (long) (Math.random() * Integer.MAX_VALUE);
        if (width % 16 != 0 || height % 16 != 0 || width < 256 || height < 256 || width > 2048 || height > 2048) {
            throw new IllegalArgumentException("width and height are multiples of 16, from 256 to 2048");
        }
        var request = new DiffusionModel.Request(prompt, width, height, steps, cfg, seed);

        try (var sse = new Sse(ex)) {
            try (var lease = leaseSaying(sse, m, Engines.Painter.class)) {
                long t0 = System.nanoTime();
                var img = lease.model().paint(request, (s, n, secs) -> sse.send("progress", Json.obj("step", s, "steps", n, "secondsPerStep", secs)));
                double seconds = (System.nanoTime() - t0) / 1e9;
                String name = "image-" + stamp() + ".png";
                ImageIO.write(img, "png", studio.outputs().resolve(name).toFile());
                sse.send("done", Json.obj("url", "/api/outputs/" + name, "file", studio.outputs().resolve(name).toString(),
                        "width", width, "height", height, "steps", steps, "cfg", cfg, "seed", seed, "seconds", seconds,
                        "secondsPerStep", seconds / Math.max(1, steps), "stub", studio.stubbed(Catalog.Kind.image)));
            } catch (RuntimeException e) {
                sse.send("error", Json.obj("message", message(e)));
            }
        }
    }

    /**
     * {"model", "caption", "lyrics", "duration", "bpm", "keyscale", "seed", "steps", "cfg", "shift", "language"}, made into
     * the engine's own request (as kiln-examples' requests are), or {"request": {...}} to send one as it is. For YuE2 the
     * caption is its style (genre, voice, instruments, and the tempo and key in words), and only the lyrics, duration,
     * seed and steps go with it.
     */
    @SuppressWarnings("unchecked")
    private void song(HttpExchange ex) throws IOException {
        Map<String, Object> req = body(ex);
        var m = studio.model(Json.str(req, "model", ""), Catalog.Kind.song);
        var d = m.defaults();
        boolean yue2 = m.engine().equals("yue2.cpp");
        Map<String, Object> ace;
        if (req.get("request") instanceof Map<?, ?> given) {
            ace = (Map<String, Object>) given;
        } else if (yue2) {
            String style = Json.str(req, "caption", "").trim();
            if (style.isEmpty()) throw new IllegalArgumentException("no style: describe the music");
            long seed = Json.num(req, "seed", -1);
            if (seed < 0) seed = (long) (Math.random() * Integer.MAX_VALUE);
            // yue2-rusty.json's fields; YuE2's own defaults for the rest (full chain of thought, 32 steps).
            ace = Json.obj("style", style, "lyrics", Json.str(req, "lyrics", ""),
                    "duration", Json.num(req, "duration", Json.num(d, "duration", 60)),
                    "seed", seed, "lm_seed", seed, "output_format", "wav16");
            if (req.get("steps") != null) ace.put("steps", Json.num(req, "steps", 32));
        } else {
            String caption = Json.str(req, "caption", "").trim();
            if (caption.isEmpty()) throw new IllegalArgumentException("no caption: describe the music");
            long seed = Json.num(req, "seed", -1);
            if (seed < 0) seed = (long) (Math.random() * Integer.MAX_VALUE);
            ace = Json.obj("caption", caption, "lyrics", Json.str(req, "lyrics", ""),
                    "bpm", Json.num(req, "bpm", Json.num(d, "bpm", 96)),
                    "duration", Json.num(req, "duration", Json.num(d, "duration", 60)),
                    "keyscale", Json.str(req, "keyscale", Json.str(d, "keyscale", "C major")),
                    "timesignature", "4", "vocal_language", Json.str(req, "language", "en"),
                    "seed", seed, "lm_seed", seed,
                    "inference_steps", Json.num(req, "steps", Json.num(d, "steps", 8)),
                    "guidance_scale", Json.dbl(req, "cfg", Json.dbl(d, "cfg", 1.0)),
                    "shift", Json.dbl(req, "shift", Json.dbl(d, "shift", 3.0)),
                    "output_format", "wav16");
        }
        if (Json.dbl(ace, "duration", 0) > 600) throw new IllegalArgumentException("at most 600 seconds");
        String requestJson = Json.write(ace);

        try (var sse = new Sse(ex)) {
            try (var lease = leaseSaying(sse, m, Engines.Singer.class)) {
                sse.send("status", Json.obj("state", "generating"));
                long t0 = System.nanoTime();
                var r = lease.model().sing(requestJson);
                double seconds = (System.nanoTime() - t0) / 1e9;
                String name = "song-" + stamp();
                r.audio().writeWav(studio.outputs().resolve(name + ".wav").toFile());
                if (r.resolvedRequestJson() != null) Files.writeString(studio.outputs().resolve(name + ".json"), r.resolvedRequestJson());
                if (r.score() != null) Files.writeString(studio.outputs().resolve(name + ".abc"), r.score());
                sse.send("done", Json.obj("url", "/api/outputs/" + name + ".wav", "file", studio.outputs().resolve(name + ".wav").toString(),
                        "seconds", seconds, "audioSeconds", r.audio().seconds(), "realtime", r.audio().seconds() / Math.max(1e-9, seconds),
                        "request", new Json.Raw(r.resolvedRequestJson() != null ? r.resolvedRequestJson() : requestJson),
                        "score", r.score(), "license", m.license(), "stub", studio.stubbed(Catalog.Kind.song)));
            } catch (RuntimeException e) {
                sse.send("error", Json.obj("message", message(e)));
            }
        }
    }

    // VideoClip's two-stage recipe (ComfyUI's "Image to Video (LTX-2.5)", as Endless Country used it; docs/engine-notes.md).
    private static final float[] DRAFT_SIGMAS = {1.0f, 0.99375f, 0.9875f, 0.98125f, 0.975f, 0.909375f, 0.725f, 0.421875f, 0.0f};
    private static final float[] REFINE_SIGMAS = {0.85f, 0.725f, 0.4219f, 0.0f};

    /**
     * {"model", "prompt", "image" (the first frame: a data: URL or base64 of a PNG or JPEG), "width", "height", "frames",
     * "fps", "steps", "cfg", "seed", "audio" (true), "twoStage" (true), "strength"}: the preset's defaults for those left
     * out. Two-stage drafts at width x height with manual sigmas, upscales the latent x2 with the learned upscaler and
     * refines, so the clip is twice the size; otherwise one pass at width x height in steps. Without audio, the model is
     * loaded without its audio VAE.
     */
    private void video(HttpExchange ex) throws IOException {
        Map<String, Object> req = body(ex);
        var m = studio.model(Json.str(req, "model", ""), Catalog.Kind.video);
        var d = m.defaults();
        String prompt = Json.str(req, "prompt", "").trim();
        if (prompt.isEmpty()) throw new IllegalArgumentException("no prompt");
        int width = (int) Json.num(req, "width", Json.num(d, "width", 672)), height = (int) Json.num(req, "height", Json.num(d, "height", 448));
        int frames = (int) Json.num(req, "frames", Json.num(d, "frames", 49)), fps = (int) Json.num(req, "fps", Json.num(d, "fps", 24));
        int steps = (int) Json.num(req, "steps", Json.num(d, "steps", 8));
        double cfg = Json.dbl(req, "cfg", Json.dbl(d, "cfg", 1.0));
        boolean twoStage = Json.bool(req, "twoStage", true), audio = Json.bool(req, "audio", true);
        long seed = Json.num(req, "seed", -1);
        if (seed < 0) seed = (long) (Math.random() * Integer.MAX_VALUE);
        if (width % 16 != 0 || height % 16 != 0 || width < 128 || height < 128 || width > 1536 || height > 1536) {
            throw new IllegalArgumentException("width and height are multiples of 16, from 128 to 1536");
        }
        if (frames < 9 || frames % 8 != 1) throw new IllegalArgumentException("frames are 8n + 1, from 9 up");
        if (fps < 1 || fps > 60) throw new IllegalArgumentException("fps is from 1 to 60");
        BufferedImage first = image(Json.str(req, "image", ""));

        Map<String, Object> extra = new java.util.HashMap<>(Map.of("sample.method", "euler", "sample.scheduler", "ltx2"));
        if (twoStage) {
            extra.put("sample.method", "euler_a");
            extra.put("sample.sigmas", DRAFT_SIGMAS);
            extra.put("strength", Json.dbl(req, "strength", Json.dbl(d, "strength", 0.7)));
            extra.put("hires.enabled", true);
            // sd.cpp runs the learned upscaler only when hires.upscaler is "model"; it otherwise interpolates, which ghosts.
            extra.put("hires.upscaler", "model");
            extra.put("hires.model_path", m.path("upscaler").toString());
            extra.put("hires.sigmas", REFINE_SIGMAS);
            // Tiled decode: short GPU dispatches stay under the driver's compute timeout.
            extra.put("vae_tiling.enabled", true);
        }
        var request = new DiffusionModel.VideoRequest(prompt, width, height, frames, fps, steps, cfg, seed).withExtra(extra);
        // LTX was trained on compressed frames: the keyframe goes through H.264 at CRF 18 first, as LTXVPreprocess does.
        if (first != null) request = request.withInitImage(twoStage ? Mp4.preprocess(first, 18, ffmpeg) : first);
        var model = audio ? m : studio.silent(m);

        try (var sse = new Sse(ex)) {
            try (var lease = leaseSaying(sse, model, Engines.Filmmaker.class)) {
                sse.send("status", Json.obj("state", "generating"));
                long t0 = System.nanoTime();
                // The draft's steps, then the refine's: each stage counts from 1 again.
                var stage = new AtomicInteger(1);
                var last = new AtomicInteger();
                var video = lease.model().film(request, (s, n, secs) -> {
                    if (s < last.getAndSet(s)) stage.incrementAndGet();
                    sse.send("progress", Json.obj("step", s, "steps", n, "secondsPerStep", secs, "stage", stage.get(),
                            "stages", twoStage ? 2 : 1));
                });
                double seconds = (System.nanoTime() - t0) / 1e9;
                if (video.frames().isEmpty()) throw new IllegalStateException("the model made no frames");
                sse.send("status", Json.obj("state", "encoding"));
                String name = "video-" + stamp() + ".mp4";
                try {
                    Mp4.write(video, studio.outputs().resolve(name), ffmpeg);
                } catch (IOException e) {
                    throw new java.io.UncheckedIOException(e);
                }
                var f0 = video.frames().get(0);
                sse.send("done", Json.obj("url", "/api/outputs/" + name, "file", studio.outputs().resolve(name).toString(),
                        "width", f0.getWidth(), "height", f0.getHeight(), "frames", video.frames().size(), "fps", video.fps(),
                        "videoSeconds", video.seconds(), "audio", video.audio() != null, "steps", steps, "cfg", cfg, "seed", seed,
                        "twoStage", twoStage, "seconds", seconds, "stub", studio.stubbed(Catalog.Kind.video)));
            } catch (RuntimeException e) {
                sse.send("error", Json.obj("message", message(e)));
            }
        }
    }

    /** An image sent in a request, as a data: URL or bare base64 (null for none). */
    private static BufferedImage image(String data) throws IOException {
        if (data.isBlank()) return null;
        String b64 = data.startsWith("data:") ? data.substring(data.indexOf(',') + 1) : data;
        byte[] bytes;
        try { bytes = java.util.Base64.getMimeDecoder().decode(b64); }
        catch (IllegalArgumentException e) { throw new IllegalArgumentException("the image is not base64"); }
        BufferedImage img = ImageIO.read(new java.io.ByteArrayInputStream(bytes));
        if (img == null) throw new IllegalArgumentException("the image is not a PNG or JPEG");
        return img;
    }

    private void output(HttpExchange ex) throws IOException {
        String name = ex.getRequestURI().getPath().substring("/api/outputs/".length());
        if (!OUTPUT_NAME.matcher(name).matches()) { json(ex, 404, Json.obj("error", "not found")); return; }
        Path file = studio.outputs().resolve(name);
        if (!Files.isRegularFile(file)) { json(ex, 404, Json.obj("error", "not found")); return; }
        String type = name.endsWith(".png") ? "image/png" : name.endsWith(".wav") ? "audio/wav"
                : name.endsWith(".mp4") ? "video/mp4" : "application/json";
        long size = Files.size(file);
        var h = ex.getResponseHeaders();
        h.set("Content-Type", type);
        h.set("Accept-Ranges", "bytes");
        // One byte range, so the audio player can seek.
        String range = ex.getRequestHeaders().getFirst("Range");
        long from = 0, to = size - 1;
        int status = 200;
        if (range != null && range.startsWith("bytes=") && !range.contains(",")) {
            String[] fromTo = range.substring(6).split("-", 2);
            try {
                if (fromTo[0].isEmpty()) from = Math.max(0, size - Long.parseLong(fromTo[1]));
                else {
                    from = Long.parseLong(fromTo[0]);
                    if (fromTo.length > 1 && !fromTo[1].isEmpty()) to = Math.min(size - 1, Long.parseLong(fromTo[1]));
                }
                status = 206;
                h.set("Content-Range", "bytes " + from + "-" + to + "/" + size);
            } catch (NumberFormatException ignored) {
                from = 0;
                to = size - 1;
            }
        }
        long length = Math.max(0, to - from + 1);
        if ("HEAD".equals(ex.getRequestMethod())) { ex.sendResponseHeaders(status, -1); ex.close(); return; }
        ex.sendResponseHeaders(status, length);
        try (InputStream in = Files.newInputStream(file); OutputStream out = ex.getResponseBody()) {
            in.skipNBytes(from);
            byte[] buf = new byte[1 << 16];
            long left = length;
            while (left > 0) {
                int n = in.read(buf, 0, (int) Math.min(buf.length, left));
                if (n < 0) break;
                out.write(buf, 0, n);
                left -= n;
            }
        }
    }

    // ---- helpers ----

    /** Borrows the model, saying first when it has to be loaded, and how long that took. */
    private <M extends AutoCloseable> org.liberationdata.kiln.ModelPool.Lease<M> leaseSaying(Sse sse, Catalog.Model m, Class<M> type) {
        boolean cold = !studio.isLoaded(m.id());
        if (cold) sse.send("status", Json.obj("state", "loading", "model", m.id(), "sizeBytes", m.sizeBytes()));
        var lease = studio.lease(m, type);
        if (cold) sse.send("status", Json.obj("state", "loaded", "model", m.id(), "loadSeconds", studio.loadSeconds(m.id())));
        return lease;
    }

    /** Seconds overall, to the first token, and tokens per second while generating (after the prompt is read). */
    private static Map<String, Object> timing(long t0, long first, long promptTokens, long generated, Map<String, Object> more) {
        long end = System.nanoTime();
        double seconds = (end - t0) / 1e9;
        double firstToken = first == 0 ? seconds : (first - t0) / 1e9;
        double generating = first == 0 ? 0 : (end - first) / 1e9;
        // The first token's time is the prompt's; the rest are generated in the time after it.
        double tps = generating > 0 && generated > 1 ? (generated - 1) / generating : 0;
        var out = Json.obj("seconds", seconds, "firstTokenSeconds", firstToken, "promptTokens", promptTokens,
                "generatedTokens", generated, "tokensPerSecond", tps);
        out.putAll(more);
        return out;
    }

    private static String stamp() {
        return java.time.LocalDateTime.now().format(java.time.format.DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"))
                + "-" + UUID.randomUUID().toString().substring(0, 6);
    }

    /** What went wrong, in words: the cause's, where Kiln wrapped a native failure in a RuntimeException of its own. */
    private static String message(Throwable e) {
        Throwable t = e;
        while (t.getCause() != null && (t.getMessage() == null || t.getMessage().equals(t.getCause().toString()))) t = t.getCause();
        return t.getMessage() != null ? t.getMessage() : t.toString();
    }

    interface Handler { void handle(HttpExchange ex) throws IOException; }

    private void route(String path, String method, Handler h) {
        // createContext matches by prefix; a route is its path alone.
        http.createContext(path, ex -> {
            if (!ex.getRequestURI().getPath().equals(path)) { json(ex, 404, Json.obj("error", "not found")); return; }
            handle(ex, method, h);
        });
    }

    private static void handle(HttpExchange ex, String method, Handler h) throws IOException {
        try {
            String m = ex.getRequestMethod();
            if (!m.equals(method) && !(method.equals("GET") && m.equals("HEAD"))) {
                json(ex, 405, Json.obj("error", "method not allowed"));
                return;
            }
            h.handle(ex);
        } catch (IllegalArgumentException e) {
            tryJson(ex, 400, e);
        } catch (RuntimeException | IOException | Error e) {
            System.err.println("[kiln-studio] " + ex.getRequestMethod() + " " + ex.getRequestURI() + ": " + e);
            tryJson(ex, 500, e);
        } finally {
            ex.close();
        }
    }

    // A failure before a reply has begun is sent as JSON; after (a stream under way), there is nothing more to send.
    private static void tryJson(HttpExchange ex, int status, Throwable e) {
        try { json(ex, status, Json.obj("error", message(e))); } catch (IOException | RuntimeException ignored) {}
    }

    private static Map<String, Object> body(HttpExchange ex) throws IOException {
        String s = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        return s.isBlank() ? new java.util.LinkedHashMap<>() : Json.object(s);
    }

    private static void json(HttpExchange ex, int status, Object body) throws IOException {
        byte[] b = Json.write(body).getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        if ("HEAD".equals(ex.getRequestMethod())) { ex.sendResponseHeaders(status, -1); return; }
        ex.sendResponseHeaders(status, b.length);
        try (OutputStream out = ex.getResponseBody()) { out.write(b); }
    }
}
