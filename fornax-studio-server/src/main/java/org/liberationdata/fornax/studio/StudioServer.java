package org.liberationdata.fornax.studio;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.sd.DiffusionModel;

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
 * The backend Fornax Studio (the desktop app) starts: Fornax's engines behind a small HTTP API on localhost, JDK only.
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
 * POST /api/inspect                asks a vision model about an image, free text or to a JSON schema (status, token, done | error)
 * GET  /api/outputs/{file}         an image, song or video made, from the studio's home
 * GET  /api/catalogue              Fornax's model sets: what of each is on disk, licence, gated, downloading
 * POST /api/models/plan  {id, token?}   what getting one would download, and the free disk ({@link Pulls#plan})
 * POST /api/models/pull  {id, token?}   downloads it (start, progress, done | cancelled | error), or watches its download
 * POST /api/models/pull/cancel {id}     stops its download; what has arrived stays, for the next pull to resume
 * </pre>
 *
 * A {@code token} is a Hugging Face token for a gated repo, from the app's keychain; it is sent to the Hub only, and
 * never logged. Without one, the hub's own is used ({@code HF_TOKEN}, or the token file {@code hf auth login} writes).
 *
 * Settings, as system properties or the environment: {@code fornax.studio.port} / {@code FORNAX_STUDIO_PORT} (5191),
 * {@code fornax.studio.budget} / {@code FORNAX_STUDIO_BUDGET_GB} (the pool's budget in GB, 24),
 * {@code fornax.studio.stub} / {@code FORNAX_STUDIO_STUB} (kinds answered by stubs, as {@code image,song,video}),
 * {@code fornax.studio.ffmpeg} / {@code FORNAX_STUDIO_FFMPEG} (the ffmpeg that makes a video's .mp4; else the PATH's),
 * {@code FORNAX_STUDIO_HOME} (~/.fornax-studio, where outputs are kept), {@code FORNAX_STUDIO_IDLE_MINUTES} (15), and
 * {@code FORNAX_STUDIO_PARENT_PID}: the app that started it, whose exit ends the server too, so a model never stays on
 * the GPU after the app has gone.
 */
public final class StudioServer {
    private static final Pattern OUTPUT_NAME = Pattern.compile("[A-Za-z0-9._-]+\\.(png|wav|json|mp4)");

    private final Studio studio;
    private final HttpServer http;
    private final String ffmpeg;
    private final Pulls pulls;

    StudioServer(Studio studio, int port) throws IOException {
        this(studio, port, Mp4.find(setting("fornax.studio.ffmpeg", "FORNAX_STUDIO_FFMPEG", "")));
    }

    /** ffmpeg: the one that makes a video's .mp4, or null for none (a video then fails, saying where its frames are). */
    StudioServer(Studio studio, int port, String ffmpeg) throws IOException {
        this(studio, port, ffmpeg, Pulls.standard(studio.catalog()));
    }

    /** pulls: where models are got from (a test's hub on localhost). */
    StudioServer(Studio studio, int port, String ffmpeg, Pulls pulls) throws IOException {
        this.studio = studio;
        this.ffmpeg = ffmpeg;
        this.pulls = pulls;
        this.http = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), port), 0);
        // Platform threads: a generation is one long native call, which would pin a virtual thread's carrier throughout.
        http.setExecutor(Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "fornax-studio-http");
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
        route("/api/inspect", "POST", this::inspect);
        http.createContext("/api/outputs/", ex -> handle(ex, "GET", this::output));
        route("/api/catalogue", "GET", ex -> json(ex, 200, Json.obj("models", pulls.catalogue())));
        route("/api/models/plan", "POST", ex -> {
            var b = body(ex);
            json(ex, 200, pulls.plan(Json.str(b, "id", null), Json.str(b, "token", null)));
        });
        route("/api/models/pull", "POST", this::pull);
        route("/api/models/pull/cancel", "POST", ex -> json(ex, 200, Json.obj("cancelled", pulls.cancel(Json.str(body(ex), "id", null)))));
    }

    int port() { return http.getAddress().getPort(); }

    void start() { http.start(); }

    void stop() { http.stop(0); }

    public static void main(String[] args) throws Exception {
        int port = Integer.parseInt(setting("fornax.studio.port", "FORNAX_STUDIO_PORT", "5191"));
        long budget = (long) (Double.parseDouble(setting("fornax.studio.budget", "FORNAX_STUDIO_BUDGET_GB", "24")) * (1L << 30));
        Set<Catalog.Kind> stubs = EnumSet.noneOf(Catalog.Kind.class);
        for (String k : setting("fornax.studio.stub", "FORNAX_STUDIO_STUB", "").split(",")) {
            if (!k.isBlank()) stubs.add(Catalog.Kind.valueOf(k.trim()));
        }
        Path home = Path.of(setting("fornax.studio.home", "FORNAX_STUDIO_HOME", defaultHome(Path.of(System.getProperty("user.home"))).toString()));
        Duration idle = Duration.ofMinutes(Long.parseLong(setting("fornax.studio.idle", "FORNAX_STUDIO_IDLE_MINUTES", "15")));

        var studio = new Studio(new Catalog(), budget, idle, stubs, home.resolve("outputs"));
        var server = new StudioServer(studio, port);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            server.stop();
            studio.close();  // frees every model's memory on the GPU
        }, "fornax-studio-shutdown"));
        String parent = System.getenv("FORNAX_STUDIO_PARENT_PID");
        if (parent != null && !parent.isBlank()) {
            ProcessHandle.of(Long.parseLong(parent.trim())).ifPresentOrElse(
                    p -> p.onExit().thenRun(() -> System.exit(0)),
                    () -> System.exit(0));
        }
        server.start();
        // The line the desktop app waits for (it then asks /api/health, as it does of the Next.js server).
        System.out.println("[fornax-studio] listening on http://127.0.0.1:" + server.port() + " (cache " + studio.catalog().root()
                + ", budget " + budget / (1L << 30) + " GB" + (stubs.isEmpty() ? "" : ", stubs " + stubs) + ")");
    }

    /** {@code ~/.fornax-studio}, or {@code ~/.kiln-studio} from before the rename when only that one exists, so earlier outputs stay. */
    static Path defaultHome(Path userHome) {
        Path home = userHome.resolve(".fornax-studio"), old = userHome.resolve(".kiln-studio");
        return !Files.exists(home) && Files.isDirectory(old) ? old : home;
    }

    private static String setting(String property, String env, String dflt) {
        String v = System.getProperty(property);
        if (v == null || v.isBlank()) v = System.getenv(env);
        return v == null || v.isBlank() ? dflt : v;
    }

    // ---- routes ----

    private void health(HttpExchange ex) throws IOException {
        json(ex, 200, Json.obj("ok", true, "name", "fornax-studio-server", "java", Runtime.version().toString(),
                "fornaxLibraryPath", System.getProperty("fornax.library.path", System.getenv("FORNAX_LIBRARY_PATH")),
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
                    sse.stream("token", Json.obj("text", piece));
                });
                sse.send("done", timing(t0, first.get(), r.promptTokens(), r.generatedTokens(), Json.obj("text", r.text())));
            } catch (RuntimeException | LinkageError e) {
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
                        sse.stream("token", Json.obj("text", piece));
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
            } catch (RuntimeException | LinkageError e) {
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
                var img = lease.model().paint(request, (s, n, secs) -> sse.stream("progress", Json.obj("step", s, "steps", n, "secondsPerStep", secs)));
                double seconds = (System.nanoTime() - t0) / 1e9;
                String name = "image-" + stamp() + ".png";
                ImageIO.write(img, "png", studio.outputs().resolve(name).toFile());
                sse.send("done", Json.obj("url", "/api/outputs/" + name, "file", studio.outputs().resolve(name).toString(),
                        "width", width, "height", height, "steps", steps, "cfg", cfg, "seed", seed, "seconds", seconds,
                        "secondsPerStep", seconds / Math.max(1, steps), "stub", studio.stubbed(Catalog.Kind.image)));
            } catch (RuntimeException | LinkageError e) {
                sse.send("error", Json.obj("message", message(e)));
            }
        }
    }

    /**
     * {"model", "caption", "lyrics", "duration", "bpm", "keyscale", "seed", "steps", "cfg", "shift", "language"}, made into
     * the engine's own request (as fornax-examples' requests are), or {"request": {...}} to send one as it is. For YuE2 the
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
            } catch (RuntimeException | LinkageError e) {
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
                    sse.stream("progress", Json.obj("step", s, "steps", n, "secondsPerStep", secs, "stage", stage.get(),
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
            } catch (RuntimeException | LinkageError e) {
                sse.send("error", Json.obj("message", message(e)));
            }
        }
    }

    /** An image's longest side, past which it is scaled down before a vision model sees it (its tokens grow with its area). */
    static final int INSPECT_MAX_SIDE = 1280;
    // A PaddleOCR-VL task prompt, as its model card lists them: "OCR:", "Table Recognition:", ...
    private static final Pattern OCR_TASK = Pattern.compile("[A-Za-z ]{2,32}:");

    /**
     * {"model", "image" (a data: URL or base64 of a PNG or JPEG), "prompt", "schema" (a JSON Schema, optional),
     * "maxTokens", "temperature"}: the preset's defaults for those left out. The reply streams as tokens; with a schema it
     * is constrained to it, and done carries it parsed as "json". An OCR model (PaddleOCR-VL) reads the image with its task
     * prompt ("OCR:" unless the prompt is one); with a schema, another vision model then fills it from the text read, as
     * Fornax's Invoice example does, and the stream says so with a "structuring" status between the two.
     */
    @SuppressWarnings("unchecked")
    private void inspect(HttpExchange ex) throws IOException {
        Map<String, Object> req = body(ex);
        var m = studio.model(Json.str(req, "model", ""), Catalog.Kind.vision);
        var d = m.defaults();
        BufferedImage picture = image(Json.str(req, "image", ""));
        if (picture == null) throw new IllegalArgumentException("no image");
        picture = fit(picture, INSPECT_MAX_SIDE);
        String prompt = Json.str(req, "prompt", "").trim();
        boolean ocr = Json.bool(d, "ocr", false);
        Object rawSchema = req.get("schema");
        Map<String, Object> schema = rawSchema instanceof Map<?, ?> given ? (Map<String, Object>) given
                : rawSchema instanceof String text && !text.isBlank() ? Json.object(text) : null;
        if (schema == null && rawSchema != null && !(rawSchema instanceof String)) {
            throw new IllegalArgumentException("the schema is a JSON object");
        }
        if (prompt.isEmpty() && !ocr) throw new IllegalArgumentException("no prompt: say what to look for");
        long maxTokens = Json.num(req, "maxTokens", Json.num(d, "maxTokens", 1024));
        double temperature = Json.dbl(req, "temperature", Json.dbl(d, "temperature", 0));
        Catalog.Model filler = null;
        if (ocr && schema != null) {
            filler = studio.catalog().models().stream()
                    .filter(v -> v.kind() == Catalog.Kind.vision && v.available() && !Json.bool(v.defaults(), "ocr", false))
                    .findFirst().orElseThrow(() -> new IllegalArgumentException(m.name()
                            + " reads text but doesn't fill a JSON schema well; download Qwen3-VL 4B to fill it from what it reads"));
        }
        // What the model is asked: an OCR model its task, and the text it reads then goes to the filler with the prompt.
        String look = !ocr ? prompt : OCR_TASK.matcher(prompt).matches() ? prompt : "OCR:";
        var images = List.of(LlamaModel.Image.of(picture));
        var ask = Json.obj("messages", List.of(Json.obj("role", "user", "content", List.of(
                        Json.obj("type", "image", "index", 0L), Json.obj("type", "text", "text", look)))),
                "max_tokens", maxTokens, "temperature", temperature);
        if (schema != null && filler == null) ask.put("json_schema", schema);
        int width = picture.getWidth(), height = picture.getHeight();

        try (var sse = new Sse(ex)) {
            try {
                String read = null;
                double readSeconds = 0;
                long t0 = System.nanoTime();
                var first = new AtomicLong();
                java.util.function.Consumer<String> onToken = piece -> {
                    if (first.get() == 0) first.set(System.nanoTime());
                    sse.stream("token", Json.obj("text", piece));
                };
                Map<String, Object> reply;
                try (var lease = leaseSaying(sse, m, Engines.Seer.class)) {
                    sse.send("status", Json.obj("state", "generating"));
                    t0 = System.nanoTime();
                    reply = Json.object(lease.model().look(Json.write(ask), images, onToken));
                }
                if (filler != null) {
                    read = content(reply);
                    readSeconds = (System.nanoTime() - t0) / 1e9;
                    String fill = prompt.isEmpty() || OCR_TASK.matcher(prompt).matches() ? "Fill in the fields from this text." : prompt;
                    var structure = Json.obj("messages", List.of(Json.obj("role", "user", "content",
                                    fill + "\n\nThe text, as read from the image:\n\n" + read)),
                            "json_schema", schema, "max_tokens", maxTokens, "temperature", temperature);
                    sse.send("status", Json.obj("state", "structuring", "model", filler.id(), "read", read, "readSeconds", readSeconds));
                    try (var lease = leaseSaying(sse, filler, Engines.Seer.class)) {
                        first.set(0);
                        t0 = System.nanoTime();
                        reply = Json.object(lease.model().look(Json.write(structure), List.of(), onToken));
                    }
                }
                var usage = reply.get("usage") instanceof Map<?, ?> u ? (Map<String, Object>) u : Map.<String, Object>of();
                String text = content(reply);
                Object parsed = null;
                if (schema != null) {
                    try { parsed = Json.parse(text); } catch (IllegalArgumentException cutOff) { parsed = null; }
                }
                sse.send("done", timing(t0, first.get(), Json.num(usage, "prompt_tokens", 0), Json.num(usage, "completion_tokens", 0),
                        Json.obj("text", text, "json", parsed, "finishReason", reply.get("finish_reason"),
                                "width", width, "height", height, "read", read, "readSeconds", read == null ? null : readSeconds,
                                "filledBy", filler == null ? null : filler.id(), "stub", studio.stubbed(Catalog.Kind.vision))));
            } catch (RuntimeException | LinkageError e) {
                sse.send("error", Json.obj("message", message(e)));
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static String content(Map<String, Object> reply) {
        var message = reply.get("message") instanceof Map<?, ?> msg ? (Map<String, Object>) msg : Map.<String, Object>of();
        return Json.str(message, "content", "");
    }

    /**
     * The image, scaled down (keeping its shape) so its longest side is at most max; as it is when it fits. One with
     * transparency is laid on white: the model sees RGB alone, and a transparent PNG's clear pixels would otherwise be
     * black, hiding black text.
     */
    static BufferedImage fit(BufferedImage img, int max) {
        int w = img.getWidth(), h = img.getHeight();
        boolean alpha = img.getColorModel().hasAlpha();
        if (Math.max(w, h) <= max && !alpha) return img;
        double k = Math.min(1, (double) max / Math.max(w, h));
        int nw = Math.max(1, (int) Math.round(w * k)), nh = Math.max(1, (int) Math.round(h * k));
        var out = new BufferedImage(nw, nh, BufferedImage.TYPE_INT_RGB);
        var g = out.createGraphics();
        g.setColor(java.awt.Color.WHITE);
        g.fillRect(0, 0, nw, nh);
        g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION, java.awt.RenderingHints.VALUE_INTERPOLATION_BICUBIC);
        g.setRenderingHint(java.awt.RenderingHints.KEY_RENDERING, java.awt.RenderingHints.VALUE_RENDER_QUALITY);
        g.drawImage(img, 0, 0, nw, nh, null);
        g.dispose();
        return out;
    }

    /** The most pixels an image sent in a request may have (a 64 MP photo is 8000 × 8000). */
    static final long MAX_IMAGE_PIXELS = 64_000_000L;

    /** An image sent in a request, as a data: URL or bare base64 (null for none). */
    private static BufferedImage image(String data) throws IOException {
        if (data.isBlank()) return null;
        String b64 = data.startsWith("data:") ? data.substring(data.indexOf(',') + 1) : data;
        byte[] bytes;
        try { bytes = java.util.Base64.getMimeDecoder().decode(b64); }
        catch (IllegalArgumentException e) { throw new IllegalArgumentException("the image is not base64"); }
        // Its size from its header first: a small file can claim enough pixels to exhaust the heap when decoded.
        try (var in = ImageIO.createImageInputStream(new java.io.ByteArrayInputStream(bytes))) {
            var readers = ImageIO.getImageReaders(in);
            if (!readers.hasNext()) throw new IllegalArgumentException("the image is not a PNG or JPEG");
            var reader = readers.next();
            try {
                reader.setInput(in);
                if ((long) reader.getWidth(0) * reader.getHeight(0) > MAX_IMAGE_PIXELS)
                    throw new IllegalArgumentException("the image is too large: at most " + MAX_IMAGE_PIXELS / 1_000_000 + " megapixels");
            } finally { reader.dispose(); }
        }
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

    private void pull(HttpExchange ex) throws IOException {
        var b = body(ex);
        String id = Json.str(b, "id", null), token = Json.str(b, "token", null);
        pulls.set(id);   // an unknown id is a 400, before the stream begins
        try (var sse = new Sse(ex)) {
            try { pulls.pull(id, token, sse); }
            catch (RuntimeException e) { sse.send("error", Json.obj("id", id, "message", message(e), "reason", "IO")); }
        }
    }

    // ---- helpers ----

    /** Borrows the model, saying first when it has to be loaded, and how long that took. */
    private <M extends AutoCloseable> org.liberationdata.fornax.ModelPool.Lease<M> leaseSaying(Sse sse, Catalog.Model m, Class<M> type) {
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

    /**
     * What went wrong, in words: the cause's, where Fornax wrapped a native failure in a RuntimeException of its own, or
     * where an engine's class failed to load before (NoClassDefFoundError, whose cause holds the first failure's words).
     */
    private static String message(Throwable e) {
        Throwable t = e;
        while (t.getCause() != null && (t instanceof LinkageError || t.getMessage() == null
                || t.getMessage().equals(t.getCause().toString()))) t = t.getCause();
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
            System.err.println("[fornax-studio] " + ex.getRequestMethod() + " " + ex.getRequestURI() + ": " + e);
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
