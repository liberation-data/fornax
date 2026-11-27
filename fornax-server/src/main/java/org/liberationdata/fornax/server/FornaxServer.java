package org.liberationdata.fornax.server;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.ModelPool;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.PullListener;
import org.liberationdata.fornax.server.Engines.ImageJob;
import org.liberationdata.fornax.server.Engines.Stopper;
import org.liberationdata.fornax.server.Engines.Use;
import org.liberationdata.fornax.whisper.Transcript;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HexFormat;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Consumer;

/**
 * An OpenAI-compatible HTTP server over Fornax's engines, JDK only ({@code com.sun.net.httpserver}), so any OpenAI
 * client, agent framework or tool can use local models with no Java code of its own:
 *
 * <pre>{@code
 * try (var server = FornaxServer.builder().port(54154).model("qwen3-4b").build().start()) {
 *     // POST http://127.0.0.1:54154/v1/chat/completions ...
 *     server.await();
 * }
 * }</pre>
 *
 * <ul>
 *   <li>{@code POST /v1/chat/completions}: messages, tools and {@code tool_calls}, {@code response_format} (a JSON
 *       schema), {@code temperature}, {@code top_p}, {@code seed}, {@code max_tokens}, {@code stop}, images as
 *       {@code image_url} data: URIs for a vision model, and {@code stream: true} as server-sent events</li>
 *   <li>{@code GET /v1/models}: the models it was started with, those loaded, and those downloaded</li>
 *   <li>{@code POST /v1/embeddings}: a string or an array of them, as floats or base64</li>
 *   <li>{@code POST /v1/images/generations}: PNGs as {@code b64_json}; {@code size}, {@code n}, and {@code seed},
 *       {@code steps} and {@code cfg_scale} beyond OpenAI's</li>
 *   <li>{@code POST /v1/audio/speech} (WAV) and {@code POST /v1/audio/transcriptions} (a WAV file, multipart)</li>
 * </ul>
 *
 * A request may name any model Fornax can load: a catalogue alias, an {@code org/name[:TAG]} reference or a path,
 * downloaded the first time and kept loaded in a {@link ModelPool} (least recently used unloaded to stay within the
 * budget, and idle ones after the idle timeout). The models given to {@link Builder#model} are downloaded and loaded
 * at {@link #start}, and the first chat one answers a request that names no model.
 *
 * <p>Who may call it: it listens on 127.0.0.1 unless told otherwise; on loopback, it refuses a request whose
 * {@code Host} isn't its own address (DNS rebinding); it refuses a request from a web page (an {@code Origin}) unless
 * that origin is allowed with {@link Builder#corsOrigin}; and with an {@link Builder#apiKey}, it refuses a request
 * without {@code Authorization: Bearer <key>}. Errors are OpenAI's: {@code {"error": {"message", "type", "param",
 * "code"}}} with the HTTP status.
 */
public final class FornaxServer implements AutoCloseable {
    /** The port it listens on unless told otherwise. */
    public static final int DEFAULT_PORT = 54154;
    /** The embedding model {@code /v1/embeddings} knows by name (Qwen3-Embedding 0.6B), and uses when a request names none. */
    public static final String EMBEDDING_MODEL = PooledEngines.EMBEDDING;

    private final HttpServer http;
    private final ExecutorService executor;
    private final Engines engines;
    private final Guard guard;
    private final Consumer<String> log;
    private final long maxBody;
    private final Runnable warm;
    private final Object stopped = new Object();
    private boolean running;

    private FornaxServer(Builder b) throws IOException {
        InetAddress address = InetAddress.getByName(b.host);
        this.guard = new Guard(address.isLoopbackAddress(), b.cors, b.apiKey);
        this.log = b.log;
        this.maxBody = b.maxBodyBytes;
        if (b.engines != null) {
            this.engines = b.engines;
            this.warm = () -> {};
        } else {
            var pooled = new PooledEngines(b.hub != null ? b.hub : FornaxHub.standard(), new ModelPool(b.budgetBytes, b.idleTimeout),
                    b.models, b.contextSize, b.pulls, b.log);
            this.engines = pooled;
            this.warm = pooled::warm;
        }
        this.http = HttpServer.create(new InetSocketAddress(address, b.port), 0);   // binds now: a port in use fails here
        // Platform threads: a generation is one long native call, which would pin a virtual thread's carrier throughout.
        this.executor = Executors.newCachedThreadPool(r -> {
            Thread t = new Thread(r, "fornax-server-http");
            t.setDaemon(true);
            return t;
        });
        http.setExecutor(executor);
        http.createContext("/", this::handle);
    }

    public static Builder builder() { return new Builder(); }

    /**
     * Downloads and loads the models given to {@link Builder#model}, then answers requests. Returns this.
     *
     * @throws RuntimeException as the engines throw, when a model won't download or load (the server is closed then)
     */
    public FornaxServer start() {
        try { warm.run(); }
        catch (ApiError e) { close(); throw new IllegalStateException(e.getMessage(), e); }
        catch (RuntimeException | Error e) { close(); throw e; }
        http.start();
        synchronized (stopped) { running = true; }
        return this;
    }

    /** The port it listens on (the one chosen, for port 0). */
    public int port() { return http.getAddress().getPort(); }

    /** The API's base URL, as an OpenAI client's base URL: {@code http://127.0.0.1:54154/v1}. */
    public String url() {
        InetAddress a = http.getAddress().getAddress();
        String host = a.isAnyLocalAddress() ? "127.0.0.1" : a instanceof java.net.Inet6Address ? "[" + a.getHostAddress() + "]" : a.getHostAddress();
        return "http://" + host + ":" + port() + "/v1";
    }

    /** Waits until the server is closed (from another thread, or a shutdown hook). */
    public void await() throws InterruptedException {
        synchronized (stopped) { while (running) stopped.wait(); }
    }

    /** Stops answering, waits a moment for requests in flight, and unloads every model. */
    @Override public void close() {
        http.stop(1);
        executor.shutdownNow();
        engines.close();
        synchronized (stopped) { running = false; stopped.notifyAll(); }
    }

    // ---- routing ----

    private interface Route { void handle(HttpExchange ex) throws IOException; }

    private Route route(String method, String path) {
        boolean get = method.equals("GET") || method.equals("HEAD"), post = method.equals("POST");
        if (path.equals("/health")) return get ? ex -> json(ex, 200, Json.obj("status", "ok")) : null;
        if (path.equals("/v1/models")) return get ? this::models : null;
        if (path.startsWith("/v1/models/")) return get ? ex -> model(ex, path.substring("/v1/models/".length())) : null;
        return switch (path) {
            case "/v1/chat/completions" -> post ? this::chat : null;
            case "/v1/embeddings" -> post ? this::embeddings : null;
            case "/v1/images/generations" -> post ? this::images : null;
            case "/v1/audio/speech" -> post ? this::speech : null;
            case "/v1/audio/transcriptions" -> post ? this::transcriptions : null;
            default -> throw new ApiError(404, "invalid_request_error", "no such endpoint: " + method + " " + path, null, "not_found");
        };
    }

    private void handle(HttpExchange ex) {
        long t0 = System.nanoTime();
        String method = ex.getRequestMethod(), path = ex.getRequestURI().getPath();
        int[] status = {0};
        try {
            var h = ex.getRequestHeaders();
            boolean preflight = method.equals("OPTIONS") && h.containsKey("Origin") && h.containsKey("Access-Control-Request-Method");
            ApiError refused = guard.refusal(h, port(), preflight);
            if (refused != null) {
                status[0] = error(ex, refused);
                return;
            }
            guard.cors(h, ex.getResponseHeaders(), preflight);
            if (preflight) {
                ex.sendResponseHeaders(status[0] = 204, -1);
                return;
            }
            Route r = route(method, path);
            if (r == null) {
                ex.getResponseHeaders().set("Allow", path.equals("/health") || path.startsWith("/v1/models") ? "GET, HEAD" : "POST");
                status[0] = error(ex, new ApiError(405, "invalid_request_error", method + " isn't allowed on " + path, null, "method_not_allowed"));
                return;
            }
            r.handle(ex);
            status[0] = ex.getResponseCode();
        } catch (ApiError e) {
            status[0] = error(ex, e);
        } catch (CancellationException e) {
            status[0] = 499;   // the client went away mid-stream: there is no one to tell
        } catch (RuntimeException | IOException | Error e) {
            status[0] = error(ex, ApiError.server(e.getMessage() != null ? e.getMessage() : e.toString()));
            if (!(e instanceof IOException)) log.accept("error in " + method + " " + path + ": " + e);
        } finally {
            ex.close();
            log.accept(String.format(Locale.ROOT, "%s %s %d %.2fs", method, path, status[0], (System.nanoTime() - t0) / 1e9));
        }
    }

    // ---- /v1/chat/completions ----

    private void chat(HttpExchange ex) throws IOException {
        ChatRequest req = ChatRequest.of(Json.object(body(ex)), engines.defaultModel(Use.CHAT));
        String id = "chatcmpl-" + HexFormat.of().formatHex(randomBytes(12));
        long created = System.currentTimeMillis() / 1000;
        if (req.stream()) { stream(ex, req, id, created); return; }

        Stopper stopper = new Stopper();
        Stops stops = new Stops(req.stops(), piece -> {});
        String res = engines.chat(req.model(), req.engineJson(), req.images(),
                req.stops().isEmpty() ? null : piece -> { if (stops.accept(piece)) stopper.stop(); }, stopper);
        Map<String, Object> r = Json.map(org.liberationdata.fornax.Json.parse(res));
        Map<String, Object> message = Json.map(r.get("message"));
        Object finish = r.get("finish_reason");
        if (stops.stopped()) {
            message.put("content", stops.text());
            finish = "stop";
        }
        json(ex, 200, Json.obj("id", id, "object", "chat.completion", "created", created, "model", req.model(),
                "choices", List.of(Json.obj("index", 0, "message", message, "logprobs", null, "finish_reason", finish)),
                "usage", usage(r.get("usage"))));
    }

    /** {@code stream: true}: the content as it is generated, then the tool calls and the finish reason, then [DONE]. */
    private void stream(HttpExchange ex, ChatRequest req, String id, long created) {
        engines.checkChat(req.model(), req.images());   // a name it won't serve is still an error status, not a stream
        Stopper stopper = new Stopper();
        Sse sse = new Sse(ex, stopper::stop);
        boolean[] roleSent = {false};
        Consumer<Map<String, Object>> delta = d -> {
            if (!roleSent[0]) {
                roleSent[0] = true;
                sse.data(chunk(id, created, req.model(), Json.obj("role", "assistant", "content", ""), null));
            }
            sse.data(chunk(id, created, req.model(), d, null));
        };
        try {
            Stops stops = new Stops(req.stops(), piece -> delta.accept(Json.obj("content", piece)));
            String res = engines.chat(req.model(), req.engineJson(), req.images(), piece -> { if (stops.accept(piece)) stopper.stop(); }, stopper);
            stops.finish();
            if (stopper.requested() && !stops.stopped()) throw new CancellationException("the client has gone");
            Map<String, Object> r = Json.map(org.liberationdata.fornax.Json.parse(res));
            Map<String, Object> message = Json.map(r.get("message"));
            if (message.get("reasoning_content") instanceof String s && !s.isEmpty()) delta.accept(Json.obj("reasoning_content", s));
            List<Object> calls = Json.list(message.get("tool_calls"));
            if (calls != null && !calls.isEmpty()) {
                List<Object> indexed = new ArrayList<>();
                for (int i = 0; i < calls.size(); i++) {
                    Map<String, Object> c = new java.util.LinkedHashMap<>();
                    c.put("index", i);
                    c.putAll(Json.map(calls.get(i)));
                    indexed.add(c);
                }
                delta.accept(Json.obj("tool_calls", indexed));
            }
            if (!roleSent[0]) delta.accept(Json.obj("content", ""));
            sse.data(chunk(id, created, req.model(), Json.obj(), stops.stopped() ? "stop" : r.get("finish_reason")));
            if (req.includeUsage())
                sse.data(Json.obj("id", id, "object", "chat.completion.chunk", "created", created, "model", req.model(),
                        "choices", List.of(), "usage", usage(r.get("usage"))));
            sse.done();
        } catch (ApiError | CancellationException e) {
            if (!sse.abandon()) streamError(sse, e);
            else throw e;
        } catch (RuntimeException e) {
            if (!sse.abandon()) streamError(sse, e);
            else throw e;
        } finally {
            sse.close();
        }
    }

    /** A failure after the stream has started: an error event (if the client is still there), as OpenAI sends one. */
    private void streamError(Sse sse, RuntimeException e) {
        if (e instanceof CancellationException) throw e;
        ApiError a = e instanceof ApiError x ? x : ApiError.server(e.getMessage() != null ? e.getMessage() : e.toString());
        try { sse.data(a.body()); } catch (CancellationException gone) { /* nobody to tell */ }
        log.accept("error mid-stream: " + e);
    }

    private static Map<String, Object> chunk(String id, long created, String model, Map<String, Object> delta, Object finish) {
        return Json.obj("id", id, "object", "chat.completion.chunk", "created", created, "model", model,
                "choices", List.of(Json.obj("index", 0, "delta", delta, "logprobs", null, "finish_reason", finish)));
    }

    private static Map<String, Object> usage(Object u) {
        long p = org.liberationdata.fornax.Json.get(u, "prompt_tokens") instanceof Number n ? n.longValue() : 0;
        long c = org.liberationdata.fornax.Json.get(u, "completion_tokens") instanceof Number n ? n.longValue() : 0;
        return Json.obj("prompt_tokens", p, "completion_tokens", c, "total_tokens", p + c);
    }

    // ---- /v1/models ----

    private void models(HttpExchange ex) throws IOException {
        List<Object> data = new ArrayList<>();
        for (var m : engines.models()) data.add(modelJson(m));
        json(ex, 200, Json.obj("object", "list", "data", data));
    }

    private void model(HttpExchange ex, String id) throws IOException {
        for (var m : engines.models()) if (m.id().equals(id)) { json(ex, 200, modelJson(m)); return; }
        throw ApiError.noSuchModel("no model " + id + " here (GET /v1/models lists them)");
    }

    private static Map<String, Object> modelJson(Engines.Listed m) {
        return Json.obj("id", m.id(), "object", "model", "created", 0, "owned_by", "fornax", "kind", m.kind(), "loaded", m.loaded());
    }

    // ---- /v1/embeddings ----

    private void embeddings(HttpExchange ex) throws IOException {
        Map<String, Object> body = Json.object(body(ex));
        String model = ChatRequest.model(body, engines.defaultModel(Use.EMBED));
        List<String> texts = new ArrayList<>();
        Object input = body.get("input");
        if (input instanceof String s) texts.add(s);
        else if (Json.list(input) != null) {
            for (Object o : Json.list(input)) {
                if (!(o instanceof String s)) throw ApiError.badRequest("input must be a string or an array of strings (token arrays aren't taken)", "input");
                texts.add(s);
            }
        }
        if (texts.isEmpty()) throw ApiError.badRequest("input must be a string or a non-empty array of strings", "input");
        Object format = body.getOrDefault("encoding_format", "float");
        if (!"float".equals(format) && !"base64".equals(format)) throw ApiError.badRequest("encoding_format must be float or base64", "encoding_format");
        Object dims = body.get("dimensions");
        int dimensions = dims == null ? 0 : (int) Math.min(Integer.MAX_VALUE, ChatRequest.whole(dims, "dimensions", 1));

        float[][] vectors = engines.embed(model, texts);
        List<Object> data = new ArrayList<>();
        for (int i = 0; i < vectors.length; i++) {
            float[] v = dimensions == 0 ? vectors[i] : truncate(vectors[i], dimensions);
            data.add(Json.obj("object", "embedding", "index", i, "embedding", "base64".equals(format) ? base64(v) : v));
        }
        json(ex, 200, Json.obj("object", "list", "data", data, "model", model, "usage", Json.obj("prompt_tokens", 0, "total_tokens", 0)));
    }

    /** The first n dimensions, normalised again (Qwen3-Embedding is trained for this: Matryoshka representation). */
    static float[] truncate(float[] v, int n) {
        if (n > v.length) throw ApiError.badRequest("dimensions can be at most " + v.length + " for this model", "dimensions");
        float[] t = java.util.Arrays.copyOf(v, n);
        double norm = 0;
        for (float x : t) norm += x * x;
        norm = Math.sqrt(norm);
        if (norm > 0) for (int i = 0; i < n; i++) t[i] = (float) (t[i] / norm);
        return t;
    }

    /** Little-endian float32s, base64, as OpenAI's {@code encoding_format: "base64"} gives them. */
    static String base64(float[] v) {
        ByteBuffer b = ByteBuffer.allocate(v.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) b.putFloat(x);
        return Base64.getEncoder().encodeToString(b.array());
    }

    // ---- /v1/images/generations ----

    private void images(HttpExchange ex) throws IOException {
        Map<String, Object> body = Json.object(body(ex));
        ImageJob job = imageJob(body);
        String model = ChatRequest.model(body, engines.defaultModel(Use.IMAGE));
        List<Object> data = new ArrayList<>();
        for (BufferedImage img : engines.images(model, job)) data.add(Json.obj("b64_json", Base64.getEncoder().encodeToString(png(img))));
        json(ex, 200, Json.obj("created", System.currentTimeMillis() / 1000, "data", data, "output_format", "png",
                "size", job.width() + "x" + job.height()));
    }

    /** The image request, checked: a prompt, n from 1 to 8, a size in multiples of 16, and Fornax's own steps and cfg_scale. */
    static ImageJob imageJob(Map<String, Object> body) {
        if (!(body.get("prompt") instanceof String prompt) || prompt.isBlank()) throw ApiError.badRequest("prompt is required", "prompt");
        Object format = body.get("response_format");
        if (format != null && !"b64_json".equals(format))
            throw ApiError.badRequest("response_format must be b64_json: the server keeps no files to link to", "response_format");
        Object output = body.get("output_format");
        if (output != null && !"png".equals(output)) throw ApiError.badRequest("output_format must be png", "output_format");
        int n = body.get("n") == null ? 1 : (int) ChatRequest.whole(body.get("n"), "n", 1);
        if (n > 8) throw ApiError.badRequest("n can be at most 8", "n");
        Object size = body.get("size");
        int w = 1024, h = 1024;
        if (size != null && !"auto".equals(size)) {
            String[] wh = size instanceof String s ? s.split("x") : new String[0];
            try {
                if (wh.length != 2) throw new NumberFormatException();
                w = Integer.parseInt(wh[0]);
                h = Integer.parseInt(wh[1]);
            } catch (NumberFormatException e) { throw ApiError.badRequest("size must be WIDTHxHEIGHT, such as 1024x1024", "size"); }
            if (w < 64 || h < 64 || w > 4096 || h > 4096 || w % 16 != 0 || h % 16 != 0)
                throw ApiError.badRequest("size must be from 64 to 4096 a side, in multiples of 16", "size");
        }
        int steps = body.get("steps") == null ? 20 : (int) Math.min(1000, ChatRequest.whole(body.get("steps"), "steps", 1));
        Object cfg = body.get("cfg_scale");
        if (cfg != null && !(cfg instanceof Number c && c.doubleValue() >= 0)) throw ApiError.badRequest("cfg_scale must be 0 or more", "cfg_scale");
        long seed = body.get("seed") == null ? ThreadLocalRandom.current().nextInt(Integer.MAX_VALUE) : ChatRequest.whole(body.get("seed"), "seed", 0);
        return new ImageJob(prompt, w, h, steps, cfg == null ? null : ((Number) cfg).doubleValue(), seed, n);
    }

    private static byte[] png(BufferedImage img) {
        var out = new ByteArrayOutputStream();
        try { ImageIO.write(img, "png", out); } catch (IOException e) { throw new UncheckedIOException(e); }
        return out.toByteArray();
    }

    // ---- /v1/audio ----

    private void speech(HttpExchange ex) throws IOException {
        Map<String, Object> body = Json.object(body(ex));
        if (!(body.get("input") instanceof String input) || input.isBlank()) throw ApiError.badRequest("input is required", "input");
        String model = ChatRequest.model(body, engines.defaultModel(Use.SPEECH));
        Object format = body.get("response_format");
        if (format != null && !"wav".equals(format)) throw ApiError.badRequest("response_format must be wav", "response_format");
        Object voice = body.get("voice");
        if (voice != null && !(voice instanceof String)) throw ApiError.badRequest("voice must be a string", "voice");
        Object speed = body.get("speed");
        if (speed != null && !(speed instanceof Number s && s.doubleValue() >= 0.25 && s.doubleValue() <= 4))
            throw ApiError.badRequest("speed must be from 0.25 to 4", "speed");
        Audio audio = engines.speak(model, input, (String) voice, speed == null ? 0 : ((Number) speed).floatValue());
        send(ex, 200, "audio/wav", audio.wav());
    }

    private void transcriptions(HttpExchange ex) throws IOException {
        var parts = Multipart.parse(ex.getRequestHeaders().getFirst("Content-Type"), bytes(ex));
        var file = parts.get("file");
        if (file == null) throw ApiError.badRequest("file is required (a WAV)", "file");
        String model = parts.containsKey("model") && !parts.get("model").text().isBlank() ? parts.get("model").text().strip()
                : engines.defaultModel(Use.TRANSCRIPTION);
        if (model == null) throw ApiError.badRequest("model is required", "model");
        String format = parts.containsKey("response_format") ? parts.get("response_format").text().strip() : "json";
        if (!Set.of("json", "text", "verbose_json", "srt", "vtt").contains(format))
            throw ApiError.badRequest("response_format must be json, text, verbose_json, srt or vtt", "response_format");
        String language = parts.containsKey("language") && !parts.get("language").text().isBlank() ? parts.get("language").text().strip() : null;
        String prompt = parts.containsKey("prompt") && !parts.get("prompt").text().isBlank() ? parts.get("prompt").text() : null;
        Audio audio;
        try { audio = Audio.readWav(file.bytes()); }
        catch (IOException | RuntimeException e) {
            throw ApiError.badRequest("file must be a WAV (" + e.getMessage() + "); convert other formats first, e.g. ffmpeg -i in.mp3 in.wav", "file");
        }
        Transcript t = engines.transcribe(model, audio, language, prompt);
        String text = t.text().strip();
        switch (format) {
            case "text" -> send(ex, 200, "text/plain; charset=utf-8", (text + "\n").getBytes(StandardCharsets.UTF_8));
            case "srt" -> send(ex, 200, "text/plain; charset=utf-8", t.srt().getBytes(StandardCharsets.UTF_8));
            case "vtt" -> send(ex, 200, "text/vtt; charset=utf-8", t.vtt().getBytes(StandardCharsets.UTF_8));
            case "verbose_json" -> {
                List<Object> segments = new ArrayList<>(), words = new ArrayList<>();
                for (int i = 0; i < t.segments().size(); i++) {
                    var s = t.segments().get(i);
                    segments.add(Json.obj("id", i, "start", s.start(), "end", s.end(), "text", s.text()));
                }
                for (var w : t.words()) words.add(Json.obj("word", w.text().strip(), "start", w.start(), "end", w.end()));
                json(ex, 200, Json.obj("task", "transcribe", "language", t.language(), "duration", audio.seconds(), "text", text,
                        "segments", segments, "words", words));
            }
            default -> json(ex, 200, Json.obj("text", text));
        }
    }

    // ---- requests and replies ----

    private String body(HttpExchange ex) throws IOException { return new String(bytes(ex), StandardCharsets.UTF_8); }

    private byte[] bytes(HttpExchange ex) throws IOException {
        String length = ex.getRequestHeaders().getFirst("Content-Length");
        if (length != null) {
            try {
                if (Long.parseLong(length.strip()) > maxBody) throw tooLarge();
            } catch (NumberFormatException e) { throw ApiError.badRequest("a bad Content-Length", null); }
        }
        try (InputStream in = ex.getRequestBody()) {
            byte[] b = in.readNBytes((int) Math.min(Integer.MAX_VALUE - 8, maxBody + 1));
            if (b.length > maxBody) throw tooLarge();
            return b;
        }
    }

    private ApiError tooLarge() {
        return new ApiError(413, "invalid_request_error", "the request is larger than " + maxBody / (1 << 20) + " MB", null, "request_too_large");
    }

    private static void json(HttpExchange ex, int status, Object body) throws IOException {
        send(ex, status, "application/json", Json.write(body).getBytes(StandardCharsets.UTF_8));
    }

    private static void send(HttpExchange ex, int status, String type, byte[] body) throws IOException {
        ex.getResponseHeaders().set("Content-Type", type);
        boolean head = ex.getRequestMethod().equals("HEAD");
        ex.sendResponseHeaders(status, head ? -1 : body.length);
        if (!head) try (OutputStream out = ex.getResponseBody()) { out.write(body); }
    }

    /** Sends the error, unless a reply has already started; returns the status. */
    private static int error(HttpExchange ex, ApiError e) {
        try {
            if (ex.getResponseCode() == -1) json(ex, e.status, e.body());
        } catch (IOException ignored) { /* the client has gone */ }
        return e.status;
    }

    private static byte[] randomBytes(int n) {
        byte[] b = new byte[n];
        ThreadLocalRandom.current().nextBytes(b);
        return b;
    }

    /** How the server is set up. Every setting has a default: {@code FornaxServer.builder().build().start()} serves on 127.0.0.1:54154. */
    public static final class Builder {
        private String host = "127.0.0.1";
        private int port = DEFAULT_PORT;
        private String apiKey;
        private final Set<String> cors = new LinkedHashSet<>();
        private long budgetBytes = 24L << 30;
        private Duration idleTimeout = Duration.ofMinutes(15);
        private int contextSize = 8192;
        private FornaxHub hub;
        private PullListener pulls = PullListener.logging();
        private Consumer<String> log = logger();
        private final List<String> models = new ArrayList<>();
        private long maxBodyBytes = 64L << 20;
        private Engines engines;

        private Builder() {}

        /** The address to listen on (default 127.0.0.1). Another than loopback lets other machines call it: set an API key. */
        public Builder host(String host) { this.host = Objects.requireNonNull(host, "host"); return this; }

        /** The port (default {@value #DEFAULT_PORT}; 0 picks a free one, which {@link #port()} gives). */
        public Builder port(int port) {
            if (port < 0 || port > 65535) throw new IllegalArgumentException("port must be from 0 to 65535, not " + port);
            this.port = port;
            return this;
        }

        /** The key every request must carry as {@code Authorization: Bearer <key>}; null or blank for none (the default). */
        public Builder apiKey(String key) { this.apiKey = key; return this; }

        /**
         * An origin a web page may call from ({@code https://app.example}, or {@code http://localhost:5173}), which then
         * gets CORS headers; {@code *} allows any. Without one, a request from a web page is refused.
         */
        public Builder corsOrigin(String origin) {
            String o = Objects.requireNonNull(origin, "origin").strip();
            if (o.endsWith("/")) o = o.substring(0, o.length() - 1);
            cors.add(o);
            return this;
        }

        /** The model pool's budget: loaded models' sizes, in bytes (default 24 GB). */
        public Builder budgetBytes(long bytes) { this.budgetBytes = bytes; return this; }

        /** How long an unused model stays loaded (default 15 minutes; null keeps it until the budget needs the room). */
        public Builder idleTimeout(Duration timeout) { this.idleTimeout = timeout; return this; }

        /** Context size in tokens for language and embedding models (default 8192; 0 for the model's own). */
        public Builder contextSize(int tokens) { this.contextSize = tokens; return this; }

        /** Where models are found and downloaded (default {@link FornaxHub#standard()}). */
        public Builder hub(FornaxHub hub) { this.hub = hub; return this; }

        /** Download progress, for a model downloaded at start or on a request's first use (default: logged). */
        public Builder pullListener(PullListener listener) { this.pulls = Objects.requireNonNull(listener); return this; }

        /** A line per request, and models loading (default: {@code System.Logger} {@code org.liberationdata.fornax.server}, INFO). */
        public Builder log(Consumer<String> log) { this.log = Objects.requireNonNull(log); return this; }

        /**
         * A model to download and load at start: a catalogue alias ({@code qwen3-4b}), an {@code org/name[:TAG]}
         * reference or a path. The first chat model answers requests that name none. A reference or path whose name
         * says "embed" is loaded for embeddings; any other one for chat.
         */
        public Builder model(String model) { models.add(Objects.requireNonNull(model, "model").strip()); return this; }

        /** The largest request body taken (default 64 MB: an image or a WAV file in it). */
        public Builder maxBodyBytes(long bytes) { this.maxBodyBytes = bytes; return this; }

        /** A test's engines instead of Fornax's. */
        Builder engines(Engines engines) { this.engines = engines; return this; }

        /** Binds the port (failing here if it is in use); {@link FornaxServer#start} downloads, loads and serves. */
        public FornaxServer build() throws IOException { return new FornaxServer(this); }

        private static Consumer<String> logger() {
            System.Logger l = System.getLogger("org.liberationdata.fornax.server");
            return s -> l.log(System.Logger.Level.INFO, s);
        }
    }
}
