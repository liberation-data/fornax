package org.liberationdata.fornax.server;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A {@code POST /v1/chat/completions} body, checked, and turned into the request {@code LlamaModel.chatJson} takes:
 * the same messages and tools (llama.cpp parses them as llama-server does), with OpenAI's {@code response_format},
 * {@code max_completion_tokens}, a {@code tool_choice} naming one function and {@code chat_template_kwargs} mapped to
 * what the engine calls them. {@code stop} sequences are applied by the server ({@link Stops}).
 *
 * @param model        the model asked for
 * @param engineJson   the request for chatJson
 * @param stream       whether to stream the reply as server-sent events
 * @param includeUsage {@code stream_options.include_usage}: a last chunk with the token counts
 * @param stops        the stop sequences, possibly none
 * @param images       whether a message has an image (the model must take them)
 */
record ChatRequest(String model, String engineJson, boolean stream, boolean includeUsage, List<String> stops, boolean images) {

    /** Message content part types the engine reads. */
    private static final Set<String> PARTS = Set.of("text", "image_url");
    /**
     * The most pixels a request's images may have in all (8192 x 8192): the engine decodes each to RGB before scaling it
     * down, and a small compressed image can say it is enormous.
     */
    static final long MAX_PIXELS = 1L << 26;

    /** {@code body} as a request for chatJson; {@code defaultModel} (or null) when it names none. */
    static ChatRequest of(Map<String, Object> body, String defaultModel) {
        String model = model(body, defaultModel);
        List<Object> messages = Json.list(body.get("messages"));
        if (messages == null || messages.isEmpty()) throw ApiError.badRequest("messages must be a non-empty array", "messages");
        long[] pixels = {0};
        for (Object m : messages) {
            Map<String, Object> msg = Json.map(m);
            if (msg == null || !(msg.get("role") instanceof String)) throw ApiError.badRequest("each message needs a role", "messages");
            images(msg, pixels);
        }
        boolean images = pixels[0] > 0;
        Object n = body.get("n");
        if (n != null && !(n instanceof Long l && l == 1)) throw ApiError.badRequest("n must be 1: Fornax makes one choice a request", "n");

        Map<String, Object> req = new LinkedHashMap<>();
        req.put("messages", messages);
        List<Object> tools = Json.list(body.get("tools"));
        Object choice = body.get("tool_choice");
        if (choice instanceof String s) {
            if (!Set.of("auto", "none", "required").contains(s))
                throw ApiError.badRequest("tool_choice must be auto, none, required or a function", "tool_choice");
            req.put("tool_choice", s);
        } else if (choice != null) {
            // {"type": "function", "function": {"name": "f"}}: that tool alone, which must be called
            String name = org.liberationdata.fornax.Json.get(choice, "function", "name") instanceof String s ? s : null;
            if (name == null) throw ApiError.badRequest("tool_choice must be auto, none, required or a function", "tool_choice");
            List<Object> named = tools == null ? List.of()
                    : tools.stream().filter(t -> name.equals(org.liberationdata.fornax.Json.get(t, "function", "name"))).toList();
            if (named.isEmpty()) throw ApiError.badRequest("tool_choice names " + name + ", which isn't among the tools", "tool_choice");
            tools = named;
            req.put("tool_choice", "required");
        }
        if (tools != null && !tools.isEmpty()) req.put("tools", tools);
        if (body.get("parallel_tool_calls") instanceof Boolean b) req.put("parallel_tool_calls", b);
        Object schema = schema(body.get("response_format"));
        if (schema != null) req.put("json_schema", schema);
        number(body, "temperature", 0, 2, req, "temperature");
        number(body, "top_p", 0, 1, req, "top_p");
        number(body, "min_p", 0, 1, req, "min_p");
        Object topK = body.get("top_k");
        if (topK != null) req.put("top_k", whole(topK, "top_k", 0));
        Object seed = body.get("seed");
        if (seed != null) req.put("seed", whole(seed, "seed", Long.MIN_VALUE));
        Object max = body.get("max_completion_tokens") != null ? body.get("max_completion_tokens") : body.get("max_tokens");
        if (max != null) req.put("max_tokens", whole(max, body.get("max_completion_tokens") != null ? "max_completion_tokens" : "max_tokens", 1));
        Object thinking = body.get("enable_thinking");
        if (thinking == null) thinking = org.liberationdata.fornax.Json.get(body.get("chat_template_kwargs"), "enable_thinking");
        if (thinking instanceof Boolean b) req.put("enable_thinking", b);

        boolean stream = body.get("stream") instanceof Boolean b && b;
        boolean usage = Boolean.TRUE.equals(org.liberationdata.fornax.Json.get(body.get("stream_options"), "include_usage"));
        return new ChatRequest(model, Json.write(req), stream, usage, stops(body.get("stop")), images);
    }

    /** The model the body names, else the default, else a 400. */
    static String model(Map<String, Object> body, String defaultModel) {
        Object m = body.get("model");
        if (m != null && !(m instanceof String)) throw ApiError.badRequest("model must be a string", "model");
        String model = m == null || ((String) m).isBlank() ? defaultModel : ((String) m).strip();
        if (model == null) throw ApiError.badRequest("model is required (this server has no default model)", "model");
        return model;
    }

    /** Checks the message's content parts are ones the engine reads, adding its images' pixels to {@code pixels[0]}. */
    private static void images(Map<String, Object> msg, long[] pixels) {
        List<Object> parts = Json.list(msg.get("content"));
        if (parts == null) return;
        for (Object p : parts) {
            Object type = org.liberationdata.fornax.Json.get(p, "type");
            if (!(type instanceof String t) || !PARTS.contains(t))
                throw ApiError.badRequest("content parts must be text or image_url" + (type instanceof String t ? ", not " + t : ""), "messages");
            if (t.equals("image_url")) {
                Object url = org.liberationdata.fornax.Json.get(p, "image_url", "url");
                if (!(url instanceof String u) || !u.startsWith("data:") || !u.contains(";base64,"))
                    throw ApiError.badRequest("image_url must be a base64 data: URI (data:image/png;base64,...); the server doesn't fetch URLs", "messages");
                pixels[0] += pixels(u.substring(u.indexOf(";base64,") + 8));
                if (pixels[0] > MAX_PIXELS)
                    throw ApiError.badRequest("the images can have at most " + MAX_PIXELS / 1_000_000 + " megapixels in all; scale them down", "messages");
            }
        }
    }

    /** An image_url's width times height, read from its header (it isn't decoded here); a 400 for one that isn't an image. */
    static long pixels(String base64) {
        byte[] bytes;
        try { bytes = Base64.getDecoder().decode(base64.replaceAll("\\s", "").replace('-', '+').replace('_', '/')); }
        catch (IllegalArgumentException e) { throw ApiError.badRequest("an image_url's data isn't valid base64", "messages"); }
        try (var in = ImageIO.createImageInputStream(new ByteArrayInputStream(bytes))) {
            var readers = in == null ? null : ImageIO.getImageReaders(in);
            if (readers != null && readers.hasNext()) {
                ImageReader r = readers.next();
                try {
                    r.setInput(in, true, true);
                    long px = (long) r.getWidth(0) * r.getHeight(0);
                    if (px > 0) return px;
                } finally { r.dispose(); }
            }
        } catch (IOException | RuntimeException e) { /* not an image it can read: said below */ }
        throw ApiError.badRequest("an image_url isn't a PNG, JPEG, GIF or BMP image", "messages");
    }

    /** response_format as the engine's json_schema, or null for plain text. */
    private static Object schema(Object format) {
        if (format == null) return null;
        Object type = org.liberationdata.fornax.Json.get(format, "type");
        if ("text".equals(type)) return null;
        if ("json_object".equals(type)) return Json.obj("type", "object");
        if ("json_schema".equals(type)) {
            Object schema = org.liberationdata.fornax.Json.get(format, "json_schema", "schema");
            if (Json.map(schema) == null) throw ApiError.badRequest("response_format.json_schema.schema must be a JSON schema object", "response_format");
            return schema;
        }
        throw ApiError.badRequest("response_format.type must be text, json_object or json_schema", "response_format");
    }

    private static List<String> stops(Object stop) {
        if (stop == null) return List.of();
        if (stop instanceof String s) return List.of(s);
        List<Object> l = Json.list(stop);
        if (l == null) throw ApiError.badRequest("stop must be a string or an array of strings", "stop");
        List<String> out = new ArrayList<>();
        for (Object o : l) {
            if (!(o instanceof String s)) throw ApiError.badRequest("stop must be a string or an array of strings", "stop");
            out.add(s);
        }
        return out;
    }

    private static void number(Map<String, Object> body, String key, double min, double max, Map<String, Object> req, String to) {
        Object v = body.get(key);
        if (v == null) return;
        if (!(v instanceof Number n) || !(n.doubleValue() >= min && n.doubleValue() <= max))
            throw ApiError.badRequest(key + " must be a number from " + fmt(min) + " to " + fmt(max), key);
        req.put(to, n);
    }

    static long whole(Object v, String key, long min) {
        if (!(v instanceof Long l) || l < min)
            throw ApiError.badRequest(key + " must be a whole number" + (min == 0 ? " (0 or more)" : min == 1 ? " (1 or more)" : ""), key);
        return l;
    }

    private static String fmt(double d) { return d == Math.rint(d) ? String.valueOf((long) d) : String.valueOf(d); }
}
