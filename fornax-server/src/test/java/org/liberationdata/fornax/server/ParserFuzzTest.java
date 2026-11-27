package org.liberationdata.fornax.server;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Audio;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The hand-written parsers a request goes through, fed random and adversarial input: each must end in a result or a
 * 400, never another exception, an Error (a stack overflow, out of memory), a hang, or time that grows with the
 * square of the input. Seeded, so a failure repeats; each test is bounded in time.
 */
class ParserFuzzTest {
    static final int ROUNDS = 3000;
    static final Duration EACH = Duration.ofSeconds(4);

    /** Runs f, which must return or throw a 400 ApiError; anything else fails, naming the input. */
    static void survives(String what, Object input, Runnable f) {
        try { f.run(); }
        catch (ApiError e) { assertEquals(400, e.status, what + ": " + e.getMessage() + " for " + show(input)); }
        catch (Throwable t) { throw new AssertionError(what + " threw " + t + " for " + show(input), t); }
    }

    static String show(Object input) {
        String s = input instanceof byte[] b ? new String(b, StandardCharsets.ISO_8859_1) : String.valueOf(input);
        return s.length() > 300 ? s.substring(0, 300) + "... (" + s.length() + " chars)" : s;
    }

    /** Asserts make finishes within the time, so a quadratic case (seconds at this size) fails. */
    static <T> T fast(Duration limit, Supplier<T> make) {
        return assertTimeoutPreemptively(limit, make::get);
    }

    // ---- mutations ----

    static final String[] JSON_TOKENS = {"{", "}", "[", "]", ":", ",", "\"", "\\", "\\u", "\\ud800", "\\udc00", "\\u00e9",
            "\\u+041", "\\u-041", "\\uzzzz", "\\u12", "\\x", "\\\"", "true", "fals", "null", "nul", "0", "-0", "01", "1e999",
            "-1e-999", "1.5e+3", "9223372036854775808", "-9223372036854775809", "1.", ".1", "+1", "--1", "1e", "NaN",
            "Infinity", " ", "\t", "\r\n", "\n", "\u0000", "\u00e9", "\ud83d\ude00", "\ud800", "\udfff", "\"a\"", "\"messages\"",
            "\"role\"", "\"content\"", "\"stop\"", "\"model\"", "\"stream\"", "\"n\"", "\"tools\"", "\"image_url\""};

    static String mutate(Random r, String s) {
        StringBuilder b = new StringBuilder(s);
        int edits = 1 + r.nextInt(4);
        for (int k = 0; k < edits; k++) {
            int at = b.isEmpty() ? 0 : r.nextInt(b.length() + 1);
            switch (r.nextInt(6)) {
                case 0 -> b.setLength(at);                                                     // truncated
                case 1 -> b.insert(at, JSON_TOKENS[r.nextInt(JSON_TOKENS.length)]);
                case 2 -> { if (at < b.length()) b.deleteCharAt(at); }
                case 3 -> { if (at < b.length()) b.setCharAt(at, (char) r.nextInt(0x10000)); }   // invalid surrogates among them
                case 4 -> { int end = Math.min(b.length(), at + r.nextInt(16)); b.insert(at, b.substring(at, end)); }
                default -> b.insert(at, (char) (32 + r.nextInt(95)));
            }
        }
        return b.toString();
    }

    static byte[] mutate(Random r, byte[] in) {
        var b = new ByteArrayOutputStream();
        b.writeBytes(in);
        byte[] a = b.toByteArray();
        int edits = 1 + r.nextInt(4);
        for (int k = 0; k < edits; k++) {
            if (a.length == 0) { a = new byte[]{(byte) r.nextInt()}; continue; }
            int at = r.nextInt(a.length);
            switch (r.nextInt(5)) {
                case 0 -> a = java.util.Arrays.copyOf(a, at);                                    // truncated
                case 1 -> a[at] = (byte) r.nextInt();
                case 2 -> a[at] = (byte) (r.nextBoolean() ? 0xff : 0x7f);                        // a size made huge
                case 3 -> { a[at] = 0; if (at + 1 < a.length) a[at + 1] = 0; }                  // a size made zero
                default -> {
                    byte[] c = new byte[a.length + 4], four = new byte[4];
                    r.nextBytes(four);
                    System.arraycopy(a, 0, c, 0, at);
                    System.arraycopy(four, 0, c, at, 4);
                    System.arraycopy(a, at, c, at + 4, a.length - at);
                    a = c;
                }
            }
        }
        return a;
    }

    /** A random JSON value made of the keys and shapes requests use, of the given depth at most. */
    static Object value(Random r, int depth) {
        int pick = r.nextInt(depth <= 0 ? 6 : 9);
        return switch (pick) {
            case 0 -> null;
            case 1 -> r.nextBoolean();
            case 2 -> r.nextBoolean() ? (long) (r.nextInt(5) - 1) : r.nextLong();
            case 3 -> r.nextBoolean() ? r.nextDouble() * 4 - 1 : (r.nextBoolean() ? 1e308 * 10 : -0.0);
            case 4, 5 -> List.of("auto", "none", "required", "text", "json_object", "json_schema", "image_url", "function",
                    "user", "", "1024x1024", "b64_json", "png", "wav", "data:image/png;base64,iVBORw0KGgo=", "\ud800",
                    "qwen3-4b", "missing", "vision").get(r.nextInt(19));
            case 6, 7 -> {
                Map<String, Object> m = new java.util.LinkedHashMap<>();
                int n = r.nextInt(4);
                for (int i = 0; i < n; i++) m.put(KEYS[r.nextInt(KEYS.length)], value(r, depth - 1));
                yield m;
            }
            default -> {
                List<Object> l = new ArrayList<>();
                int n = r.nextInt(4);
                for (int i = 0; i < n; i++) l.add(value(r, depth - 1));
                yield l;
            }
        };
    }

    static final String[] KEYS = {"model", "messages", "role", "content", "type", "text", "image_url", "url", "tools",
            "function", "name", "tool_choice", "parallel_tool_calls", "response_format", "json_schema", "schema",
            "temperature", "top_p", "min_p", "top_k", "seed", "max_tokens", "max_completion_tokens", "enable_thinking",
            "chat_template_kwargs", "stream", "stream_options", "include_usage", "stop", "n", "input", "encoding_format",
            "dimensions", "prompt", "size", "steps", "cfg_scale", "output_format", "voice", "speed"};

    /** A plausible request body, so mutations reach past the first check. */
    static Map<String, Object> request(Random r) {
        Map<String, Object> m = new java.util.LinkedHashMap<>();
        List<Object> messages = new ArrayList<>();
        messages.add(Json.obj("role", "user", "content", r.nextBoolean() ? "Hi" : value(r, 3)));
        if (r.nextBoolean()) messages.add(value(r, 3));
        m.put("messages", messages);
        int extra = r.nextInt(6);
        for (int i = 0; i < extra; i++) m.put(KEYS[r.nextInt(KEYS.length)], value(r, 3));
        return m;
    }

    // ---- JSON ----

    @Test void jsonBodies() {
        Random r = new Random(1);
        assertTimeoutPreemptively(EACH, () -> {
            for (int i = 0; i < ROUNDS; i++) {
                String s = mutate(r, Json.write(request(r)));
                survives("Json.object", s, () -> Json.object(s));
            }
            for (int i = 0; i < ROUNDS; i++) {   // random tokens, no structure at all
                StringBuilder b = new StringBuilder();
                for (int k = r.nextInt(30); k > 0; k--) b.append(JSON_TOKENS[r.nextInt(JSON_TOKENS.length)]);
                survives("Json.object", b, () -> Json.object(b.toString()));
            }
        });
    }

    @Test void jsonEdgeCases() {
        String deepest = "{\"a\":".repeat(Json.MAX_DEPTH - 1) + "[]" + "}".repeat(Json.MAX_DEPTH - 1);
        assertNotNull(Json.object(deepest), "just under the limit parses");
        String over = "[".repeat(Json.MAX_DEPTH + 1) + "]".repeat(Json.MAX_DEPTH + 1);
        assertEquals(400, assertThrows(ApiError.class, () -> Json.object("{\"a\":" + over + "}")).status);
        // closing brackets first don't buy deeper nesting after them
        String unbalanced = "{\"a\":1}" + "]".repeat(1000) + "[".repeat(100_000);
        survives("Json.object", unbalanced, () -> Json.object(unbalanced));
        survives("Json.object", "]".repeat(1000) + "[".repeat(100_000), () -> Json.object("]".repeat(1000) + "[".repeat(100_000)));

        assertEquals("\ud800x", Json.object("{\"a\": \"\\ud800x\"}").get("a"), "a lone surrogate is a char, as JSON allows");
        for (String bad : List.of("\\u+041", "\\u-041", "\\u 041", "\\u12", "\\uzzzz", "\\u0x41"))
            assertEquals(400, assertThrows(ApiError.class, () -> Json.object("{\"a\": \"" + bad + "\"}"), bad).status, bad);
        assertEquals("A\u00e9", Json.object("{\"a\": \"\\u0041\\u00E9\"}").get("a"));
        for (String bad : List.of("+1", "01", "1.", ".1", "1e", "-", "1e+", "0x10", "1.5.5"))
            assertEquals(400, assertThrows(ApiError.class, () -> Json.object("{\"a\": " + bad + "}"), bad).status, bad);
        assertEquals(Double.POSITIVE_INFINITY, Json.object("{\"a\": 1e999}").get("a"));
        assertEquals(-0.0, Json.object("{\"a\": -0.0}").get("a"));
        assertEquals(0L, Json.object("{\"a\": -0}").get("a"));
    }

    /** Big inputs that would take seconds if any step were quadratic, or overflow a stack if it recursed per element. */
    @Test void largeJsonIsLinear() {
        String digits = "1".repeat(4 << 20);
        assertInstanceOf(Double.class, fast(Duration.ofSeconds(3), () -> Json.object("{\"a\": " + digits + "}").get("a")));
        String bad = "1.".repeat(2 << 20);
        ApiError e = fast(Duration.ofSeconds(3), () -> assertThrows(ApiError.class, () -> Json.object("{\"a\": " + bad + "}")));
        assertTrue(e.getMessage().length() < 1000, "a bad number isn't echoed whole: " + e.getMessage().length() + " chars");
        String escapes = "\\u00e9".repeat(1 << 20);
        assertEquals(1 << 20, fast(Duration.ofSeconds(3), () -> ((String) Json.object("{\"a\": \"" + escapes + "\"}").get("a")).length()));
        String wide = "{\"a\": [" + "[],".repeat(1 << 20) + "[]]}";
        assertNotNull(fast(Duration.ofSeconds(3), () -> Json.object(wide)));
        String keys = "{" + "\"k\":1,".repeat(1 << 20) + "\"k\":2}";
        assertEquals(2L, fast(Duration.ofSeconds(3), () -> Json.object(keys).get("k")));
        String unterminated = "{\"a\": \"" + "x".repeat(4 << 20);
        assertThrows(ApiError.class, () -> fast(Duration.ofSeconds(3), () -> Json.object(unterminated)));
    }

    // ---- the request mapping ----

    @Test void chatRequests() {
        Random r = new Random(2);
        assertTimeoutPreemptively(EACH, () -> {
            for (int i = 0; i < ROUNDS; i++) {
                Map<String, Object> body = request(r);
                survives("ChatRequest.of", body, () -> {
                    ChatRequest req = ChatRequest.of(body, r.nextBoolean() ? "qwen3-4b" : null);
                    org.liberationdata.fornax.Json.parse(req.engineJson());   // what goes to the engine is JSON
                    new Stops(req.stops(), s -> {}).accept("some reply text");
                });
                survives("imageJob", body, () -> FornaxServer.imageJob(body));
            }
        });
    }

    /**
     * Random bodies over HTTP, to every route, streamed or not: each answer is a reply or a 4xx in OpenAI's shape, never
     * a 500 (an exception the server didn't expect), and a stream ends with [DONE].
     */
    @Test void requestsOverHttp() throws Exception {
        Random r = new Random(7);
        var http = java.net.http.HttpClient.newHttpClient();
        try (var server = FornaxServer.builder().port(0).engines(new StubEngines()).log(s -> {}).maxBodyBytes(1 << 20).build().start()) {
            assertTimeoutPreemptively(Duration.ofSeconds(10), () -> {
                String[] paths = {"/v1/chat/completions", "/v1/embeddings", "/v1/images/generations", "/v1/audio/speech"};
                for (int i = 0; i < 400; i++) {
                    Map<String, Object> m = request(r);
                    if (r.nextBoolean()) m.put("stream", true);
                    if (r.nextInt(3) == 0) m.put("input", r.nextBoolean() ? "Hi" : value(r, 2));
                    if (r.nextInt(3) == 0) m.put("prompt", "a cat");
                    String body = r.nextInt(3) == 0 ? mutate(r, Json.write(m)) : Json.write(m);
                    String path = paths[r.nextInt(paths.length)];
                    var res = http.send(java.net.http.HttpRequest.newBuilder(java.net.URI.create("http://127.0.0.1:" + server.port() + path))
                            .header("Content-Type", "application/json").POST(java.net.http.HttpRequest.BodyPublishers.ofString(body)).build(),
                            java.net.http.HttpResponse.BodyHandlers.ofByteArray());
                    int status = res.statusCode();
                    String text = new String(res.body(), StandardCharsets.UTF_8);
                    assertTrue(status == 200 || status == 400 || status == 404, path + " " + status + " " + text + " for " + show(body));
                    String type = res.headers().firstValue("Content-Type").orElse("");
                    if (type.startsWith("text/event-stream")) FornaxServerTest.events(text);
                    else if (type.startsWith("application/json")) org.liberationdata.fornax.Json.parse(text);
                    else assertEquals("audio/wav", type, path);
                }
            });
        }
    }

    // ---- multipart ----

    static byte[] multipart(String boundary, String crlf, byte[] file) {
        var b = new ByteArrayOutputStream();
        b.writeBytes(("--" + boundary + crlf + "Content-Disposition: form-data; name=\"model\"" + crlf + crlf + "whisper-base.en" + crlf
                + "--" + boundary + crlf + "Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"" + crlf
                + "Content-Type: audio/wav" + crlf + crlf).getBytes(StandardCharsets.UTF_8));
        b.writeBytes(file);
        b.writeBytes((crlf + "--" + boundary + "--" + crlf).getBytes(StandardCharsets.UTF_8));
        return b.toByteArray();
    }

    @Test void multipartBodies() {
        Random r = new Random(3);
        String[] types = {"multipart/form-data; boundary=XyZ", "multipart/form-data; boundary=\"XyZ\"", "MULTIPART/FORM-DATA;BOUNDARY=XyZ",
                "multipart/form-data; boundary=", "multipart/form-data; boundary=\"\"", "multipart/form-data; boundary=\"XyZ",
                "multipart/form-data; charset=utf-8; boundary=XyZ", "multipart/form-data", "text/plain; boundary=XyZ", ""};
        String[] crlfs = {"\r\n", "\n", "\r", "\n\r"};
        assertTimeoutPreemptively(EACH, () -> {
            for (int i = 0; i < ROUNDS; i++) {
                byte[] file = new byte[r.nextInt(64)];
                r.nextBytes(file);
                byte[] body = multipart("XyZ", crlfs[r.nextInt(crlfs.length)], file);
                byte[] fuzzed = r.nextInt(4) == 0 ? body : mutate(r, body);
                String type = types[r.nextInt(types.length)];
                survives("Multipart.parse", fuzzed, () -> Multipart.parse(type, fuzzed));
            }
        });
        assertArrayEquals(new byte[]{1, 2}, Multipart.parse(types[0], multipart("XyZ", "\r\n", new byte[]{1, 2})).get("file").bytes());
    }

    @Test void largeMultipartIsLinear() {
        int size = 16 << 20;
        // many delimiters, each a part with no headers' end; many parts; headers' ends with no delimiter after them
        for (String unit : List.of("--XyZ\r\n", "--XyZ\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\nx\r\n", "\r\n\r\n--Xy",
                "--XyZ\r\nContent-Disposition: form-data; name=\"" + ";name=\"".repeat(50) + "\r\n\r\n\r\n",
                "--XyZ\r\nContent-Disposition: " + ";".repeat(4000) + "\r\n\r\n\r\n")) {
            byte[] body = unit.repeat(size / unit.length()).getBytes(StandardCharsets.ISO_8859_1);
            fast(Duration.ofSeconds(3), () -> {
                survives("Multipart.parse", unit, () -> Multipart.parse("multipart/form-data; boundary=XyZ", body));
                return null;
            });
        }
    }

    // ---- data: URIs (images) ----

    static byte[] image(String format) throws IOException {
        var img = new BufferedImage(5, 3, BufferedImage.TYPE_INT_RGB);
        var out = new ByteArrayOutputStream();
        assertTrue(ImageIO.write(img, format, out), format);
        return out.toByteArray();
    }

    @Test void imageUris() throws IOException {
        Random r = new Random(4);
        List<byte[]> images = List.of(image("png"), image("jpeg"), image("gif"), image("bmp"), image("tiff"));
        for (byte[] img : images) assertEquals(15, ChatRequest.pixels(Base64.getEncoder().encodeToString(img)));
        assertTimeoutPreemptively(Duration.ofSeconds(8), () -> {
            for (int i = 0; i < ROUNDS; i++) {
                byte[] img = mutate(r, images.get(r.nextInt(images.size())));
                String b64 = switch (r.nextInt(5)) {
                    case 0 -> Base64.getUrlEncoder().encodeToString(img);
                    case 1 -> Base64.getMimeEncoder().encodeToString(img);
                    case 2 -> mutate(r, Base64.getEncoder().encodeToString(img));   // not base64 any more, perhaps
                    default -> Base64.getEncoder().encodeToString(img);
                };
                survives("ChatRequest.pixels", b64, () -> ChatRequest.pixels(b64));
                if (i % 10 == 0) {
                    var body = Json.object("{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\", \"image_url\": "
                            + "{\"url\": " + org.liberationdata.fornax.Json.quote("data:image/png;base64," + b64) + "}}]}]}");
                    survives("ChatRequest.of", b64, () -> ChatRequest.of(body, "vision"));
                }
            }
        });
    }

    @Test void largeImageUrisAreLinear() {
        String spaces = "iVBO" + " \n".repeat(8 << 20) + "Rw==";
        fast(Duration.ofSeconds(3), () -> { survives("pixels", "spaces", () -> ChatRequest.pixels(spaces)); return null; });
        byte[] junk = new byte[24 << 20];
        new Random(5).nextBytes(junk);
        String b64 = Base64.getEncoder().encodeToString(junk);
        fast(Duration.ofSeconds(3), () -> assertThrows(ApiError.class, () -> ChatRequest.pixels(b64)));
    }

    // ---- WAV ----

    static byte[] wav(int format, int channels, int rate, int bits, int dataBytes) {
        ByteBuffer b = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN);
        b.put("RIFF".getBytes(StandardCharsets.ISO_8859_1)).putInt(36 + dataBytes).put("WAVE".getBytes(StandardCharsets.ISO_8859_1))
         .put("fmt ".getBytes(StandardCharsets.ISO_8859_1)).putInt(16).putShort((short) format).putShort((short) channels)
         .putInt(rate).putInt(rate * channels * bits / 8).putShort((short) (channels * bits / 8)).putShort((short) bits)
         .put("data".getBytes(StandardCharsets.ISO_8859_1)).putInt(dataBytes);
        return b.array();
    }

    @Test void wavFiles() {
        Random r = new Random(6);
        int[][] formats = {{1, 1, 16000, 16}, {1, 2, 44100, 8}, {1, 2, 48000, 24}, {1, 1, 8000, 32}, {3, 2, 16000, 32}, {3, 1, 16000, 64},
                {0xfffe, 2, 16000, 16}};
        assertTimeoutPreemptively(EACH, () -> {
            for (int i = 0; i < ROUNDS; i++) {
                int[] f = formats[r.nextInt(formats.length)];
                byte[] w = wav(f[0], f[1], f[2], f[3], r.nextInt(256));
                byte[] fuzzed = mutate(r, w);
                try { Audio.readWav(fuzzed); }
                catch (IOException expected) { /* a 400 */ }
                catch (Throwable t) { throw new AssertionError("readWav threw " + t + " for " + show(fuzzed), t); }
            }
        });
        // a header whose sizes say more than there is: what there is, or an IOException, at once
        byte[] liar = wav(1, 65535, Integer.MAX_VALUE, 32, 16);
        ByteBuffer.wrap(liar).order(ByteOrder.LITTLE_ENDIAN).putInt(40, Integer.MAX_VALUE).putInt(16, Integer.MAX_VALUE);
        fast(Duration.ofSeconds(1), () -> {
            try { return Audio.readWav(liar); } catch (IOException e) { return null; }
        });
    }
}
