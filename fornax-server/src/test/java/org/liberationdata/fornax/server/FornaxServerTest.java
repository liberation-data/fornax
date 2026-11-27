package org.liberationdata.fornax.server;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Audio;

import javax.imageio.ImageIO;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.Socket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.liberationdata.fornax.Json.get;

/** The routes, with stand-in engines: what goes to the engine, and what comes back in OpenAI's shapes. */
class FornaxServerTest {
    private final StubEngines engines = new StubEngines();
    private FornaxServer server;
    private final HttpClient http = HttpClient.newHttpClient();

    @BeforeEach void start() throws Exception {
        server = FornaxServer.builder().port(0).engines(engines).log(s -> {}).maxBodyBytes(1 << 20).build().start();
    }

    @AfterEach void stop() { server.close(); }

    HttpResponse<String> post(String path, String body) throws Exception {
        return http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + path))
                .header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(body)).build(),
                HttpResponse.BodyHandlers.ofString());
    }

    HttpResponse<String> getPath(String path) throws Exception {
        return http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + path)).GET().build(),
                HttpResponse.BodyHandlers.ofString());
    }

    static Object json(HttpResponse<String> r) { return org.liberationdata.fornax.Json.parse(r.body()); }

    /** The data: lines of a stream, parsed, up to (not including) [DONE], which must end it. */
    static List<Object> events(String body) {
        List<Object> out = new ArrayList<>();
        boolean done = false;
        for (String block : body.split("\n\n")) {
            if (block.isBlank() || block.startsWith(":")) continue;
            assertTrue(block.startsWith("data: "), block);
            assertFalse(done, "nothing after [DONE]");
            String data = block.substring(6);
            if (data.equals("[DONE]")) { done = true; continue; }
            out.add(org.liberationdata.fornax.Json.parse(data));
        }
        assertTrue(done, "the stream ends with data: [DONE]");
        return out;
    }

    @Test void aChatCompletion() throws Exception {
        var r = post("/v1/chat/completions", "{\"model\": \"qwen3-4b\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"temperature\": 0.2, \"seed\": 7, \"max_tokens\": 30}");
        assertEquals(200, r.statusCode(), r.body());
        assertEquals("application/json", r.headers().firstValue("Content-Type").orElseThrow());
        Object j = json(r);
        assertEquals("chat.completion", get(j, "object"));
        assertTrue(((String) get(j, "id")).startsWith("chatcmpl-"));
        assertEquals("qwen3-4b", get(j, "model"));
        assertEquals("assistant", get(j, "choices", 0, "message", "role"));
        assertEquals("Hello!", get(j, "choices", 0, "message", "content"));
        assertEquals("stop", get(j, "choices", 0, "finish_reason"));
        assertEquals(7L, get(j, "usage", "prompt_tokens"));
        assertEquals(3L, get(j, "usage", "completion_tokens"));
        assertEquals(10L, get(j, "usage", "total_tokens"));
        Object req = org.liberationdata.fornax.Json.parse(engines.requests.get(0));
        assertEquals("Hi", get(req, "messages", 0, "content"));
        assertEquals(0.2, get(req, "temperature"));
        assertEquals(7L, get(req, "seed"));
        assertEquals(30L, get(req, "max_tokens"));
    }

    @Test void withoutAModelTheDefaultAnswers() throws Exception {
        var r = post("/v1/chat/completions", "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}");
        assertEquals(200, r.statusCode(), r.body());
        assertEquals("qwen3-4b", get(json(r), "model"));
    }

    @Test void toolCallsPassThrough() throws Exception {
        engines.pieces = List.of();
        engines.finish = "tool_calls";
        engines.toolCalls = "[{\"id\": \"call_0\", \"type\": \"function\", \"function\": {\"name\": \"weather\", \"arguments\": \"{\\\"city\\\":\\\"Nashville\\\"}\"}}]";
        String tools = "[{\"type\": \"function\", \"function\": {\"name\": \"weather\", \"parameters\": {\"type\": \"object\"}}}]";
        var r = post("/v1/chat/completions", "{\"messages\": [{\"role\": \"user\", \"content\": \"Weather?\"}], \"tools\": " + tools + "}");
        Object j = json(r);
        assertEquals("tool_calls", get(j, "choices", 0, "finish_reason"));
        assertEquals("weather", get(j, "choices", 0, "message", "tool_calls", 0, "function", "name"));
        assertEquals("{\"city\":\"Nashville\"}", get(j, "choices", 0, "message", "tool_calls", 0, "function", "arguments"));
        // the engine's call_0 becomes an id of its own, so the next round of calls doesn't repeat it
        String id = (String) get(j, "choices", 0, "message", "tool_calls", 0, "id");
        assertTrue(id.matches("call_[0-9a-f]{24}"), id);
        String again = (String) get(json(post("/v1/chat/completions", "{\"messages\": [{\"role\": \"user\", \"content\": \"Weather?\"}], \"tools\": " + tools + "}")),
                "choices", 0, "message", "tool_calls", 0, "id");
        assertNotEquals(id, again);
        assertEquals("weather", get(org.liberationdata.fornax.Json.parse(engines.requests.get(0)), "tools", 0, "function", "name"));

        // streamed: the calls come in one delta, indexed, before the finish
        var s = post("/v1/chat/completions", "{\"stream\": true, \"messages\": [{\"role\": \"user\", \"content\": \"Weather?\"}], \"tools\": " + tools + "}");
        List<Object> ev = events(s.body());
        Object calls = ev.stream().filter(e -> get(e, "choices", 0, "delta", "tool_calls") != null).findFirst().orElseThrow();
        assertEquals(0L, get(calls, "choices", 0, "delta", "tool_calls", 0, "index"));
        assertTrue(((String) get(calls, "choices", 0, "delta", "tool_calls", 0, "id")).matches("call_[0-9a-f]{24}"));
        assertEquals("function", get(calls, "choices", 0, "delta", "tool_calls", 0, "type"));
        assertEquals("weather", get(calls, "choices", 0, "delta", "tool_calls", 0, "function", "name"));
        assertEquals("tool_calls", get(ev.get(ev.size() - 1), "choices", 0, "finish_reason"));
    }

    @Test void aStreamedChatIsOpenAIsChunksThenDone() throws Exception {
        var r = post("/v1/chat/completions", "{\"model\": \"qwen3-4b\", \"stream\": true, \"stream_options\": {\"include_usage\": true}, "
                + "\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}");
        assertEquals(200, r.statusCode());
        assertTrue(r.headers().firstValue("Content-Type").orElseThrow().startsWith("text/event-stream"));
        List<Object> ev = events(r.body());
        String id = (String) get(ev.get(0), "id");
        for (Object e : ev) {
            assertEquals(id, get(e, "id"), "one id for the stream");
            assertEquals("chat.completion.chunk", get(e, "object"));
        }
        assertEquals("assistant", get(ev.get(0), "choices", 0, "delta", "role"));
        StringBuilder content = new StringBuilder();
        for (Object e : ev) if (get(e, "choices", 0, "delta", "content") instanceof String c) content.append(c);
        assertEquals("Hello!", content.toString());
        assertEquals("Hel", get(ev.get(1), "choices", 0, "delta", "content"), "each piece as it comes");
        Object finish = ev.get(ev.size() - 2);
        assertEquals("stop", get(finish, "choices", 0, "finish_reason"));
        Object usage = ev.get(ev.size() - 1);
        assertEquals(List.of(), get(usage, "choices"));
        assertEquals(10L, get(usage, "usage", "total_tokens"));
    }

    @Test void stopSequencesEndTheReplyAndStopTheModel() throws Exception {
        engines.pieces = List.of("one, ", "tw", "o, th", "ree");
        var r = post("/v1/chat/completions", "{\"stop\": [\"two\", \"xyz\"], \"messages\": [{\"role\": \"user\", \"content\": \"Count\"}]}");
        assertEquals("one, ", get(json(r), "choices", 0, "message", "content"));
        assertEquals("stop", get(json(r), "choices", 0, "finish_reason"));
        assertEquals(0, engines.stopped.getCount(), "the generation was stopped");

        var s = post("/v1/chat/completions", "{\"stream\": true, \"stop\": \"two\", \"messages\": [{\"role\": \"user\", \"content\": \"Count\"}]}");
        StringBuilder content = new StringBuilder();
        for (Object e : events(s.body())) if (get(e, "choices", 0, "delta", "content") instanceof String c) content.append(c);
        assertEquals("one, ", content.toString());
    }

    @Test void aClientThatGoesAwayStopsTheGeneration() throws Exception {
        Sse.keepAliveMillis = 50;
        try {
            engines.pieces = java.util.Collections.nCopies(400, "word ");
            engines.pieceDelayMillis = 10;
            try (Socket s = new Socket(InetAddress.getLoopbackAddress(), server.port())) {
                byte[] body = "{\"stream\": true, \"messages\": [{\"role\": \"user\", \"content\": \"Go on\"}]}".getBytes(StandardCharsets.UTF_8);
                OutputStream out = s.getOutputStream();
                out.write(("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:" + server.port() + "\r\nContent-Type: application/json\r\n"
                        + "Content-Length: " + body.length + "\r\n\r\n").getBytes(StandardCharsets.UTF_8));
                out.write(body);
                out.flush();
                InputStream in = s.getInputStream();
                byte[] some = new byte[200];
                assertTrue(in.read(some) > 0, "the stream started");
            }   // and the client hangs up, 4 seconds before the reply would end
            assertTrue(engines.stopped.await(3, TimeUnit.SECONDS), "the server stopped the generation");
        } finally {
            Sse.keepAliveMillis = 3_000;
        }
    }

    @Test void aStreamForAModelThatIsntThereIsStillAnErrorStatus() throws Exception {
        var r = post("/v1/chat/completions", "{\"model\": \"missing\", \"stream\": true, \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}");
        assertEquals(404, r.statusCode());
        assertEquals("model_not_found", get(json(r), "error", "code"));
        assertEquals("invalid_request_error", get(json(r), "error", "type"));
    }

    @Test void imagesGoToAVisionModelOnly() throws Exception {
        String png = ChatRequestTest.png(2, 2);
        String msg = "[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"What is it?\"}, "
                + "{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64," + png + "\"}}]}]";
        assertEquals(200, post("/v1/chat/completions", "{\"model\": \"vision\", \"messages\": " + msg + "}").statusCode());
        var r = post("/v1/chat/completions", "{\"model\": \"qwen3-4b\", \"messages\": " + msg + "}");
        assertEquals(400, r.statusCode());
        assertTrue(((String) get(json(r), "error", "message")).contains("takes no images"));
        var url = post("/v1/chat/completions", "{\"model\": \"vision\", \"messages\": [{\"role\": \"user\", \"content\": "
                + "[{\"type\": \"image_url\", \"image_url\": {\"url\": \"https://example.com/a.png\"}}]}]}");
        assertEquals(400, url.statusCode());
        assertTrue(((String) get(json(url), "error", "message")).contains("data: URI"));
    }

    @Test void badRequestsAreOpenAIErrors() throws Exception {
        for (String body : List.of("not json", "[1, 2]", "{}", "{\"messages\": []}", "{\"messages\": [{\"content\": \"no role\"}]}",
                "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"n\": 2}",
                "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"temperature\": 9}",
                "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"max_tokens\": 0}",
                "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}], \"response_format\": {\"type\": \"yaml\"}}")) {
            var r = post("/v1/chat/completions", body);
            assertEquals(400, r.statusCode(), body);
            Object e = json(r);
            assertEquals("invalid_request_error", get(e, "error", "type"), body);
            assertInstanceOf(String.class, get(e, "error", "message"));
            assertTrue(((Map<?, ?>) get(e, "error")).containsKey("code"));
        }
        assertEquals(404, post("/v1/chat/completions", "{\"model\": \"missing\", \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}").statusCode());
        assertEquals(404, post("/v1/completions", "{}").statusCode(), "no legacy completions");
        var wrong = getPath("/v1/chat/completions");
        assertEquals(405, wrong.statusCode());
        assertEquals("POST", wrong.headers().firstValue("Allow").orElseThrow());
        var big = post("/v1/chat/completions", "{\"messages\": \"" + "x".repeat(2 << 20) + "\"}");
        assertEquals(413, big.statusCode());
    }

    @Test void theModels() throws Exception {
        var r = getPath("/v1/models");
        assertEquals(200, r.statusCode());
        Object j = json(r);
        assertEquals("list", get(j, "object"));
        assertEquals("qwen3-4b", get(j, "data", 0, "id"));
        assertEquals("model", get(j, "data", 0, "object"));
        assertEquals("fornax", get(j, "data", 0, "owned_by"));
        assertEquals(true, get(j, "data", 0, "loaded"));
        assertEquals("image", get(j, "data", 1, "kind"));
        assertEquals("flux2-klein", get(json(getPath("/v1/models/flux2-klein")), "id"));
        assertEquals(404, getPath("/v1/models/nope").statusCode());
        assertEquals("ok", get(json(getPath("/health")), "status"));
    }

    @Test void embeddings() throws Exception {
        var r = post("/v1/embeddings", "{\"input\": [\"a\", \"b\"]}");
        assertEquals(200, r.statusCode(), r.body());
        Object j = json(r);
        assertEquals("list", get(j, "object"));
        assertEquals("qwen3-embedding-0.6b", get(j, "model"));
        assertEquals("embedding", get(j, "data", 1, "object"));
        assertEquals(1L, get(j, "data", 1, "index"));
        assertEquals(List.of(0.6, 0.8, 0.0, 1.0), get(j, "data", 1, "embedding"));
        assertNotNull(get(j, "usage", "prompt_tokens"));

        // base64: little-endian float32s, which the openai clients ask for by default
        Object b = json(post("/v1/embeddings", "{\"model\": \"e\", \"input\": \"a\", \"encoding_format\": \"base64\"}"));
        ByteBuffer buf = ByteBuffer.wrap(Base64.getDecoder().decode((String) get(b, "data", 0, "embedding"))).order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(0.6f, buf.getFloat());
        assertEquals(0.8f, buf.getFloat());
        assertEquals("e", engines.models.get(engines.models.size() - 1));

        // dimensions: the first n, normalised again
        Object d = json(post("/v1/embeddings", "{\"input\": \"a\", \"dimensions\": 1}"));
        assertEquals(List.of(1.0), get(d, "data", 0, "embedding"));
        assertEquals(400, post("/v1/embeddings", "{\"input\": \"a\", \"dimensions\": 99}").statusCode());
        assertEquals(400, post("/v1/embeddings", "{\"input\": [[1, 2, 3]]}").statusCode(), "token arrays");
        assertEquals(400, post("/v1/embeddings", "{\"input\": []}").statusCode());
        for (String empty : List.of("\"\"", "[\"a\", \"\"]"))   // OpenAI refuses an empty string, which has no embedding
            assertEquals(400, post("/v1/embeddings", "{\"input\": " + empty + "}").statusCode(), empty);
        assertEquals(400, post("/v1/embeddings", "{\"input\": \"a\", \"encoding_format\": \"int8\"}").statusCode());
    }

    @Test void images() throws Exception {
        var r = post("/v1/images/generations", "{\"model\": \"flux2-klein\", \"prompt\": \"a kiln\", \"size\": \"512x256\", \"n\": 2, \"seed\": 5, \"steps\": 4}");
        assertEquals(200, r.statusCode(), r.body());
        Object j = json(r);
        assertEquals(2, ((List<?>) get(j, "data")).size());
        var img = ImageIO.read(new ByteArrayInputStream(Base64.getDecoder().decode((String) get(j, "data", 0, "b64_json"))));
        assertEquals(512, img.getWidth());
        assertEquals(256, img.getHeight());
        assertEquals("ImageJob[prompt=a kiln, width=512, height=256, steps=4, cfg=null, seed=5, n=2]", engines.requests.get(0));
        assertEquals(400, post("/v1/images/generations", "{\"prompt\": \"a kiln\"}").statusCode(), "no default image model");
        for (String bad : List.of("{\"model\": \"m\"}", "{\"model\": \"m\", \"prompt\": \"p\", \"size\": \"500x500\"}",
                "{\"model\": \"m\", \"prompt\": \"p\", \"size\": \"big\"}", "{\"model\": \"m\", \"prompt\": \"p\", \"n\": 9}",
                "{\"model\": \"m\", \"prompt\": \"p\", \"response_format\": \"url\"}",
                "{\"model\": \"m\", \"prompt\": \"p\", \"n\": 4294967297}", "{\"model\": \"m\", \"prompt\": \"p\", \"n\": -1}"))
            assertEquals(400, post("/v1/images/generations", bad).statusCode(), bad);
    }

    @Test void speech() throws Exception {
        var r = http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + "/v1/audio/speech"))
                .POST(HttpRequest.BodyPublishers.ofString("{\"model\": \"kokoro\", \"input\": \"Hello\", \"voice\": \"alloy\", \"speed\": 1.5}")).build(),
                HttpResponse.BodyHandlers.ofByteArray());
        assertEquals(200, r.statusCode());
        assertEquals("audio/wav", r.headers().firstValue("Content-Type").orElseThrow());
        Audio a = Audio.readWav(r.body());
        assertEquals(24000, a.sampleRate());
        assertEquals("Hello|alloy|1.5", engines.requests.get(0));
        assertEquals(400, post("/v1/audio/speech", "{\"input\": \"Hello\", \"response_format\": \"mp3\"}").statusCode());
        assertEquals(400, post("/v1/audio/speech", "{\"input\": \"\"}").statusCode());
        var tooLong = post("/v1/audio/speech", "{\"input\": \"" + "a".repeat(FornaxServer.MAX_SPEECH_INPUT + 1) + "\"}");
        assertEquals(400, tooLong.statusCode());
        assertEquals("input", get(json(tooLong), "error", "param"));
        assertEquals(200, post("/v1/audio/speech", "{\"input\": \"" + "\u00e9".repeat(FornaxServer.MAX_SPEECH_INPUT) + "\"}").statusCode());
    }

    @Test void transcriptions() throws Exception {
        java.io.File wav = java.io.File.createTempFile("fornax-server", ".wav");
        try {
            Audio.ofMono(new float[1600], 16000).writeWav(wav);
            byte[] audio = java.nio.file.Files.readAllBytes(wav.toPath());
            for (String format : List.of("json", "text", "verbose_json", "srt")) {
                var r = multipart(Map.of("model", "whisper-base.en", "language", "en", "response_format", format), audio);
                assertEquals(200, r.statusCode(), r.body());
                switch (format) {
                    case "json" -> assertEquals("Hello there.", get(json(r), "text"));
                    case "text" -> assertEquals("Hello there.\n", r.body());
                    case "verbose_json" -> {
                        assertEquals("en", get(json(r), "language"));
                        assertEquals("Hello", get(json(r), "words", 0, "word"));
                        assertEquals(0.4, get(json(r), "words", 1, "start"));
                    }
                    default -> assertTrue(r.body().contains("00:00:00,000 --> 00:00:00,800"), r.body());
                }
            }
            assertEquals("16000|en|null", engines.requests.get(0));
            assertEquals(400, multipart(Map.of("model", "m"), "not a wav".getBytes(StandardCharsets.UTF_8)).statusCode());
            assertEquals(400, post("/v1/audio/transcriptions", "{}").statusCode(), "not multipart");
            String boundary = "b0";
            var noFile = http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + "/v1/audio/transcriptions"))
                    .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                    .POST(HttpRequest.BodyPublishers.ofString("--b0\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nm\r\n--b0--\r\n")).build(),
                    HttpResponse.BodyHandlers.ofString());
            assertEquals(400, noFile.statusCode(), noFile.body());
            assertEquals("file", get(json(noFile), "error", "param"));
        } finally {
            assertTrue(wav.delete());
        }
    }

    /** A small request that would make the server allocate or compute without bound is a 400 before any engine runs. */
    @Test void oneRequestCantExhaustTheServer() throws Exception {
        // 2049 one-letter inputs: within the body limit, but a 64 MB body of them would be 16M vectors
        String many = "[" + String.join(",", java.util.Collections.nCopies(FornaxServer.MAX_EMBEDDING_INPUTS + 1, "\"a\"")) + "]";
        var embeddings = post("/v1/embeddings", "{\"input\": " + many + "}");
        assertEquals(400, embeddings.statusCode(), embeddings.body());
        assertEquals("input", get(json(embeddings), "error", "param"));

        // a WAV header's sample rate: 1 Hz upsampled to 16 kHz overflows, and 1000000001 Hz makes a 128 GB filter
        java.io.File wav = java.io.File.createTempFile("fornax-server", ".wav");
        try {
            for (int rate : new int[]{1, 4000, 1_000_000_001}) {
                Audio.ofMono(new float[1600], rate).writeWav(wav);
                var r = multipart(Map.of("model", "whisper-base.en"), java.nio.file.Files.readAllBytes(wav.toPath()));
                assertEquals(400, r.statusCode(), rate + " Hz: " + r.body());
                assertEquals("file", get(json(r), "error", "param"));
            }
        } finally {
            assertTrue(wav.delete());
        }
        assertTrue(engines.requests.isEmpty(), "no engine ran: " + engines.requests);

        // steps, and an image's pixels: a request is one job, not days of the GPU
        for (String bad : List.of("{\"model\": \"m\", \"prompt\": \"p\", \"steps\": 101}",
                "{\"model\": \"m\", \"prompt\": \"p\", \"size\": \"4096x4096\"}"))
            assertEquals(400, post("/v1/images/generations", bad).statusCode(), bad);
        assertEquals(200, post("/v1/images/generations", "{\"model\": \"m\", \"prompt\": \"p\", \"size\": \"4096x1024\", \"steps\": 100}").statusCode());

        // JSON nested beyond any request's needs is refused before it is parsed (the parser, and the engine's, recurse)
        var deep = post("/v1/chat/completions", "[".repeat(100_000) + "]".repeat(100_000));
        assertEquals(400, deep.statusCode(), deep.body());
        String nested = "{\"a\": ".repeat(Json.MAX_DEPTH - 1) + "1" + "}".repeat(Json.MAX_DEPTH - 1);
        assertEquals(400, post("/v1/chat/completions", nested).statusCode(), "deep, but only 400 for having no messages");
        assertTrue(get(json(post("/v1/chat/completions", nested)), "error", "message").toString().contains("messages"));
        assertTrue(get(json(deep), "error", "message").toString().contains("nested"), deep.body());
        assertTrue(get(json(post("/v1/chat/completions", "{\"s\": \"" + "[".repeat(1000) + "\"}")), "error", "message").toString()
                .contains("messages"), "brackets in a string aren't nesting");
    }

    HttpResponse<String> multipart(Map<String, String> fields, byte[] file) throws Exception {
        String boundary = "----fornax" + System.nanoTime();
        var out = new java.io.ByteArrayOutputStream();
        for (var f : fields.entrySet())
            out.write(("--" + boundary + "\r\nContent-Disposition: form-data; name=\"" + f.getKey() + "\"\r\n\r\n" + f.getValue() + "\r\n")
                    .getBytes(StandardCharsets.UTF_8));
        out.write(("--" + boundary + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
                .getBytes(StandardCharsets.UTF_8));
        out.write(file);
        out.write(("\r\n--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
        return http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + "/v1/audio/transcriptions"))
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .POST(HttpRequest.BodyPublishers.ofByteArray(out.toByteArray())).build(), HttpResponse.BodyHandlers.ofString());
    }

    @Test void closingStopsAGenerationInFlight() throws Exception {
        engines.pieces = java.util.Collections.nCopies(1000, "word ");
        engines.pieceDelayMillis = 10;   // 10 seconds, unstopped
        var reply = http.sendAsync(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + "/v1/chat/completions"))
                .POST(HttpRequest.BodyPublishers.ofString("{\"messages\": [{\"role\": \"user\", \"content\": \"Go on\"}]}")).build(),
                HttpResponse.BodyHandlers.ofString());
        for (int i = 0; i < 200 && engines.requests.isEmpty(); i++) Thread.sleep(10);
        assertFalse(engines.requests.isEmpty(), "the generation started");
        server.close();
        assertTrue(engines.stopped.await(3, TimeUnit.SECONDS), "closing stopped it, so unloading needn't wait for its end");
        reply.handle((r, e) -> null).get(5, TimeUnit.SECONDS);
    }

    /** Reads one HTTP/1.1 response (Content-Length or chunked) off a kept-alive connection: its status line and body. */
    static String[] response(java.io.BufferedInputStream in) throws Exception {
        String status = line(in), h;
        int length = -1;
        boolean chunked = false;
        while (!(h = line(in)).isEmpty()) {
            String lower = h.toLowerCase(java.util.Locale.ROOT);
            if (lower.startsWith("content-length:")) length = Integer.parseInt(h.substring(15).strip());
            if (lower.startsWith("transfer-encoding:") && lower.contains("chunked")) chunked = true;
        }
        var body = new java.io.ByteArrayOutputStream();
        if (chunked) {
            for (int n; (n = Integer.parseInt(line(in).strip(), 16)) > 0; line(in)) body.write(in.readNBytes(n));
            line(in);
        } else if (length > 0) body.write(in.readNBytes(length));
        return new String[]{status, body.toString(StandardCharsets.UTF_8)};
    }

    private static String line(InputStream in) throws Exception {
        var b = new StringBuilder();
        for (int c; (c = in.read()) != '\n'; ) { if (c < 0) throw new java.io.EOFException(b.toString()); if (c != '\r') b.append((char) c); }
        return b.toString();
    }

    @Test void aChunkedBodyAndRequestsOnOneKeptAliveConnection() throws Exception {
        try (Socket s = new Socket(InetAddress.getLoopbackAddress(), server.port())) {
            s.setSoTimeout(5000);
            OutputStream out = s.getOutputStream();
            var in = new java.io.BufferedInputStream(s.getInputStream());
            String host = "Host: 127.0.0.1:" + server.port() + "\r\n";
            // a body sent chunked, with no Content-Length, as a streaming client sends one
            String json = "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}";
            String a = json.substring(0, 10), b = json.substring(10);
            out.write(("POST /v1/chat/completions HTTP/1.1\r\n" + host + "Transfer-Encoding: chunked\r\n\r\n"
                    + Integer.toHexString(a.length()) + "\r\n" + a + "\r\n" + Integer.toHexString(b.length()) + "\r\n" + b + "\r\n0\r\n\r\n")
                    .getBytes(StandardCharsets.UTF_8));
            out.flush();
            String[] r = response(in);
            assertTrue(r[0].contains(" 200 "), r[0] + r[1]);
            assertEquals("Hello!", get(org.liberationdata.fornax.Json.parse(r[1]), "choices", 0, "message", "content"));
            // the same connection: an error whose body the server never reads, then a stream, then a plain request
            String unread = "{\"x\": 1}";
            out.write(("POST /v1/completions HTTP/1.1\r\n" + host + "Content-Length: " + unread.length() + "\r\n\r\n" + unread).getBytes(StandardCharsets.UTF_8));
            out.flush();
            assertTrue(response(in)[0].contains(" 404 "));
            String stream = "{\"stream\": true, \"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}";
            out.write(("POST /v1/chat/completions HTTP/1.1\r\n" + host + "Content-Length: " + stream.length() + "\r\n\r\n" + stream).getBytes(StandardCharsets.UTF_8));
            out.flush();
            String[] st = response(in);
            assertTrue(st[0].contains(" 200 "), st[0]);
            assertEquals(5, events(st[1]).size(), st[1]);
            out.write(("GET /v1/models HTTP/1.1\r\n" + host + "\r\n").getBytes(StandardCharsets.UTF_8));
            out.flush();
            String[] m = response(in);
            assertTrue(m[0].contains(" 200 "), m[0]);
            assertEquals("list", get(org.liberationdata.fornax.Json.parse(m[1]), "object"));
        }
    }

    /** A request whose headers say a body of {@code length} is coming, and half of it, so the server holds room for it all. */
    static Socket holding(int port, String path, int length) throws Exception {
        Socket s = new Socket(InetAddress.getLoopbackAddress(), port);
        s.setSoTimeout(5000);
        OutputStream out = s.getOutputStream();
        out.write(("POST " + path + " HTTP/1.1\r\nHost: 127.0.0.1:" + port + "\r\nContent-Length: " + length + "\r\n\r\n")
                .getBytes(StandardCharsets.UTF_8));
        out.write(new byte[length / 2]);
        out.flush();
        return s;
    }

    /** A small request's status, sent until it is {@code want} (the server reads the bodies in hand as it gets them). */
    int statusOnceItIs(int port, int want) throws Exception {
        int got = 0;
        for (int i = 0; i < 300 && got != want; i++) {
            if (i > 0) Thread.sleep(10);
            got = http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + "/v1/embeddings"))
                    .POST(HttpRequest.BodyPublishers.ofString("{\"input\": \"a\"}")).build(), HttpResponse.BodyHandlers.discarding()).statusCode();
        }
        return got;
    }

    @Test void requestBodiesHeldAtOnceAreBounded() throws Exception {
        int mb = 1 << 20;
        try (var small = FornaxServer.builder().port(0).engines(engines).log(s -> {}).maxBodyBytes(mb).maxBufferedBytes(2L * mb).build().start()) {
            int port = small.port();
            Socket a = holding(port, "/v1/chat/completions", mb), b = holding(port, "/v1/chat/completions", mb);
            assertEquals(503, statusOnceItIs(port, 503), "two 1 MB bodies fill 2 MB: the next is refused, not queued");
            var busy = http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + "/v1/chat/completions"))
                    .POST(HttpRequest.BodyPublishers.ofString("{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}")).build(),
                    HttpResponse.BodyHandlers.ofString());
            assertEquals(503, busy.statusCode());
            assertEquals("1", busy.headers().firstValue("Retry-After").orElseThrow());
            assertEquals("server_busy", get(json(busy), "error", "code"));
            assertEquals("server_error", get(json(busy), "error", "type"));

            a.close();   // a client gone mid-body: its room comes back
            assertEquals(200, statusOnceItIs(port, 200));
            OutputStream out = b.getOutputStream();   // a body that ends, but isn't JSON: a 400, and its room back
            out.write(new byte[mb - mb / 2]);
            out.flush();
            assertTrue(response(new java.io.BufferedInputStream(b.getInputStream()))[0].contains(" 400 "));
            b.close();

            // all of it came back, and no more: two bodies fill it again, and the next is refused again
            Socket c = holding(port, "/v1/embeddings", mb), d = holding(port, "/v1/audio/transcriptions", mb);
            assertEquals(503, statusOnceItIs(port, 503));
            c.close();
            d.close();
            assertEquals(200, statusOnceItIs(port, 200));

            // a body sent in many small chunks holds its size, not a KB a chunk (4000 of them would be past 2 MB)
            String json = "{\"input\": \"a\"" + " ".repeat(800_000) + "}";
            var chunked = new StringBuilder("POST /v1/embeddings HTTP/1.1\r\nHost: 127.0.0.1:" + port + "\r\nTransfer-Encoding: chunked\r\n\r\n");
            for (int i = 0; i < json.length(); i += 200) {
                String piece = json.substring(i, Math.min(json.length(), i + 200));
                chunked.append(Integer.toHexString(piece.length())).append("\r\n").append(piece).append("\r\n");
            }
            try (Socket e = new Socket(InetAddress.getLoopbackAddress(), port)) {
                e.setSoTimeout(5000);
                e.getOutputStream().write(chunked.append("0\r\n\r\n").toString().getBytes(StandardCharsets.UTF_8));
                String[] r = response(new java.io.BufferedInputStream(e.getInputStream()));
                assertTrue(r[0].contains(" 200 "), r[0] + r[1]);
            }
            assertThrows(IllegalArgumentException.class, () -> FornaxServer.builder().port(0).engines(engines).maxBodyBytes(2L * mb)
                    .maxBufferedBytes(mb).build(), "a body the limit allows must fit");
        }
    }

    @Test void aModelsOwnToolCallIdIsKept() {
        Map<String, Object> message = org.liberationdata.fornax.server.Json.map(org.liberationdata.fornax.Json.parse(
                "{\"tool_calls\": [{\"id\": \"Ab3dE6gH9\"}, {\"id\": \"\"}, {\"id\": \"call_12\"}]}"));
        FornaxServer.uniqueToolCallIds(message);
        assertEquals("Ab3dE6gH9", get(message, "tool_calls", 0, "id"));
        assertTrue(((String) get(message, "tool_calls", 1, "id")).matches("call_[0-9a-f]{24}"));
        assertTrue(((String) get(message, "tool_calls", 2, "id")).matches("call_[0-9a-f]{24}"));
    }

    @Test void closingUnloadsTheModels() {
        server.close();
        assertTrue(engines.closed);
    }

    @Test void theBuilderRefusesSettingsThatCouldNotWork() {
        var b = FornaxServer.builder();
        assertThrows(IllegalArgumentException.class, () -> b.budgetBytes(0));
        assertThrows(IllegalArgumentException.class, () -> b.contextSize(-1));
        assertThrows(IllegalArgumentException.class, () -> b.maxBodyBytes(0));
        assertThrows(IllegalArgumentException.class, () -> b.maxBufferedBytes(0));
        assertThrows(IllegalArgumentException.class, () -> b.idleTimeout(java.time.Duration.ZERO));
        b.idleTimeout(null).contextSize(0).budgetBytes(1).maxBodyBytes(1);   // never unload; the model's own context
    }
}
