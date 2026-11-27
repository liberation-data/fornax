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
        assertEquals("weather", get(org.liberationdata.fornax.Json.parse(engines.requests.get(0)), "tools", 0, "function", "name"));

        // streamed: the calls come in one delta, indexed, before the finish
        var s = post("/v1/chat/completions", "{\"stream\": true, \"messages\": [{\"role\": \"user\", \"content\": \"Weather?\"}], \"tools\": " + tools + "}");
        List<Object> ev = events(s.body());
        Object calls = ev.stream().filter(e -> get(e, "choices", 0, "delta", "tool_calls") != null).findFirst().orElseThrow();
        assertEquals(0L, get(calls, "choices", 0, "delta", "tool_calls", 0, "index"));
        assertEquals("call_0", get(calls, "choices", 0, "delta", "tool_calls", 0, "id"));
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
        String png = Base64.getEncoder().encodeToString(new byte[]{1, 2, 3});
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
                "{\"model\": \"m\", \"prompt\": \"p\", \"response_format\": \"url\"}"))
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
        } finally {
            assertTrue(wav.delete());
        }
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

    @Test void closingUnloadsTheModels() {
        server.close();
        assertTrue(engines.closed);
    }
}
