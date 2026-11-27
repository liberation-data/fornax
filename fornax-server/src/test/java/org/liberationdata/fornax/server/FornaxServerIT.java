package org.liberationdata.fornax.server;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.liberationdata.fornax.Json.get;

/** The server with real models: Qwen3 4B for chat, and Qwen3-Embedding 0.6B if it is downloaded too. */
class FornaxServerIT {
    static FornaxServer server;
    static final HttpClient HTTP = HttpClient.newHttpClient();

    @BeforeAll
    static void start() throws Exception {
        assumeTrue(Weights.QWEN3_4B.downloaded(), "model not downloaded: " + Weights.QWEN3_4B.downloadCommand());
        server = FornaxServer.builder().port(0).model("qwen3-4b").budgetBytes(8L << 30).log(s -> {}).build().start();
    }

    @AfterAll
    static void stop() { if (server != null) server.close(); }

    static HttpResponse<String> post(String path, String body) throws Exception {
        return HTTP.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + path))
                .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString());
    }

    @Test void chat() throws Exception {
        var r = post("/v1/chat/completions", "{\"temperature\": 0, \"max_tokens\": 20, \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer with the number only.\"}]}");
        assertEquals(200, r.statusCode(), r.body());
        Object j = org.liberationdata.fornax.Json.parse(r.body());
        assertTrue(((String) get(j, "choices", 0, "message", "content")).contains("4"), r.body());
        assertEquals("stop", get(j, "choices", 0, "finish_reason"));
    }

    @Test void aStreamedChat() throws Exception {
        var r = post("/v1/chat/completions", "{\"stream\": true, \"temperature\": 0, \"max_tokens\": 30, \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 5.\"}]}");
        assertEquals(200, r.statusCode());
        List<Object> ev = FornaxServerTest.events(r.body());
        StringBuilder text = new StringBuilder();
        for (Object e : ev) if (get(e, "choices", 0, "delta", "content") instanceof String c) text.append(c);
        assertTrue(text.toString().contains("5"), text.toString());
        assertTrue(ev.size() > 4, "several chunks");
    }

    @Test void aToolCall() throws Exception {
        var r = post("/v1/chat/completions", "{\"temperature\": 0, \"messages\": [{\"role\": \"user\", \"content\": \"What's the weather in Nashville?\"}], "
                + "\"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"parameters\": {\"type\": \"object\", "
                + "\"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}}]}");
        Object j = org.liberationdata.fornax.Json.parse(r.body());
        assertEquals("tool_calls", get(j, "choices", 0, "finish_reason"), r.body());
        assertEquals("get_weather", get(j, "choices", 0, "message", "tool_calls", 0, "function", "name"));
        assertTrue(((String) get(j, "choices", 0, "message", "tool_calls", 0, "function", "arguments")).contains("Nashville"));
    }

    @Test void embeddings() throws Exception {
        Weights w = Weights.QWEN3_EMBEDDING_0_6B;
        assumeTrue(w.downloaded(), "model not downloaded: " + w.downloadCommand());
        var r = post("/v1/embeddings", "{\"model\": \"qwen3-embedding-0.6b\", \"input\": [\"a brick kiln\", \"a furnace for pottery\", \"tax returns\"]}");
        assertEquals(200, r.statusCode(), r.body());
        Object j = org.liberationdata.fornax.Json.parse(r.body());
        List<?> a = (List<?>) get(j, "data", 0, "embedding"), b = (List<?>) get(j, "data", 1, "embedding"), c = (List<?>) get(j, "data", 2, "embedding");
        assertEquals(1024, a.size());
        assertTrue(dot(a, b) > dot(a, c), "a kiln is nearer a furnace than tax");
    }

    static double dot(List<?> a, List<?> b) {
        double s = 0;
        for (int i = 0; i < a.size(); i++) s += ((Number) a.get(i)).doubleValue() * ((Number) b.get(i)).doubleValue();
        return s;
    }
}
