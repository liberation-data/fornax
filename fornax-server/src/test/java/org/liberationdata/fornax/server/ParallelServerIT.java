package org.liberationdata.fornax.server;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.Json;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.Socket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** The server with parallel 2: a language model with two slots, against Qwen3 0.6B when it's downloaded. */
class ParallelServerIT {
    static FornaxServer server;
    static final HttpClient HTTP = HttpClient.newHttpClient();
    static final String COUNT = "{\"stream\": %s, \"temperature\": 0, \"max_tokens\": %d, \"messages\": [{\"role\": \"user\", "
            + "\"content\": \"Count from 1 to 500, one number per line, nothing else. /no_think\"}]}";

    @BeforeAll
    static void start() throws Exception {
        assumeTrue(HfCache.find("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf").isPresent(),
                "model not downloaded: hf download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf");
        server = FornaxServer.builder().port(0).model("qwen3-0.6b").contextSize(2048).parallel(2).budgetBytes(8L << 30)
                .log(s -> {}).build().start();
    }

    @AfterAll
    static void stop() { if (server != null) server.close(); }

    static HttpResponse<String> post(String body) throws Exception {
        return HTTP.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + "/v1/chat/completions"))
                .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString());
    }

    /** When each of a streamed reply's chunks arrived, in nanoseconds. */
    static List<Long> chunkTimes(int maxTokens) throws Exception {
        List<Long> times = new ArrayList<>();
        var r = HTTP.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + "/v1/chat/completions"))
                        .POST(HttpRequest.BodyPublishers.ofString(COUNT.formatted(true, maxTokens))).build(),
                HttpResponse.BodyHandlers.ofLines());
        assertEquals(200, r.statusCode());
        r.body().filter(l -> l.startsWith("data: {")).forEach(l -> times.add(System.nanoTime()));
        return times;
    }

    @Test void twoStreamsDecodeTogetherRatherThanOneAfterTheOther() throws Exception {
        var a = CompletableFuture.supplyAsync(() -> { try { return chunkTimes(150); } catch (Exception e) { throw new RuntimeException(e); } });
        var b = CompletableFuture.supplyAsync(() -> { try { return chunkTimes(150); } catch (Exception e) { throw new RuntimeException(e); } });
        List<Long> ta = a.join(), tb = b.join();
        assertTrue(ta.size() > 100 && tb.size() > 100, ta.size() + " and " + tb.size() + " chunks");
        // queued one after the other, one stream would end before the other began
        assertTrue(tb.getFirst() < ta.getLast() && ta.getFirst() < tb.getLast(), "the streams overlap");
    }

    @Test void aClientThatHangsUpStopsOnlyItsOwnRequest() throws Exception {
        // short: on Metal, Qwen3 0.6B's greedy count can end on its own before 300 (#104), which isn't "length"
        var other = CompletableFuture.supplyAsync(() -> { try { return post(COUNT.formatted(false, 60)); } catch (Exception e) { throw new RuntimeException(e); } });
        try (Socket s = new Socket(InetAddress.getLoopbackAddress(), server.port())) {
            byte[] body = COUNT.formatted(true, 2000).getBytes(StandardCharsets.UTF_8);
            OutputStream out = s.getOutputStream();
            out.write(("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:" + server.port() + "\r\nContent-Length: " + body.length
                    + "\r\n\r\n").getBytes(StandardCharsets.UTF_8));
            out.write(body);
            out.flush();
            InputStream in = s.getInputStream();
            int seen = 0;
            for (int c; seen < 10 && (c = in.read()) != -1; ) if (c == '\n') seen++;   // a few chunks in, then hang up
        }
        var r = other.join();
        assertEquals(200, r.statusCode(), r.body());
        Object j = Json.parse(r.body());
        assertEquals("length", Json.get(j, "choices", 0, "finish_reason"), "the other request ran on to its max_tokens");
        assertEquals(60, ((Number) Json.get(j, "usage", "completion_tokens")).intValue());
    }
}
