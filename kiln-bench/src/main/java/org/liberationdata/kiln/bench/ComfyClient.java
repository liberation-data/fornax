package org.liberationdata.kiln.bench;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Duration;

/** Minimal ComfyUI HTTP API client: queue an API-format prompt, wait, fetch its first output file. */
final class ComfyClient {
    private static final ObjectMapper JSON = new ObjectMapper();
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    private final String base;

    ComfyClient(String base) { this.base = base; }

    boolean available() {
        try { return get("/system_stats") != null; } catch (Exception e) { return false; }
    }

    /** Unloads every model so the next run is cold and the GPU memory is free for Kiln. */
    void free() throws Exception { post("/free", "{\"unload_models\": true, \"free_memory\": true}"); }

    /** Queues the prompt and blocks until it finishes. Returns the history entry. */
    JsonNode run(ObjectNode prompt) throws Exception {
        ObjectNode body = JSON.createObjectNode();
        body.set("prompt", prompt);
        String id = JSON.readTree(post("/prompt", JSON.writeValueAsString(body))).path("prompt_id").asText();
        while (true) {
            JsonNode h = JSON.readTree(get("/history/" + id)).path(id);
            if (!h.isMissingNode() && h.path("status").has("completed")) {
                if (!"success".equals(h.path("status").path("status_str").asText()))
                    throw new IllegalStateException("ComfyUI run failed: " + h.path("status"));
                return h;
            }
            Thread.sleep(250);
        }
    }

    /** Downloads the first saved image/audio/video file of a finished run. */
    void download(JsonNode history, Path to) throws Exception {
        for (JsonNode out : history.path("outputs")) {
            for (String key : new String[] {"images", "audio", "videos"}) {
                for (JsonNode item : out.path(key)) {
                    if (!"output".equals(item.path("type").asText())) continue;
                    String q = "/view?filename=" + enc(item.path("filename").asText()) + "&subfolder="
                            + enc(item.path("subfolder").asText()) + "&type=output";
                    http.send(HttpRequest.newBuilder(URI.create(base + q)).build(), HttpResponse.BodyHandlers.ofFile(to));
                    return;
                }
            }
        }
        throw new IllegalStateException("no output file in " + history.path("outputs"));
    }

    private static String enc(String s) { return URLEncoder.encode(s, StandardCharsets.UTF_8); }

    private String get(String path) throws Exception {
        return http.send(HttpRequest.newBuilder(URI.create(base + path)).build(), HttpResponse.BodyHandlers.ofString()).body();
    }

    private String post(String path, String json) throws Exception {
        var r = http.send(HttpRequest.newBuilder(URI.create(base + path)).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json)).build(), HttpResponse.BodyHandlers.ofString());
        if (r.statusCode() >= 400) throw new IllegalStateException("ComfyUI " + path + " -> " + r.statusCode() + ": " + r.body());
        return r.body();
    }
}
