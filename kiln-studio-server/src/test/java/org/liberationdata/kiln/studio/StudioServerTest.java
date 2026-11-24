package org.liberationdata.kiln.studio;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Every route, over HTTP, with each engine a stub: the wiring the page relies on, with nothing loaded on the GPU. */
class StudioServerTest {
    @TempDir Path dir;
    private Studio studio;
    private StudioServer server;
    private final HttpClient client = HttpClient.newHttpClient();

    @BeforeEach void start() throws Exception {
        Path cache = dir.resolve("hub");
        put(cache, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf");
        put(cache, "black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors");
        put(cache, "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors");
        put(cache, "Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors");
        for (String f : List.of("Qwen3-Embedding-0.6B-Q8_0.gguf", "acestep-v15-turbo-Q8_0.gguf", "vae-BF16.gguf")) {
            put(cache, "Serveurperso/ACE-Step-1.5-GGUF", f);
        }
        studio = new Studio(new Catalog(cache), 1L << 30, null, EnumSet.allOf(Catalog.Kind.class), dir.resolve("outputs"));
        server = new StudioServer(studio, 0);
        server.start();
    }

    @AfterEach void stop() {
        server.stop();
        studio.close();
    }

    private static void put(Path cache, String repo, String file) throws Exception {
        Path p = cache.resolve("models--" + repo.replace("/", "--")).resolve("snapshots/abc").resolve(file);
        Files.createDirectories(p.getParent());
        Files.write(p, new byte[100]);
    }

    private HttpResponse<String> call(String method, String path, String body) throws Exception {
        var b = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + path)).timeout(Duration.ofSeconds(20));
        b = body == null ? b.method(method, HttpRequest.BodyPublishers.noBody()) : b.method(method, HttpRequest.BodyPublishers.ofString(body));
        return client.send(b.build(), HttpResponse.BodyHandlers.ofString());
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> model(String id) throws Exception {
        var status = Json.object(call("GET", "/api/models", null).body());
        return Json.list(status, "models").stream().map(m -> (Map<String, Object>) m)
                .filter(m -> m.get("id").equals(id)).findFirst().orElseThrow();
    }

    /** The data of the last event of that name in a stream. */
    private static Map<String, Object> event(String stream, String name) {
        Map<String, Object> found = null;
        for (String block : stream.split("\n\n")) {
            if (block.startsWith("event: " + name + "\n")) found = Json.object(block.substring(block.indexOf("data: ") + 6));
        }
        assertNotNull(found, "no " + name + " event in:\n" + stream);
        return found;
    }

    @Test void healthAndCatalogue() throws Exception {
        assertEquals(true, Json.object(call("GET", "/api/health", null).body()).get("ok"));
        var qwen = model("chat:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf");
        assertEquals(true, qwen.get("available"));
        assertEquals(false, qwen.get("loaded"));
        assertEquals(false, model("song:acestep-turbo-lm4b").get("available"));
    }

    @Test void loadAndUnload() throws Exception {
        String id = "chat:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
        assertEquals(200, call("POST", "/api/models/load", "{\"id\": \"" + id + "\"}").statusCode());
        assertEquals(true, model(id).get("loaded"));
        assertEquals(100L, Json.object(call("GET", "/api/models", null).body()).get("loadedBytes"));
        assertEquals(true, Json.object(call("POST", "/api/models/unload", "{\"id\": \"" + id + "\"}").body()).get("unloaded"));
        assertEquals(false, model(id).get("loaded"));
    }

    @Test void chatStreamsTokensThenTimings() throws Exception {
        String s = call("POST", "/api/chat", "{\"model\": \"chat:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf\","
                + "\"messages\": [{\"role\": \"user\", \"content\": \"hello there\"}], \"think\": false}").body();
        assertEquals("loading", Json.object(s.substring(s.indexOf("data: ") + 6, s.indexOf("\n\n"))).get("state"));
        assertTrue(s.contains("event: token"));
        var done = event(s, "done");
        assertEquals("(stub) You said: hello there /no_think", done.get("text"));
        assertTrue(((Number) done.get("generatedTokens")).intValue() > 0);
    }

    @Test void toolsDemoAnswers() throws Exception {
        String s = call("POST", "/api/chat/tools", "{\"model\": \"chat:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf\","
                + "\"messages\": [{\"role\": \"user\", \"content\": \"2+2?\"}]}").body();
        assertEquals("stop", event(s, "assistant").get("finishReason"));
        event(s, "done");
    }

    @Test void imageReportsStepsAndServesThePng() throws Exception {
        String s = call("POST", "/api/image", "{\"model\": \"image:flux2-klein\", \"prompt\": \"a neon sign\", \"steps\": 3,"
                + "\"width\": 256, \"height\": 256, \"seed\": 7}").body();
        assertEquals(3L, event(s, "progress").get("step"));
        var done = event(s, "done");
        assertEquals(7L, done.get("seed"));
        var png = client.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + done.get("url"))).build(),
                HttpResponse.BodyHandlers.ofByteArray());
        assertEquals("image/png", png.headers().firstValue("Content-Type").orElse(""));
        assertEquals((byte) 0x89, png.body()[0]);
    }

    @Test void songServesTheWavWithRanges() throws Exception {
        String s = call("POST", "/api/song", "{\"model\": \"song:acestep-turbo\", \"caption\": \"country\", \"lyrics\": \"la\","
                + "\"duration\": 2, \"seed\": 5}").body();
        var done = event(s, "done");
        assertEquals(2.0, done.get("audioSeconds"));
        assertEquals("country", ((Map<?, ?>) done.get("request")).get("caption"));
        var part = client.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + done.get("url")))
                .header("Range", "bytes=0-43").build(), HttpResponse.BodyHandlers.ofByteArray());
        assertEquals(206, part.statusCode());
        assertEquals(44, part.body().length);
        assertEquals("RIFF", new String(part.body(), 0, 4));
    }

    @Test void refusesWhatItCannotDo() throws Exception {
        assertEquals(400, call("POST", "/api/chat", "{\"model\": \"chat:nope\"}").statusCode());
        assertEquals(400, call("POST", "/api/image", "{\"model\": \"song:acestep-turbo\", \"prompt\": \"x\"}").statusCode());
        assertEquals(400, call("POST", "/api/song", "{\"model\": \"song:acestep-turbo-lm4b\", \"caption\": \"x\"}").statusCode());
        assertEquals(400, call("POST", "/api/image", "{\"model\": \"image:flux2-klein\", \"prompt\": \"x\", \"width\": 100}").statusCode());
        assertEquals(405, call("GET", "/api/chat", null).statusCode());
        assertEquals(404, call("GET", "/api/outputs/..%2Fsecret.png", null).statusCode());
        assertEquals(404, call("GET", "/api/models/extra", null).statusCode());
    }
}
