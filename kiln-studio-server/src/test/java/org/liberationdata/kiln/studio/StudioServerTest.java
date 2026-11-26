package org.liberationdata.kiln.studio;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.kiln.llama.LlamaModel;

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
import static org.junit.jupiter.api.Assumptions.assumeTrue;

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
        put(cache, "Serveurperso/YuE2-GGUF", "YuE2-3B-Q8_0.gguf");
        put(cache, "Serveurperso/YuE2-GGUF", "YuE2-Vae-F32.gguf");
        for (String f : List.of("diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
                "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors", "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
                "vae/ltx-2.5-audio-vae-bf16.safetensors", "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")) {
            put(cache, "Lightricks/LTX-2.5", f);
        }
        for (var w : List.of(LlamaModel.Weights.QWEN3_VL_4B, LlamaModel.Weights.PADDLEOCR_VL)) {
            put(cache, w.repo, w.modelFile);
            put(cache, w.repo, w.mmprojFile);
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
        var yue2 = model("song:yue2-3b");
        assertEquals("yue2.cpp", yue2.get("engine"));
        assertTrue(yue2.get("license").toString().startsWith("CC BY-NC 4.0"));
        assertNull(model("song:acestep-turbo").get("license"));
        assertEquals(true, model("video:ltx-2.5-distilled").get("available"));
        assertEquals(false, model("video:ltx-2.5-dev").get("available"));
    }

    @Test void loadAndUnload() throws Exception {
        String id = "chat:Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
        assertEquals(200, call("POST", "/api/models/load", "{\"id\": \"" + id + "\"}").statusCode());
        assertEquals(true, model(id).get("loaded"));
        // A stub loads nothing, so it holds nothing of the budget.
        assertEquals(0L, Json.object(call("GET", "/api/models", null).body()).get("loadedBytes"));
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

    @Test void yue2SongTakesItsOwnRequestAndKeepsTheScore() throws Exception {
        String s = call("POST", "/api/song", "{\"model\": \"song:yue2-3b\", \"caption\": \"country rock\", \"lyrics\": \"la\","
                + "\"duration\": 2, \"bpm\": 120, \"keyscale\": \"D major\", \"seed\": 5}").body();
        var done = event(s, "done");
        var request = (Map<?, ?>) done.get("request");
        // yue2-rusty.json's shape: a style, no caption, BPM or key.
        assertEquals("country rock", request.get("style"));
        assertEquals(5L, request.get("lm_seed"));
        assertFalse(request.containsKey("caption") || request.containsKey("bpm") || request.containsKey("keyscale"));
        assertTrue(done.get("score").toString().startsWith("X:1"));
        assertTrue(done.get("license").toString().startsWith("CC BY-NC"));
        assertEquals(2.0, done.get("audioSeconds"));
        String name = done.get("url").toString().replaceAll(".*/|\\.wav$", "");
        assertTrue(Files.readString(dir.resolve("outputs").resolve(name + ".abc")).startsWith("X:1"));
    }

    private static boolean hasFfmpeg() { return Mp4.find(null) != null; }

    @Test void videoReportsBothStagesAndServesTheMp4() throws Exception {
        assumeTrue(hasFfmpeg(), "no ffmpeg on the PATH");
        String png = java.util.Base64.getEncoder().encodeToString(Files.readAllBytes(stillPng()));
        String s = call("POST", "/api/video", "{\"model\": \"video:ltx-2.5-distilled\", \"prompt\": \"a kiln at dusk\","
                + "\"image\": \"data:image/png;base64," + png + "\", \"width\": 160, \"height\": 128, \"frames\": 9, \"fps\": 8, \"seed\": 3}").body();
        var progress = event(s, "progress");
        assertEquals(2L, progress.get("stage"));
        assertEquals(3L, progress.get("steps"));
        assertTrue(s.contains("\"state\":\"encoding\""));
        var done = event(s, "done");
        assertEquals(320L, done.get("width"));  // the refine doubles the draft
        assertEquals(9L, done.get("frames"));
        assertEquals(true, done.get("audio"));
        var mp4 = client.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + done.get("url"))).build(),
                HttpResponse.BodyHandlers.ofByteArray());
        assertEquals("video/mp4", mp4.headers().firstValue("Content-Type").orElse(""));
        assertEquals("ftyp", new String(mp4.body(), 4, 4));
        // Only the .mp4 is kept.
        try (var files = Files.list(dir.resolve("outputs"))) {
            assertEquals(List.of(done.get("url").toString().substring("/api/outputs/".length())),
                    files.map(p -> p.getFileName().toString()).toList());
        }
    }

    @Test void videoWithoutAudioIsAModelOfItsOwn() throws Exception {
        assumeTrue(hasFfmpeg(), "no ffmpeg on the PATH");
        String s = call("POST", "/api/video", "{\"model\": \"video:ltx-2.5-distilled\", \"prompt\": \"rain\", \"width\": 128,"
                + "\"height\": 128, \"frames\": 9, \"fps\": 8, \"steps\": 2, \"twoStage\": false, \"audio\": false}").body();
        var done = event(s, "done");
        assertEquals(false, done.get("audio"));
        assertEquals(128L, done.get("width"));
        assertEquals(2L, event(s, "progress").get("steps"));
        var silent = model("video:ltx-2.5-distilled+silent");
        assertEquals(true, silent.get("loaded"));
        assertFalse(((List<?>) silent.get("parts")).stream().anyMatch(p -> ((Map<?, ?>) p).get("role").equals("audio_vae")));
        assertEquals(false, model("video:ltx-2.5-distilled").get("loaded"));
        assertEquals(true, Json.object(call("POST", "/api/models/unload", "{\"id\": \"video:ltx-2.5-distilled+silent\"}").body()).get("unloaded"));
    }

    @Test void videoWithoutFfmpegSaysWhereTheFramesAre() throws Exception {
        var bare = new StudioServer(studio, 0, null);
        bare.start();
        try {
            var r = client.send(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + bare.port() + "/api/video"))
                    .POST(HttpRequest.BodyPublishers.ofString("{\"model\": \"video:ltx-2.5-distilled\", \"prompt\": \"x\","
                            + "\"width\": 128, \"height\": 128, \"frames\": 9, \"twoStage\": false, \"steps\": 1}")).build(),
                    HttpResponse.BodyHandlers.ofString());
            String message = event(r.body(), "error").get("message").toString();
            assertTrue(message.contains("needs ffmpeg") && message.contains("-frames"), message);
        } finally {
            bare.stop();
        }
    }

    private Path stillPng() throws Exception {
        var img = new java.awt.image.BufferedImage(64, 48, java.awt.image.BufferedImage.TYPE_INT_RGB);
        Path p = dir.resolve("still.png");
        javax.imageio.ImageIO.write(img, "png", p.toFile());
        return p;
    }

    private static String pngDataUrl(int w, int h) throws Exception {
        var img = new java.awt.image.BufferedImage(w, h, java.awt.image.BufferedImage.TYPE_INT_RGB);
        var out = new java.io.ByteArrayOutputStream();
        javax.imageio.ImageIO.write(img, "png", out);
        return "data:image/png;base64," + java.util.Base64.getEncoder().encodeToString(out.toByteArray());
    }

    @Test void inspectStreamsAReplyAboutTheImage() throws Exception {
        assertEquals(true, model("vision:qwen3-vl-4b").get("available"));
        assertEquals(false, model("vision:medgemma-4b").get("available"));
        assertTrue(model("vision:medgemma-4b").get("license").toString().contains("regulator"));
        // The vision models are the Inspect window's, not chat models.
        var status = Json.object(call("GET", "/api/models", null).body());
        assertFalse(Json.list(status, "models").stream().map(m -> ((Map<?, ?>) m).get("id").toString())
                .anyMatch(id -> id.startsWith("chat:") && id.contains("VL")));
        String s = call("POST", "/api/inspect", "{\"model\": \"vision:qwen3-vl-4b\", \"prompt\": \"What is this?\","
                + "\"image\": \"" + pngDataUrl(2000, 1000) + "\"}").body();
        assertTrue(s.contains("event: token"));
        var done = event(s, "done");
        // Scaled down to 1280 on its longest side before the model sees it.
        assertEquals(1280L, done.get("width"));
        assertEquals(640L, done.get("height"));
        assertTrue(done.get("text").toString().contains("1280×640"), done.get("text").toString());
        assertNull(done.get("json"));
        assertTrue(((Number) done.get("generatedTokens")).intValue() > 0);
    }

    @Test void inspectWithASchemaGivesItsJson() throws Exception {
        String schema = "{\"type\": \"object\", \"properties\": {\"vendor\": {\"type\": \"string\"}, \"total\": {\"type\": \"number\"},"
                + "\"box\": {\"type\": \"array\", \"items\": {\"type\": \"integer\"}, \"minItems\": 4, \"maxItems\": 4}}}";
        String s = call("POST", "/api/inspect", "{\"model\": \"vision:qwen3-vl-4b\", \"prompt\": \"Read it.\", \"schema\": " + schema
                + ", \"image\": \"" + pngDataUrl(64, 48) + "\"}").body();
        var json = (Map<?, ?>) event(s, "done").get("json");
        assertEquals("(stub)", json.get("vendor"));
        assertEquals(1L, json.get("total"));
        assertEquals(List.of(250L, 250L, 500L, 500L), json.get("box"));
    }

    @Test void aSchemaSentAsTextIsReadAsJson() throws Exception {
        String schema = Json.write("{\"type\": \"object\", \"properties\": {\"total\": {\"type\": \"number\"}}}");
        var done = event(call("POST", "/api/inspect", "{\"model\": \"vision:qwen3-vl-4b\", \"prompt\": \"Read it.\", \"schema\": " + schema
                + ", \"image\": \"" + pngDataUrl(8, 8) + "\"}").body(), "done");
        assertEquals(1L, ((Map<?, ?>) done.get("json")).get("total"));
        // Text that isn't a JSON object is refused before anything is loaded.
        var r = call("POST", "/api/inspect", "{\"model\": \"vision:qwen3-vl-4b\", \"prompt\": \"x\", \"schema\": \"[1]\","
                + "\"image\": \"" + pngDataUrl(8, 8) + "\"}");
        assertEquals(400, r.statusCode());
    }

    @Test void anOcrModelReadsThenAnotherFillsTheSchema() throws Exception {
        String s = call("POST", "/api/inspect", "{\"model\": \"vision:paddleocr-vl\", \"prompt\": \"Read the invoice.\","
                + "\"schema\": {\"type\": \"object\", \"properties\": {\"total\": {\"type\": \"number\"}}},"
                + "\"image\": \"" + pngDataUrl(64, 48) + "\"}").body();
        assertTrue(s.contains("\"state\":\"structuring\""), s);
        var done = event(s, "done");
        assertEquals("vision:qwen3-vl-4b", done.get("filledBy"));
        assertTrue(done.get("read").toString().startsWith("(stub)"));
        assertEquals(1L, ((Map<?, ?>) done.get("json")).get("total"));
    }

    @Test void anOcrModelWithASchemaAndNothingToFillItIsRefused() throws Exception {
        var qwen = LlamaModel.Weights.QWEN3_VL_4B;
        Files.delete(dir.resolve("hub").resolve("models--" + qwen.repo.replace("/", "--")).resolve("snapshots/abc").resolve(qwen.modelFile));
        var r = call("POST", "/api/inspect", "{\"model\": \"vision:paddleocr-vl\", \"prompt\": \"OCR:\","
                + "\"schema\": {\"type\": \"object\"}, \"image\": \"" + pngDataUrl(8, 8) + "\"}");
        assertEquals(400, r.statusCode());
        assertTrue(r.body().contains("download Qwen3-VL 4B"), r.body());
        // Without a schema it reads on its own.
        var done = event(call("POST", "/api/inspect", "{\"model\": \"vision:paddleocr-vl\", \"image\": \"" + pngDataUrl(8, 8) + "\"}").body(), "done");
        assertNull(done.get("read"));
        assertNull(done.get("filledBy"));
    }

    @Test void anImageClaimingTooManyPixelsIsRefusedBeforeItIsDecoded() throws Exception {
        var out = new java.io.ByteArrayOutputStream();
        javax.imageio.ImageIO.write(new java.awt.image.BufferedImage(1, 1, java.awt.image.BufferedImage.TYPE_INT_RGB), "png", out);
        byte[] png = out.toByteArray();
        // IHDR's width and height (bytes 16 to 23) say 50000 × 50000, and its CRC (over bytes 12 to 28) agrees.
        var b = java.nio.ByteBuffer.wrap(png);
        b.putInt(16, 50_000).putInt(20, 50_000);
        var crc = new java.util.zip.CRC32();
        crc.update(png, 12, 17);
        b.putInt(29, (int) crc.getValue());
        var r = call("POST", "/api/inspect", "{\"model\": \"vision:qwen3-vl-4b\", \"prompt\": \"x\", \"image\": \"data:image/png;base64,"
                + java.util.Base64.getEncoder().encodeToString(png) + "\"}");
        assertEquals(400, r.statusCode());
        assertTrue(r.body().contains("too large"), r.body());
    }

    @Test void aTransparentImageIsLaidOnWhite() {
        var clear = new java.awt.image.BufferedImage(4, 4, java.awt.image.BufferedImage.TYPE_INT_ARGB);
        assertEquals(0xFFFFFF, StudioServer.fit(clear, StudioServer.INSPECT_MAX_SIDE).getRGB(1, 1) & 0xFFFFFF);
        var big = new java.awt.image.BufferedImage(2560, 100, java.awt.image.BufferedImage.TYPE_INT_ARGB);
        var fitted = StudioServer.fit(big, StudioServer.INSPECT_MAX_SIDE);
        assertEquals(1280, fitted.getWidth());
        assertEquals(0xFFFFFF, fitted.getRGB(10, 10) & 0xFFFFFF);
    }

    @Test void aSilentStreamIsKeptAlive() throws Exception {
        long was = Sse.keepAliveMillis;
        Sse.keepAliveMillis = 100;
        try {
            // The stub singer says nothing for 800 ms.
            String s = call("POST", "/api/song", "{\"model\": \"song:yue2-3b\", \"caption\": \"folk\", \"duration\": 1}").body();
            assertTrue(s.contains(": keep-alive\n\n"), s);
            event(s, "done");
        } finally {
            Sse.keepAliveMillis = was;
        }
    }

    @Test void refusesWhatItCannotDo() throws Exception {
        assertEquals(400, call("POST", "/api/chat", "{\"model\": \"chat:nope\"}").statusCode());
        assertEquals(400, call("POST", "/api/image", "{\"model\": \"song:acestep-turbo\", \"prompt\": \"x\"}").statusCode());
        assertEquals(400, call("POST", "/api/song", "{\"model\": \"song:acestep-turbo-lm4b\", \"caption\": \"x\"}").statusCode());
        assertEquals(400, call("POST", "/api/image", "{\"model\": \"image:flux2-klein\", \"prompt\": \"x\", \"width\": 100}").statusCode());
        assertEquals(400, call("POST", "/api/song", "{\"model\": \"song:yue2-3b\", \"caption\": \" \"}").statusCode());
        assertEquals(400, call("POST", "/api/video", "{\"model\": \"video:ltx-2.5-distilled\", \"prompt\": \"x\", \"frames\": 50}").statusCode());
        assertEquals(400, call("POST", "/api/video", "{\"model\": \"video:ltx-2.5-distilled\", \"prompt\": \"x\", \"width\": 100}").statusCode());
        assertEquals(400, call("POST", "/api/video", "{\"model\": \"video:ltx-2.5-distilled\", \"prompt\": \"x\", \"image\": \"bm90IGFuIGltYWdl\"}").statusCode());
        assertEquals(400, call("POST", "/api/video", "{\"model\": \"video:ltx-2.5-dev\", \"prompt\": \"x\"}").statusCode());
        assertEquals(400, call("POST", "/api/inspect", "{\"model\": \"vision:qwen3-vl-4b\", \"prompt\": \"x\"}").statusCode());
        assertEquals(400, call("POST", "/api/inspect", "{\"model\": \"vision:qwen3-vl-4b\", \"image\": \"" + pngDataUrl(8, 8) + "\"}").statusCode());
        assertEquals(400, call("POST", "/api/inspect", "{\"model\": \"vision:medgemma-4b\", \"prompt\": \"x\"}").statusCode());
        assertEquals(405, call("GET", "/api/chat", null).statusCode());
        assertEquals(404, call("GET", "/api/outputs/..%2Fsecret.png", null).statusCode());
        assertEquals(404, call("GET", "/api/models/extra", null).statusCode());
    }
}
