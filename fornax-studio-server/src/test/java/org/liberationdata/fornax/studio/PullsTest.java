package org.liberationdata.fornax.studio;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelFolders;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelSet.Part;

import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.time.Duration;
import java.util.EnumSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/** Getting models from inside the studio, over HTTP, from a Hub on localhost: no network, no engine. */
class PullsTest {
    static final String COMMIT = "0123456789abcdef0123456789abcdef01234567";
    static final String TOKEN = "hf_studio_test_token";

    @TempDir Path dir;
    private HttpServer hub;
    private Studio studio;
    private StudioServer server;
    private Pulls pulls;
    private final HttpClient client = HttpClient.newHttpClient();
    private final Map<String, byte[]> files = new java.util.concurrent.ConcurrentHashMap<>();
    private final List<String> gets = new CopyOnWriteArrayList<>();
    private final List<String> authorizations = new CopyOnWriteArrayList<>();
    /** Bytes sent between pauses, and the pause, to make a download slow enough to watch or cancel. */
    private volatile int chunk = 1 << 20;
    private volatile long pauseMillis = 0;
    private volatile long reserveBytes = 0;

    static final List<ModelSet> SETS = List.of(
            ModelSet.of("tiny", Kind.SONG, "Tiny", "acestep.cpp",
                    Part.of("dit", "o/tiny", "dit.gguf"), Part.of("vae", "o/tiny", "vae.gguf")).withLicense("CC BY-NC 4.0").withAliases("song:tiny"),
            ModelSet.of("locked", Kind.VIDEO, "Locked", "stable-diffusion.cpp", Part.of("diffusion_model", "gated/locked", "model.safetensors"))
                    .withGated(true));

    @BeforeEach void start() throws Exception {
        files.put("o/tiny/dit.gguf", bytes(300_000, 1));
        files.put("o/tiny/vae.gguf", bytes(100_000, 2));
        files.put("gated/locked/model.safetensors", bytes(50_000, 3));
        hub = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        hub.createContext("/", ex -> {
            String auth = ex.getRequestHeaders().getFirst("Authorization");
            if (auth != null) authorizations.add(auth);
            var m = java.util.regex.Pattern.compile("/(.+?/.+?)/resolve/[^/]+/(.+)").matcher(ex.getRequestURI().getPath());
            byte[] content = m.matches() ? files.get(m.group(1) + "/" + m.group(2)) : null;
            if (content == null) { ex.sendResponseHeaders(404, -1); ex.close(); return; }
            if (m.group(1).startsWith("gated/") && !("Bearer " + TOKEN).equals(auth)) { ex.sendResponseHeaders(401, -1); ex.close(); return; }
            ex.getResponseHeaders().set("X-Repo-Commit", COMMIT);
            ex.getResponseHeaders().set("ETag", "\"" + sha256(content) + "\"");
            if (ex.getRequestMethod().equals("HEAD")) {
                ex.getResponseHeaders().set("Content-Length", String.valueOf(content.length));
                ex.sendResponseHeaders(200, -1);
                ex.close();
                return;
            }
            gets.add(m.group(2));
            int from = 0;
            String range = ex.getRequestHeaders().getFirst("Range");
            if (range != null && range.startsWith("bytes=")) from = Integer.parseInt(range.substring(6, range.indexOf('-')));
            if (from > 0) ex.getResponseHeaders().set("Content-Range", "bytes " + from + "-" + (content.length - 1) + "/" + content.length);
            ex.sendResponseHeaders(from > 0 ? 206 : 200, content.length - from);
            try (OutputStream out = ex.getResponseBody()) {
                for (int i = from; i < content.length; i += chunk) {
                    out.write(content, i, Math.min(chunk, content.length - i));
                    out.flush();
                    if (pauseMillis > 0) Thread.sleep(pauseMillis);
                }
            } catch (Exception gone) { /* the client cancelled */ }
        });
        hub.start();
        URI endpoint = URI.create("http://127.0.0.1:" + hub.getAddress().getPort());
        Path cache = dir.resolve("hub");
        pulls = new Pulls(SETS, token -> FornaxHub.builder().cacheRoot(cache).modelFolders(ModelFolders.none())
                .endpoint(endpoint).token(token).offline(false).reserveBytes(reserveBytes).retryDelay(Duration.ofMillis(10))
                .catalogue(SETS).build(), endpoint);
        studio = new Studio(new Catalog(cache), 1L << 30, null, EnumSet.allOf(Catalog.Kind.class), dir.resolve("outputs"));
        server = new StudioServer(studio, 0, null, pulls);
        server.start();
    }

    @AfterEach void stop() {
        server.stop();
        studio.close();
        hub.stop(0);
    }

    static byte[] bytes(int n, int seed) {
        byte[] b = new byte[n];
        new java.util.Random(seed).nextBytes(b);
        return b;
    }

    static String sha256(byte[] b) {
        try { return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(b)); }
        catch (Exception e) { throw new IllegalStateException(e); }
    }

    private HttpResponse<String> call(String method, String path, String body) throws Exception {
        var b = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + path)).timeout(Duration.ofSeconds(30));
        b = body == null ? b.method(method, HttpRequest.BodyPublishers.noBody()) : b.method(method, HttpRequest.BodyPublishers.ofString(body));
        return client.send(b.build(), HttpResponse.BodyHandlers.ofString());
    }

    /** A pull's stream, read as it arrives: each event as "name data". */
    private CompletableFuture<Void> pullStreaming(String body, List<String> events) {
        var req = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + "/api/models/pull"))
                .POST(HttpRequest.BodyPublishers.ofString(body)).build();
        return client.sendAsync(req, HttpResponse.BodyHandlers.ofLines()).thenAccept(r -> {
            String[] name = {null};
            r.body().forEach(line -> {
                if (line.startsWith("event: ")) name[0] = line.substring(7);
                else if (line.startsWith("data: ")) events.add(name[0] + " " + line.substring(6));
            });
        });
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

    @SuppressWarnings("unchecked")
    private Map<String, Object> entry(String id) throws Exception {
        var all = Json.list(Json.object(call("GET", "/api/catalogue", null).body()), "models");
        return all.stream().map(m -> (Map<String, Object>) m).filter(m -> m.get("id").equals(id)).findFirst().orElseThrow();
    }

    static void await(java.util.function.BooleanSupplier until) throws InterruptedException {
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(20);
        while (!until.getAsBoolean()) {
            if (System.nanoTime() > end) fail("timed out");
            Thread.sleep(10);
        }
    }

    @Test void theCatalogueSaysWhatIsOnDiskWithItsLicenceAndTerms() throws Exception {
        var tiny = entry("tiny");
        assertEquals("MISSING", tiny.get("state"));
        assertEquals("CC BY-NC 4.0", tiny.get("license"));
        assertEquals(List.of("song:tiny"), tiny.get("aliases"));
        assertEquals("song", tiny.get("kind"));
        assertEquals(false, tiny.get("downloading"));
        var locked = entry("locked");
        assertEquals(true, locked.get("gated"));
        assertTrue(((String) locked.get("termsUrl")).endsWith("/gated/locked"), locked.toString());
        assertTrue(gets.isEmpty());   // from the disk alone
    }

    @Test void aPlanSaysWhatADownloadWouldTakeWithoutDownloading() throws Exception {
        var plan = Json.object(call("POST", "/api/models/plan", "{\"id\": \"song:tiny\"}").body());
        assertEquals("tiny", plan.get("id"));
        assertEquals(400_000.0, ((Number) plan.get("bytesToDownload")).doubleValue());
        assertTrue(((Number) plan.get("freeBytes")).doubleValue() > 0);
        assertEquals(false, plan.get("complete"));
        assertTrue(gets.isEmpty());
    }

    @Test void aPlanForAGatedModelWithoutATokenSaysWhereToAcceptItsTerms() throws Exception {
        var denied = call("POST", "/api/models/plan", "{\"id\": \"locked\"}");
        assertEquals(403, denied.statusCode(), denied.body());
        var error = Json.object(denied.body());
        assertEquals("ACCESS_DENIED", error.get("reason"));
        assertTrue(((String) error.get("termsUrl")).endsWith("/gated/locked"), error.toString());
        assertNotNull(error.get("error"));

        var ok = call("POST", "/api/models/plan", "{\"id\": \"locked\", \"token\": \"" + TOKEN + "\"}");
        assertEquals(200, ok.statusCode(), ok.body());
        assertEquals(50_000.0, ((Number) Json.object(ok.body()).get("bytesToDownload")).doubleValue());
        assertFalse(ok.body().contains(TOKEN), "the token is never sent back");
        assertTrue(gets.isEmpty());
    }

    @Test void aPullStreamsProgressThenItsFilesAndTheModelIsOnDisk() throws Exception {
        chunk = 20_000;
        String stream = call("POST", "/api/models/pull", "{\"id\": \"song:tiny\"}").body();
        assertEquals(false, event(stream, "start").get("attached"));
        assertEquals("CC BY-NC 4.0", event(stream, "start").get("license"));
        var progress = event(stream, "progress");
        assertEquals(400_000.0, ((Number) progress.get("setBytesTotal")).doubleValue());
        assertTrue(stream.contains("\"phase\":\"DOWNLOADING\"") || stream.contains("\"phase\": \"DOWNLOADING\""), stream);
        @SuppressWarnings("unchecked") var files = (Map<String, Object>) event(stream, "done").get("files");
        assertArrayEquals(this.files.get("o/tiny/dit.gguf"), Files.readAllBytes(Path.of((String) files.get("dit"))));
        assertEquals("PRESENT", entry("tiny").get("state"));
        // and a second pull downloads nothing
        int before = gets.size();
        assertTrue(call("POST", "/api/models/pull", "{\"id\": \"tiny\"}").body().contains("event: done"));
        assertEquals(before, gets.size());
    }

    @Test void aGatedModelWithoutATokenSaysWhereToAcceptItsTerms() throws Exception {
        String stream = call("POST", "/api/models/pull", "{\"id\": \"locked\"}").body();
        var error = event(stream, "error");
        assertEquals("ACCESS_DENIED", error.get("reason"));
        assertTrue(((String) error.get("termsUrl")).endsWith("/gated/locked"), error.toString());

        String ok = call("POST", "/api/models/pull", "{\"id\": \"locked\", \"token\": \"" + TOKEN + "\"}").body();
        assertTrue(ok.contains("event: done"), ok);
        assertFalse(ok.contains(TOKEN), "the token is never sent back");
        assertTrue(authorizations.contains("Bearer " + TOKEN));
    }

    @Test void theAppsTokenIsUsedForRequestsWithoutOneAndNeverSentBack() throws Exception {
        assertEquals(false, Json.object(call("GET", "/api/hub", null).body()).get("tokenSet"));
        String set = call("POST", "/api/hub/token", "{\"token\": \" " + TOKEN + " \"}").body();
        assertEquals(true, Json.object(set).get("tokenSet"));
        assertFalse(set.contains(TOKEN));
        assertEquals(true, Json.object(call("GET", "/api/hub", null).body()).get("tokenSet"));

        assertEquals(200, call("POST", "/api/models/plan", "{\"id\": \"locked\"}").statusCode());
        assertEquals(false, Json.object(call("POST", "/api/hub/token", "{\"token\": \"\"}").body()).get("tokenSet"));
        assertEquals(403, call("POST", "/api/models/plan", "{\"id\": \"locked\"}").statusCode());   // cleared: no token again
        call("POST", "/api/hub/token", "{\"token\": \"" + TOKEN + "\"}");
        String ok = call("POST", "/api/models/pull", "{\"id\": \"locked\"}").body();
        assertTrue(ok.contains("event: done"), ok);
        assertFalse(ok.contains(TOKEN));
        assertTrue(authorizations.contains("Bearer " + TOKEN), "trimmed, and sent to the Hub");
    }

    @Test void aMalformedTokenIsRefusedWithoutBeingEchoed() throws Exception {
        String bad = "hf_secret\\nInjected: yes";   // a newline inside: no header can carry it
        call("POST", "/api/hub/token", "{\"token\": \"" + TOKEN + "\"}");
        for (String body : List.of("{\"token\": \"" + bad + "\"}", "{\"token\": \"hf_a b\"}", "{\"token\": 12345}",
                "{\"token\": {\"t\": \"hf_secret\"}}", "hf_secret")) {
            var r = call("POST", "/api/hub/token", body);
            assertEquals(400, r.statusCode(), body + " -> " + r.body());
            assertFalse(r.body().contains("hf_secret") || r.body().contains("12345"), r.body());
        }
        assertEquals(true, Json.object(call("GET", "/api/hub", null).body()).get("tokenSet"), "the app's token is kept");

        var plan = call("POST", "/api/models/plan", "{\"id\": \"locked\", \"token\": \"" + bad + "\"}");
        assertEquals(400, plan.statusCode(), plan.body());
        assertFalse(plan.body().contains("hf_secret"), plan.body());
        var pull = call("POST", "/api/models/pull", "{\"id\": \"locked\", \"token\": \"" + bad + "\"}");
        assertEquals(400, pull.statusCode(), "refused before the stream begins: " + pull.body());
        assertFalse(pull.body().contains("hf_secret"), pull.body());

        for (String path : List.of("/api/models/plan", "/api/models/pull")) {   // not a string: refused, not sent to the Hub
            int before = authorizations.size();
            var r = call("POST", path, "{\"id\": \"locked\", \"token\": {\"t\": \"hf_secret\"}}");
            assertEquals(400, r.statusCode(), path + " -> " + r.body());
            assertFalse(r.body().contains("hf_secret"), r.body());
            assertEquals(before, authorizations.size(), authorizations.toString());
        }
    }

    @Test void tooLittleDiskFailsBeforeAnyDownload() throws Exception {
        reserveBytes = Long.MAX_VALUE / 4;
        var error = event(call("POST", "/api/models/pull", "{\"id\": \"tiny\"}").body(), "error");
        assertEquals("DISK_SPACE", error.get("reason"));
        assertTrue(gets.isEmpty());
    }

    @Test void cancelStopsTheDownloadAndKeepsWhatArrived() throws Exception {
        chunk = 10_000;
        pauseMillis = 50;
        List<String> events = new CopyOnWriteArrayList<>();
        var done = pullStreaming("{\"id\": \"tiny\"}", events);
        await(() -> events.stream().anyMatch(e -> e.startsWith("progress ") && e.contains("DOWNLOADING")));
        var downloading = entry("tiny");
        assertEquals(true, downloading.get("downloading"));
        assertNotNull(downloading.get("progress"), downloading.toString());
        assertEquals(true, Json.object(call("POST", "/api/models/pull/cancel", "{\"id\": \"tiny\"}").body()).get("cancelled"));
        done.get(20, TimeUnit.SECONDS);
        assertTrue(events.stream().anyMatch(e -> e.startsWith("cancelled ")), events.toString());
        assertEquals("PARTIAL", entry("tiny").get("state"));
        assertEquals(false, entry("tiny").get("downloading"));
        assertEquals(false, Json.object(call("POST", "/api/models/pull/cancel", "{\"id\": \"tiny\"}").body()).get("cancelled"));
        // the next pull resumes it
        pauseMillis = 0;
        assertTrue(call("POST", "/api/models/pull", "{\"id\": \"tiny\"}").body().contains("event: done"));
        assertEquals("PRESENT", entry("tiny").get("state"));
    }

    @Test void aSecondRequestWatchesTheDownloadUnderWay() throws Exception {
        chunk = 10_000;
        pauseMillis = 20;
        List<String> first = new CopyOnWriteArrayList<>(), second = new CopyOnWriteArrayList<>();
        var a = pullStreaming("{\"id\": \"tiny\"}", first);
        await(() -> first.stream().anyMatch(e -> e.startsWith("progress ")));
        var b = pullStreaming("{\"id\": \"song:tiny\"}", second);
        a.get(30, TimeUnit.SECONDS);
        b.get(30, TimeUnit.SECONDS);
        assertTrue(first.getFirst().startsWith("start "), first.toString());
        assertTrue(second.getFirst().startsWith("start ") && second.getFirst().contains("\"attached\":true"), second.toString());
        assertTrue(first.stream().anyMatch(e -> e.startsWith("done ")));
        assertTrue(second.stream().anyMatch(e -> e.startsWith("done ")));
        assertEquals(2, gets.size(), "one download: " + gets);
    }

    @Test void aPageThatLeavesDoesntStopTheDownload() throws Exception {
        chunk = 10_000;
        pauseMillis = 20;
        var req = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.port() + "/api/models/pull"))
                .POST(HttpRequest.BodyPublishers.ofString("{\"id\": \"tiny\"}")).build();
        try (Stream<String> lines = client.send(req, HttpResponse.BodyHandlers.ofLines()).body()) {
            lines.filter(l -> l.contains("DOWNLOADING")).findFirst();   // then the page goes
        }
        await(() -> {
            try { return "PRESENT".equals(entry("tiny").get("state")); } catch (Exception e) { return false; }
        });
    }

    @Test void stoppingTheServerStopsItsDownloads() throws Exception {
        chunk = 10_000;
        pauseMillis = 50;
        List<String> events = new CopyOnWriteArrayList<>();
        pullStreaming("{\"id\": \"tiny\"}", events);
        await(() -> events.stream().anyMatch(e -> e.startsWith("progress ") && e.contains("DOWNLOADING")));
        server.stop();
        @SuppressWarnings("unchecked") java.util.function.Supplier<Map<String, Object>> tiny = () -> pulls.catalogue().stream()
                .map(m -> (Map<String, Object>) m).filter(m -> m.get("id").equals("tiny")).findFirst().orElseThrow();
        await(() -> tiny.get().get("downloading").equals(false));
        assertEquals("PARTIAL", tiny.get().get("state"), "cancelled, not finished");
    }

    @Test void anUnknownModelIsRefused() throws Exception {
        var r = call("POST", "/api/models/pull", "{\"id\": \"nothing\"}");
        assertEquals(400, r.statusCode());
        assertTrue(r.body().contains("no model set named nothing"), r.body());
        assertEquals(400, call("POST", "/api/models/plan", "{}").statusCode());
        assertEquals(400, call("POST", "/api/models/pull/cancel", "{\"id\": \"nothing\"}").statusCode());
    }

    @Test void aPlanTheHubCantAnswerIsABadGateway() throws Exception {
        files.remove("o/tiny/vae.gguf");
        var r = call("POST", "/api/models/plan", "{\"id\": \"tiny\"}");
        assertEquals(502, r.statusCode(), r.body());
        var error = Json.object(r.body());
        assertEquals("NOT_FOUND", error.get("reason"));
        assertNull(error.get("termsUrl"), error.toString());
        assertNotNull(error.get("error"));
    }

    @Test void aPullThatCantReachTheHubSaysSo() throws Exception {
        hub.stop(0);
        var error = event(call("POST", "/api/models/pull", "{\"id\": \"tiny\"}").body(), "error");
        assertEquals("IO", error.get("reason"), error.toString());
        assertEquals("tiny", error.get("id"));
        assertNotNull(error.get("message"));
    }

    @Test void aPullThatFailsToStartSaysSoInItsStream() throws Exception {
        var broken = new StudioServer(studio, 0, null, new Pulls(SETS, token -> { throw new IllegalStateException("no hub"); },
                URI.create("http://127.0.0.1:1")));
        broken.start();
        try {
            var req = HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + broken.port() + "/api/models/pull"))
                    .POST(HttpRequest.BodyPublishers.ofString("{\"id\": \"tiny\"}")).build();
            var error = event(client.send(req, HttpResponse.BodyHandlers.ofString()).body(), "error");
            assertEquals("IO", error.get("reason"), error.toString());
            assertEquals("tiny", error.get("id"));
            assertEquals("no hub", error.get("message"));
        } finally {
            broken.stop();
        }
    }

    @Test void theStandardPullsLookInTheStudiosModelFolders() throws Exception {
        Path folder = Files.createDirectories(dir.resolve("models"));
        ModelSet set = org.liberationdata.fornax.hub.ModelCatalog.all().getFirst();
        for (Part p : set.parts()) {
            Path f = folder.resolve(p.primary().file());
            Files.createDirectories(f.getParent());
            Files.write(f, new byte[] {1});
        }
        @SuppressWarnings("unchecked") var entry = Pulls.standard(new Catalog(dir.resolve("hub"), List.of(folder))).catalogue().stream()
                .map(m -> (Map<String, Object>) m).filter(m -> m.get("id").equals(set.id())).findFirst().orElseThrow();
        assertEquals("PRESENT", entry.get("state"), entry.toString());
    }
}
