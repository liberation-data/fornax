package org.liberationdata.kiln.cli;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.StringReader;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.*;

/** {@code kiln pull} against a fake Hub on localhost: no network, no GPU. */
class PullTest {
    static final String REPO = "acme/Tiny-GGUF", COMMIT = "0123456789abcdef0123456789abcdef01234567";

    @TempDir Path root;
    HttpServer server;
    final Map<String, byte[]> lfs = new ConcurrentHashMap<>();   // sha256 -> bytes, served by the "CDN"
    final List<String> requests = new ArrayList<>();
    final List<String> cdnAuth = new ArrayList<>(), hubAuth = new ArrayList<>();
    byte[] model, readme;
    HubCache cache;
    ByteArrayOutputStream out;

    @BeforeEach
    void start() throws Exception {
        model = new byte[300_000];
        for (int i = 0; i < model.length; i++) model[i] = (byte) (i * 31);
        readme = "# Tiny\n".getBytes(StandardCharsets.UTF_8);
        String sha = sha256(model);
        lfs.put(sha, model);
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        int port = server.getAddress().getPort();
        server.createContext("/api/models/", ex -> {
            record(ex, hubAuth);
            String path = ex.getRequestURI().getPath();
            if (!path.equals("/api/models/" + REPO + "/revision/main")) { send(ex, 404, new byte[0]); return; }
            String json = """
                    {"id": "%s", "sha": "%s", "gated": false, "siblings": [
                      {"rfilename": "README.md", "blobId": "readme0000", "size": %d},
                      {"rfilename": "tiny-Q8_0.gguf", "blobId": "gitblob1", "size": %d,
                       "lfs": {"sha256": "%s", "size": %d, "pointerSize": 134}},
                      {"rfilename": "tiny-Q4_0.gguf", "blobId": "gitblob2", "size": 10, "lfs": {"sha256": "%s", "size": 10}}]}"""
                    .formatted(REPO, COMMIT, readme.length, model.length, sha, model.length, "f".repeat(64));
            send(ex, 200, json.getBytes(StandardCharsets.UTF_8));
        });
        server.createContext("/" + REPO + "/resolve/", ex -> {
            record(ex, hubAuth);
            String file = ex.getRequestURI().getPath().substring(("/" + REPO + "/resolve/" + COMMIT + "/").length());
            switch (file) {
                case "README.md" -> send(ex, 200, readme);
                case "tiny-Q8_0.gguf" -> {   // LFS files redirect to a CDN on another host name
                    ex.getResponseHeaders().add("Location", "http://localhost:" + port + "/cdn/" + sha);
                    send(ex, 302, new byte[0]);
                }
                default -> send(ex, 404, new byte[0]);
            }
        });
        server.createContext("/cdn/", ex -> {
            record(ex, cdnAuth);
            byte[] body = lfs.get(ex.getRequestURI().getPath().substring("/cdn/".length()));
            String range = ex.getRequestHeaders().getFirst("Range");
            if (range != null) {
                int from = Integer.parseInt(range.replaceAll("bytes=(\\d+)-", "$1"));
                ex.getResponseHeaders().add("Content-Range", "bytes " + from + "-" + (body.length - 1) + "/" + body.length);
                byte[] tail = new byte[body.length - from];
                System.arraycopy(body, from, tail, 0, tail.length);
                send(ex, 206, tail);
            } else {
                send(ex, 200, body);
            }
        });
        server.start();
        cache = new HubCache(root.resolve("hub"));
    }

    @AfterEach
    void stop() { server.stop(0); }

    private void record(HttpExchange ex, List<String> auth) {
        synchronized (requests) {
            String range = ex.getRequestHeaders().getFirst("Range");
            requests.add(ex.getRequestURI().getPath() + (range == null ? "" : " " + range));
            auth.add(String.valueOf(ex.getRequestHeaders().getFirst("Authorization")));
        }
    }

    private static void send(HttpExchange ex, int code, byte[] body) throws IOException {
        ex.sendResponseHeaders(code, body.length == 0 ? -1 : body.length);
        if (body.length > 0) ex.getResponseBody().write(body);
        ex.close();
    }

    private int pull(String input, String... argv) throws IOException {
        out = new ByteArrayOutputStream();
        var ps = new PrintStream(out, true, StandardCharsets.UTF_8);
        var hub = new Hub(URI.create("http://127.0.0.1:" + server.getAddress().getPort()), "hf_secret");
        var in = new BufferedReader(new StringReader(input == null ? "" : input));
        return new Pull(hub, cache, ps, ps, in, input != null).run(Args.parse(List.of(argv), Pull.OPTIONS, java.util.Set.of()));
    }

    @Test
    void downloadsIntoTheHubLayout() throws Exception {
        assertEquals(0, pull(null, REPO, "README.md", "tiny-Q8_0.gguf"));
        Path snap = cache.snapshotFile(REPO, COMMIT, "tiny-Q8_0.gguf");
        assertTrue(Files.isSymbolicLink(snap));
        assertEquals(Path.of("../../blobs/" + sha256(model)), Files.readSymbolicLink(snap));
        assertArrayEquals(model, Files.readAllBytes(snap));
        assertArrayEquals(readme, Files.readAllBytes(cache.snapshotFile(REPO, COMMIT, "README.md")));
        assertTrue(Files.isRegularFile(cache.blob(REPO, "readme0000")));
        assertEquals(COMMIT, cache.ref(REPO, "main").orElseThrow());
        assertEquals(cache.find(REPO, "tiny-Q8_0.gguf"), Models.resolve(REPO + ":Q8_0", cache) instanceof Models.Llm l
                ? java.util.Optional.of(l.path()) : java.util.Optional.empty());

        // the token goes to the Hub, never to the CDN it redirects to
        assertTrue(hubAuth.stream().allMatch("Bearer hf_secret"::equals), hubAuth.toString());
        assertEquals(List.of("null"), cdnAuth);
    }

    @Test
    void skipsCompleteFilesAndResumesPartialOnes() throws Exception {
        Path part = cache.blob(REPO, sha256(model) + ".incomplete");
        Files.createDirectories(part.getParent());
        Files.write(part, java.util.Arrays.copyOf(model, 100_000));
        assertEquals(0, pull(null, REPO + ":q8_0"));
        assertTrue(requests.contains("/cdn/" + sha256(model) + " bytes=100000-"), requests.toString());
        assertArrayEquals(model, Files.readAllBytes(cache.snapshotFile(REPO, COMMIT, "tiny-Q8_0.gguf")));
        assertFalse(Files.exists(part));

        requests.clear();
        assertEquals(0, pull(null, REPO, "tiny-Q8_0.gguf"));
        assertTrue(out.toString().contains("already downloaded"), out.toString());
        assertTrue(requests.stream().noneMatch(r -> r.contains("/resolve/") || r.startsWith("/cdn/")), requests.toString());
    }

    @Test
    void rejectsACorruptDownload() throws Exception {
        lfs.put(sha256(model), new byte[model.length]);   // right size, wrong bytes
        var e = assertThrows(Hub.HubException.class, () -> pull(null, REPO, "tiny-Q8_0.gguf"));
        assertTrue(e.getMessage().contains("sha256"), e.getMessage());
        assertFalse(Files.exists(cache.blob(REPO, sha256(model))));
        assertFalse(Files.exists(cache.snapshotFile(REPO, COMMIT, "tiny-Q8_0.gguf")));
    }

    @Test
    void withoutFilesListsTheModelsAndAsksOnATerminal() throws Exception {
        assertEquals(0, pull(null, REPO));
        assertTrue(out.toString().contains("tiny-Q8_0.gguf"), out.toString());
        assertTrue(out.toString().contains("kiln pull " + REPO + ":"), out.toString());
        assertTrue(cache.files(REPO).isEmpty());

        assertEquals(0, pull("2\n", REPO));   // sorted: tiny-Q4_0, tiny-Q8_0
        assertTrue(cache.find(REPO, "tiny-Q8_0.gguf").isPresent());
        assertEquals(0, pull("\n", REPO));    // Enter cancels
    }

    @Test
    void selectsByNameGlobAndTag() throws Exception {
        var info = new Hub(URI.create("http://127.0.0.1:" + server.getAddress().getPort()), null).info(REPO, "main");
        assertEquals(COMMIT, info.commit());
        assertEquals(List.of("README.md"), Pull.select(info, null, List.of("*.md")).stream().map(Hub.RepoFile::path).toList());
        assertEquals(List.of("tiny-Q4_0.gguf", "tiny-Q8_0.gguf"),
                Pull.select(info, null, List.of("tiny-*.gguf")).stream().map(Hub.RepoFile::path).sorted().toList());
        assertEquals(sha256(model), Pull.select(info, "Q8_0", List.of()).get(0).blobId());
        assertEquals("readme0000", Pull.select(info, null, List.of("README.md")).get(0).blobId());
        assertTrue(Pull.select(info, null, List.of()).isEmpty());
        assertThrows(Main.Failure.class, () -> Pull.select(info, "tiny", List.of()));        // two models
        assertThrows(Main.Failure.class, () -> Pull.select(info, null, List.of("nope.bin")));
    }

    @Test
    void reportsMissingRepos() {
        var e = assertThrows(Hub.HubException.class, () -> pull(null, "acme/Nope", "x"));
        assertTrue(e.getMessage().contains("not found"), e.getMessage());
    }

    static String sha256(byte[] b) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(b));
    }
}
