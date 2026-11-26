package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.hub.RepoIndex;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.hub.HubException;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelFolders;

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
import java.time.Duration;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@code fornax pull} of a repo's files against a fake Hub on localhost: the file list from the Hub's API, the files
 * through {@link FornaxHub} (HEAD for commit, size and hash, then the CDN). No network, no GPU.
 */
class PullTest {
    static final String REPO = "acme/Tiny-GGUF", GATED = "acme/Gated-GGUF", COMMIT = "0123456789abcdef0123456789abcdef01234567";

    @TempDir Path root;
    HttpServer server;
    final List<String> requests = new CopyOnWriteArrayList<>(), cdnAuth = new CopyOnWriteArrayList<>();
    byte[] model, readme;
    ByteArrayOutputStream out;

    @BeforeEach
    void start() throws Exception {
        model = new byte[300_000];
        for (int i = 0; i < model.length; i++) model[i] = (byte) (i * 31);
        readme = "# Tiny\n".getBytes(StandardCharsets.UTF_8);
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        int port = server.getAddress().getPort();
        server.createContext("/api/models/", ex -> {
            requests.add("API " + ex.getRequestURI().getPath());
            String path = ex.getRequestURI().getPath();
            String repo = path.equals("/api/models/" + REPO + "/revision/main") || path.equals("/api/models/" + REPO + "/revision/" + COMMIT) ? REPO
                    : path.equals("/api/models/" + GATED + "/revision/main") ? GATED : null;
            if (repo == null) { send(ex, 404, new byte[0]); return; }
            String json = """
                    {"id": "%s", "sha": "%s", "siblings": [
                      {"rfilename": "README.md", "blobId": "x", "size": %d},
                      {"rfilename": "tiny-Q8_0.gguf", "blobId": "y", "size": %d, "lfs": {"sha256": "%s", "size": %d}},
                      {"rfilename": "tiny-Q4_0.gguf", "blobId": "z", "size": 10, "lfs": {"sha256": "%s", "size": 10}}]}"""
                    .formatted(repo, COMMIT, readme.length, model.length, sha256(model), model.length, "f".repeat(64));
            send(ex, 200, json.getBytes(StandardCharsets.UTF_8));
        });
        server.createContext("/", ex -> {   // /<repo>/resolve/<rev>/<path>, HEAD and GET
            String path = ex.getRequestURI().getPath();
            requests.add(ex.getRequestMethod() + " " + path);
            if (path.startsWith("/" + GATED + "/")) {
                ex.getResponseHeaders().set("X-Error-Code", "GatedRepo");
                send(ex, 401, new byte[0]);
                return;
            }
            String prefix = "/" + REPO + "/resolve/" + (path.startsWith("/" + REPO + "/resolve/" + COMMIT + "/") ? COMMIT : "main") + "/";
            if (!path.startsWith(prefix)) { send(ex, 404, new byte[0]); return; }
            var h = ex.getResponseHeaders();
            h.set("X-Repo-Commit", COMMIT);
            switch (path.substring(prefix.length())) {
                case "README.md" -> {   // a small git file the Hub serves itself
                    h.set("ETag", "\"" + gitSha1(readme) + "\"");
                    send(ex, 200, readme);
                }
                case "tiny-Q8_0.gguf" -> {   // an LFS file: a redirect to the CDN
                    h.set("X-Linked-Etag", "\"" + sha256(model) + "\"");
                    h.set("X-Linked-Size", String.valueOf(model.length));
                    h.set("Location", "http://localhost:" + port + "/cdn/" + sha256(model));
                    send(ex, 302, new byte[0]);
                }
                default -> { h.set("X-Error-Code", "EntryNotFound"); send(ex, 404, new byte[0]); }
            }
        });
        server.createContext("/cdn/", ex -> {
            requests.add("CDN " + ex.getRequestURI().getPath());
            String auth = ex.getRequestHeaders().getFirst("Authorization");
            if (auth != null) cdnAuth.add(auth);
            send(ex, 200, model);
        });
        server.start();
    }

    @AfterEach
    void stop() { server.stop(0); }

    private static void send(HttpExchange ex, int code, byte[] body) throws IOException {
        boolean head = ex.getRequestMethod().equals("HEAD");
        if (head) ex.getResponseHeaders().set("Content-Length", String.valueOf(body.length));
        ex.sendResponseHeaders(code, head || body.length == 0 ? -1 : body.length);
        if (!head && body.length > 0) ex.getResponseBody().write(body);
        ex.close();
    }

    URI endpoint() { return URI.create("http://127.0.0.1:" + server.getAddress().getPort()); }

    FornaxHub hub() {
        return FornaxHub.builder().cacheRoot(root.resolve("hub")).endpoint(endpoint()).token("hf_secret")
                .modelFolders(ModelFolders.none()).reserveBytes(0).retryDelay(Duration.ofMillis(10)).build();
    }

    private int pull(String input, String... argv) throws IOException {
        out = new ByteArrayOutputStream();
        var ps = new PrintStream(out, true, StandardCharsets.UTF_8);
        var in = new BufferedReader(new StringReader(input == null ? "" : input));
        return new Pull(hub(), new RepoIndex(endpoint(), "hf_secret"), ps, ps, in, input != null)
                .run(Args.parse(List.of(argv), Pull.OPTIONS, Set.of()));
    }

    Path snapshot(String file) { return root.resolve("hub/models--acme--Tiny-GGUF/snapshots/" + COMMIT + "/" + file); }

    @Test
    void downloadsIntoTheHubLayout() throws Exception {
        assertEquals(0, pull(null, REPO, "README.md", "tiny-Q8_0.gguf"));
        Path snap = snapshot("tiny-Q8_0.gguf");
        assertTrue(Files.isSymbolicLink(snap));
        assertEquals(Path.of("../../blobs/" + sha256(model)), Files.readSymbolicLink(snap));
        assertArrayEquals(model, Files.readAllBytes(snap));
        assertArrayEquals(readme, Files.readAllBytes(snapshot("README.md")));
        assertEquals(COMMIT, Files.readString(root.resolve("hub/models--acme--Tiny-GGUF/refs/main")));
        assertTrue(out.toString().contains("[2/2] tiny-Q8_0.gguf  100%"), out.toString());
        assertTrue(out.toString().contains("Try it: fornax chat " + REPO + ":tiny-Q8_0.gguf"), out.toString());
        assertEquals(snap, ((Models.Llm) Models.resolve(REPO + ":Q8_0", hub())).path());
        assertEquals(List.of(), cdnAuth, "the token goes to the Hub only");

        requests.clear();   // a second pull finds both on disk and asks the Hub for nothing but the list
        assertEquals(0, pull(null, REPO, "tiny-Q8_0.gguf"));
        assertTrue(out.toString().contains("tiny-Q8_0.gguf  300 kB  (already downloaded)"), out.toString());
        assertEquals(List.of("API /api/models/" + REPO + "/revision/main"), requests);
    }

    @Test
    void withoutFilesListsTheModelsAndAsksOnATerminal() throws Exception {
        assertEquals(0, pull(null, REPO));
        assertTrue(out.toString().contains("tiny-Q8_0.gguf"), out.toString());
        assertTrue(out.toString().contains("Pick one, e.g.: fornax pull " + REPO + ":"), out.toString());
        assertTrue(HfCache.files(root.resolve("hub"), REPO).isEmpty());

        assertEquals(0, pull("2\n", REPO));   // sorted: tiny-Q4_0, tiny-Q8_0
        assertTrue(Files.isRegularFile(snapshot("tiny-Q8_0.gguf")));
        assertEquals(0, pull("\n", REPO));    // Enter cancels
    }

    @Test
    void selectsByNameGlobAndTag() throws Exception {
        var info = new RepoIndex(endpoint(), null).info(REPO, "main");
        assertEquals(COMMIT, info.commit());
        assertEquals(List.of("README.md"), Pull.select(info, null, List.of("*.md")).stream().map(RepoIndex.RepoFile::path).toList());
        assertEquals(List.of("tiny-Q4_0.gguf", "tiny-Q8_0.gguf"),
                Pull.select(info, null, List.of("tiny-*.gguf")).stream().map(RepoIndex.RepoFile::path).sorted().toList());
        assertEquals(model.length, Pull.select(info, "Q8_0", List.of()).get(0).size());
        assertTrue(Pull.select(info, null, List.of()).isEmpty());
        assertThrows(Main.Failure.class, () -> Pull.select(info, "tiny", List.of()));        // two models
        assertThrows(Main.Failure.class, () -> Pull.select(info, null, List.of("nope.bin")));
    }

    @Test
    void aRevisionInTheReference() throws Exception {
        assertEquals(0, pull(null, REPO + "@" + COMMIT + ":Q8_0"));
        assertEquals("API /api/models/" + REPO + "/revision/" + COMMIT, requests.get(0));
        assertTrue(Files.isRegularFile(snapshot("tiny-Q8_0.gguf")));
        var twice = assertThrows(Args.UsageException.class, () -> pull(null, REPO + "@" + COMMIT + ":Q8_0", "--revision", "main"));
        assertTrue(twice.getMessage().contains("give the revision once"), twice.getMessage());
        var url = assertThrows(Args.UsageException.class, () -> pull(null, "https://huggingface.co/" + REPO + "/blob/main/tiny-Q8_0.gguf"));
        assertTrue(url.getMessage().contains("use fornax pull " + REPO + ":tiny-Q8_0.gguf"), url.getMessage());
    }

    @Test
    void aTagCanNameAFileThatIsntAGguf() throws Exception {
        var info = new RepoIndex(endpoint(), null).info(REPO, "main");
        assertEquals(List.of("README.md"), Pull.select(info, "README.md", List.of()).stream().map(RepoIndex.RepoFile::path).toList());
    }

    @Test
    void aGatedRepoSaysWhatToDo() {
        var e = assertThrows(HubException.class, () -> pull(null, GATED, "tiny-Q8_0.gguf"));
        assertEquals(HubException.Reason.ACCESS_DENIED, e.reason());
        assertTrue(e.getMessage().contains("Accept") || e.getMessage().contains("accept"), e.getMessage());
        assertTrue(e.getMessage().contains(GATED), e.getMessage());
    }

    @Test
    void reportsMissingRepos() {
        var e = assertThrows(HubException.class, () -> pull(null, "acme/Nope", "x"));
        assertEquals(HubException.Reason.NOT_FOUND, e.reason());
        assertTrue(e.getMessage().contains("not found"), e.getMessage());
    }

    @Test
    void offlineListsNothingFromTheHub() {
        var e = assertThrows(HubException.class, () -> new RepoIndex(endpoint(), null, true).info(REPO, "main"));
        assertEquals(HubException.Reason.OFFLINE, e.reason());
        assertTrue(e.getMessage().contains("HF_HUB_OFFLINE"), e.getMessage());
        assertEquals(List.of(), requests);
    }

    @Test
    void anUnreachableHubSaysSo() throws Exception {
        int port;
        try (var s = new java.net.ServerSocket(0)) { port = s.getLocalPort(); }   // closed again: nothing listens
        var e = assertThrows(IOException.class, () -> new RepoIndex(URI.create("http://127.0.0.1:" + port), null).info(REPO, "main"));
        String m = Main.describe(e);
        assertTrue(m.contains("couldn't reach") && m.contains("network"), m);
    }

    static String sha256(byte[] b) { return hex("SHA-256", b); }

    static String gitSha1(byte[] b) {
        byte[] header = ("blob " + b.length + "\0").getBytes(StandardCharsets.UTF_8);
        byte[] all = new byte[header.length + b.length];
        System.arraycopy(header, 0, all, 0, header.length);
        System.arraycopy(b, 0, all, header.length, b.length);
        return hex("SHA-1", all);
    }

    static String hex(String algorithm, byte[] b) {
        try { return HexFormat.of().formatHex(MessageDigest.getInstance(algorithm).digest(b)); }
        catch (Exception e) { throw new IllegalStateException(e); }
    }
}
