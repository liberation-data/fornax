package org.liberationdata.fornax.llama;

import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelFolders;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.junit.jupiter.api.Assertions.*;

/** How LlamaModel turns a model reference into files, against a Hub on localhost: no native library needed. */
class LlamaLocateTest {
    @TempDir Path dir;

    @Test
    void aReferenceToAFileThatIsntAGgufIsRefusedBeforeItDownloads() throws Exception {
        List<String> requests = new CopyOnWriteArrayList<>();
        HttpServer server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.createContext("/", ex -> {
            requests.add(ex.getRequestMethod() + " " + ex.getRequestURI().getPath());
            byte[] body = ex.getRequestURI().getPath().equals("/api/models/ggerganov/whisper.cpp/revision/main")
                    ? """
                      {"sha": "0123456789abcdef0123456789abcdef01234567",
                       "siblings": [{"rfilename": "ggml-base.en.bin", "size": 147951465}]}""".getBytes(StandardCharsets.UTF_8)
                    : new byte[0];
            ex.sendResponseHeaders(body.length > 0 ? 200 : 404, body.length > 0 ? body.length : -1);
            ex.getResponseBody().write(body);
            ex.close();
        });
        server.start();
        try {
            FornaxHub hub = FornaxHub.builder().cacheRoot(dir).modelFolders(ModelFolders.none()).token(null).offline(false)
                    .endpoint(URI.create("http://127.0.0.1:" + server.getAddress().getPort())).build();
            var e = assertThrows(IllegalArgumentException.class, () ->
                    LlamaModel.locate("ggerganov/whisper.cpp:ggml-base.en.bin", null, LlamaModel.Mode.CHAT, hub));
            assertTrue(e.getMessage().contains("isn't a GGUF language model"), e.getMessage());
            assertEquals(List.of("GET /api/models/ggerganov/whisper.cpp/revision/main"), requests);   // the listing only
        } finally {
            server.stop(0);
        }
    }

    @Test
    void aReferenceToAGgufIsDownloadedAndItsPathReturned() throws Exception {
        String commit = "0123456789abcdef0123456789abcdef01234567";
        byte[] content = "not really a gguf".getBytes(StandardCharsets.UTF_8);
        String etag = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(content));
        HttpServer server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.createContext("/", ex -> {
            String path = ex.getRequestURI().getPath();
            byte[] body = new byte[0];
            int code = 404;
            if (path.equals("/api/models/some/model-GGUF/revision/main")) {
                code = 200;
                body = ("{\"sha\": \"" + commit + "\", \"siblings\": [{\"rfilename\": \"model-Q8_0.gguf\", \"size\": "
                        + content.length + "}, {\"rfilename\": \"model-Q4_K_M.gguf\", \"size\": 1}]}").getBytes(StandardCharsets.UTF_8);
            } else if (path.matches("/some/model-GGUF/resolve/(main|" + commit + ")/model-Q8_0.gguf")) {
                code = 200;
                ex.getResponseHeaders().set("X-Repo-Commit", commit);
                ex.getResponseHeaders().set("ETag", "\"" + etag + "\"");
                if (ex.getRequestMethod().equals("HEAD")) ex.getResponseHeaders().set("Content-Length", String.valueOf(content.length));
                else body = content;
            }
            boolean head = ex.getRequestMethod().equals("HEAD");
            ex.sendResponseHeaders(code, head || body.length == 0 ? -1 : body.length);
            if (!head) ex.getResponseBody().write(body);
            ex.close();
        });
        server.start();
        try {
            FornaxHub hub = FornaxHub.builder().cacheRoot(dir).modelFolders(ModelFolders.none()).token(null).offline(false)
                    .endpoint(URI.create("http://127.0.0.1:" + server.getAddress().getPort())).build();
            String[] files = LlamaModel.locate("some/model-GGUF:q8_0", null, LlamaModel.Mode.CHAT, hub);
            assertTrue(files[0].endsWith("model-Q8_0.gguf"), files[0]);
            assertArrayEquals(content, Files.readAllBytes(Path.of(files[0])));
            assertNull(files[1]);   // a repo reference brings no mmproj
            assertEquals("given.mmproj", LlamaModel.locate("some/model-GGUF:Q8_0", "given.mmproj", LlamaModel.Mode.CHAT, hub)[1]);
        } finally {
            server.stop(0);
        }
    }
}
