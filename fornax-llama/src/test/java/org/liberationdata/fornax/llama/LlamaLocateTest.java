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
import java.nio.file.Path;
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
}
