package org.liberationdata.fornax.hub;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.junit.jupiter.api.Assertions.*;

/** {@link RepoIndex}'s redirects: the Hub's own (a renamed repo) keep the token; one to another host doesn't get it. */
class RepoIndexTest {
    static final String LISTING = "{\"sha\":\"" + "a".repeat(40) + "\",\"siblings\":[{\"rfilename\":\"m.gguf\",\"size\":3}]}";
    HttpServer hub, other;
    final List<String> hubAuth = new CopyOnWriteArrayList<>(), otherAuth = new CopyOnWriteArrayList<>();

    @BeforeEach void start() throws IOException {
        InetSocketAddress any = new InetSocketAddress(InetAddress.getLoopbackAddress(), 0);
        hub = HttpServer.create(any, 0);
        other = HttpServer.create(any, 0);
        hub.createContext("/", x -> {
            String path = x.getRequestURI().getPath();
            hubAuth.add(path + " " + x.getRequestHeaders().getFirst("Authorization"));
            if (path.startsWith("/api/models/old/Name/")) redirect(x, "/api/models/new/Name/revision/main?blobs=true");
            else if (path.startsWith("/api/models/away/Name/")) redirect(x, "http://127.0.0.1:" + other.getAddress().getPort() + "/listing");
            else send(x, 200, LISTING);
        });
        other.createContext("/", x -> {
            String auth = x.getRequestHeaders().getFirst("Authorization");
            if (auth != null) otherAuth.add(auth);
            send(x, 200, LISTING);
        });
        hub.start();
        other.start();
    }

    @AfterEach void stop() { hub.stop(0); other.stop(0); }

    static void redirect(HttpExchange x, String to) throws IOException {
        x.getResponseHeaders().set("Location", to);
        x.sendResponseHeaders(307, -1);
        x.close();
    }

    static void send(HttpExchange x, int code, String body) throws IOException {
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        x.sendResponseHeaders(code, b.length);
        x.getResponseBody().write(b);
        x.close();
    }

    RepoIndex index() { return new RepoIndex(URI.create("http://127.0.0.1:" + hub.getAddress().getPort()), "hf_secret"); }

    @Test void aRenamedRepoIsFollowedWithTheToken() throws IOException {
        assertEquals("m.gguf", index().info("old/Name", "main").files().get(0).path());
        assertEquals(List.of("/api/models/old/Name/revision/main Bearer hf_secret", "/api/models/new/Name/revision/main Bearer hf_secret"), hubAuth);
    }

    @Test void theTokenDoesntFollowARedirectToAnotherHost() throws IOException {
        assertEquals("m.gguf", index().info("away/Name", "main").files().get(0).path());
        assertEquals(List.of(), otherAuth);
    }

    @Test void offlineSaysWhatToDoWithoutNamingACommand() {
        var e = assertThrows(HubException.class, () -> new RepoIndex(URI.create("http://127.0.0.1:1"), null, true).info("a/b", "main"));
        assertEquals(HubException.Reason.OFFLINE, e.reason());
        assertFalse(e.getMessage().contains("fornax "), e.getMessage());
    }
}
