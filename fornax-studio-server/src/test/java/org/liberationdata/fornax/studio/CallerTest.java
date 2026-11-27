package org.liberationdata.fornax.studio;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelFolders;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelSet.Part;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.Socket;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Who may call the server: not a web page in the user's browser (an Origin), not a DNS-rebound name (a foreign Host),
 * and with a secret set, only what carries it. Requests are written by hand: Java's HttpClient won't send another Host.
 */
class CallerTest {
    static final String SECRET = "0123456789abcdef-launch-secret";

    @TempDir Path dir;
    private Studio studio;
    private StudioServer server;
    /** Hubs the pulls asked for: none, for every refused request. */
    private final AtomicInteger hubs = new AtomicInteger();

    StudioServer start(String secret) throws IOException {
        Path cache = dir.resolve("hub");
        List<ModelSet> sets = List.of(ModelSet.of("tiny", Kind.SONG, "Tiny", "acestep.cpp", Part.of("dit", "o/tiny", "dit.gguf")));
        Pulls pulls = new Pulls(sets, token -> {
            hubs.incrementAndGet();
            return FornaxHub.builder().cacheRoot(cache).modelFolders(ModelFolders.none()).endpoint(URI.create("http://127.0.0.1:9"))
                    .token(token).offline(true).catalogue(sets).build();
        }, URI.create("http://127.0.0.1:9"));
        studio = new Studio(new Catalog(cache), 1L << 30, null, EnumSet.allOf(Catalog.Kind.class), dir.resolve("outputs"));
        server = new StudioServer(studio, 0, null, pulls, secret);
        server.start();
        return server;
    }

    @AfterEach void stop() {
        if (server != null) server.stop();
        if (studio != null) studio.close();
    }

    /** The status line and body of a request written as it is given: method, path, then header lines. */
    String send(String method, String path, String body, String... headers) throws IOException {
        try (Socket s = new Socket(InetAddress.getLoopbackAddress(), server.port())) {
            StringBuilder req = new StringBuilder(method + " " + path + " HTTP/1.1\r\n");
            for (String h : headers) req.append(h).append("\r\n");
            byte[] b = body == null ? new byte[0] : body.getBytes(StandardCharsets.UTF_8);
            req.append("Content-Length: ").append(b.length).append("\r\nConnection: close\r\n\r\n");
            OutputStream out = s.getOutputStream();
            out.write(req.toString().getBytes(StandardCharsets.UTF_8));
            out.write(b);
            out.flush();
            InputStream in = s.getInputStream();
            String reply = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            return reply.substring(0, reply.indexOf("\r\n")) + "\n" + reply.substring(reply.indexOf("\r\n\r\n") + 4);
        }
    }

    String host() { return "Host: 127.0.0.1:" + server.port(); }

    @Test void theStudiosOwnRequestsAreServed() throws Exception {
        start(null);
        for (String h : List.of(host(), "Host: localhost:" + server.port(), "Host: LOCALHOST:" + server.port(), "Host: [::1]:" + server.port()))
            assertTrue(send("GET", "/api/catalogue", null, h).startsWith("HTTP/1.1 200"), h);
    }

    @Test void aWebPagesRequestIsRefused() throws Exception {
        start(null);
        // a "simple" POST a page can send with no preflight: text/plain, from another origin
        String reply = send("POST", "/api/models/pull", "{\"id\": \"tiny\"}", host(), "Origin: https://evil.example",
                "Content-Type: text/plain");
        assertTrue(reply.startsWith("HTTP/1.1 403"), reply);
        assertTrue(reply.contains("Origin"), reply);
        assertFalse(reply.contains("evil.example"), "what was sent isn't repeated");
        // even one claiming to be the studio's own page: the studio's proxy sends no Origin
        assertTrue(send("POST", "/api/hub/token", "{\"token\": \"hf_x\"}", host(), "Origin: http://localhost:5190").startsWith("HTTP/1.1 403"));
        assertTrue(send("GET", "/api/catalogue", null, host(), "Origin: null").startsWith("HTTP/1.1 403"));
        assertEquals(0, hubs.get(), "nothing reached the hub");
        assertEquals(false, Json.object(send("GET", "/api/hub", null, host()).split("\n", 2)[1]).get("tokenSet"));
    }

    @Test void aRebindingHostIsRefused() throws Exception {
        start(null);
        for (String h : List.of("Host: attacker.example:" + server.port(), "Host: 127.0.0.1", "Host: 127.0.0.1:1",
                "Host: 127.0.0.1.attacker.example:" + server.port(), "Host: 0.0.0.0:" + server.port(), "Host: 127.1:" + server.port(),
                "Host: localhost.:" + server.port(), "Host: :" + server.port(), "Host: [0:0:0:0:0:0:0:1]:" + server.port(), "Host:")) {
            String reply = send("POST", "/api/models/pull", "{\"id\": \"tiny\"}", h);
            assertTrue(reply.startsWith("HTTP/1.1 403"), h + ": " + reply);
            assertTrue(reply.contains("wrong Host"), reply);
        }
        assertTrue(send("GET", "/api/health", null, "Host: attacker.example:" + server.port()).startsWith("HTTP/1.1 403"));
        assertTrue(send("GET", "/api/outputs/a.png", null, "Host: attacker.example:" + server.port()).startsWith("HTTP/1.1 403"));
        // No Host at all: some JDKs' HttpServer answers an HTTP/1.1 request without one itself (a 400) before any handler
        // runs, and others pass it on, to be refused here. Either way it is refused, and the hub is never asked (below).
        String none = send("POST", "/api/models/pull", "{\"id\": \"tiny\"}");
        assertTrue(none.matches("(?s)HTTP/1\\.1 4\\d\\d.*") && (!none.startsWith("HTTP/1.1 403") || none.contains("wrong Host")),
                "no Host at all: " + none);
        // refused before the method is looked at: a HEAD, and a method the route doesn't take
        String bad = "Host: attacker.example:" + server.port();
        assertTrue(send("HEAD", "/api/catalogue", null, bad).startsWith("HTTP/1.1 403"));
        assertTrue(send("DELETE", "/api/catalogue", null, bad).startsWith("HTTP/1.1 403"));
        assertEquals(0, hubs.get());
    }

    @Test void anEmptyOrRepeatedOriginIsStillAnOrigin() throws Exception {
        start(null);
        assertTrue(send("POST", "/api/models/pull", "{\"id\": \"tiny\"}", host(), "Origin:").startsWith("HTTP/1.1 403"));
        assertTrue(send("POST", "/api/models/pull", "{\"id\": \"tiny\"}", host(), "Origin: ", "Origin: https://evil.example")
                .startsWith("HTTP/1.1 403"));
        assertEquals(0, hubs.get());
    }

    @Test void aBlankSecretIsNone() throws Exception {
        start("  ");
        assertTrue(send("GET", "/api/catalogue", null, host()).startsWith("HTTP/1.1 200"));
    }

    @Test void withASecretEveryRouteButHealthNeedsIt() throws Exception {
        start(SECRET);
        assertTrue(send("GET", "/api/health", null, host()).startsWith("HTTP/1.1 200"), "the app waits on health before it has a page");
        for (String[] r : List.of(new String[]{"GET", "/api/catalogue", null}, new String[]{"GET", "/api/models", null},
                new String[]{"POST", "/api/hub/token", "{\"token\": \"hf_x\"}"}, new String[]{"POST", "/api/models/pull", "{\"id\": \"tiny\"}"},
                new String[]{"GET", "/api/outputs/a.png", null})) {
            assertTrue(send(r[0], r[1], r[2], host()).startsWith("HTTP/1.1 403"), r[1] + " without the secret");
            String wrong = send(r[0], r[1], r[2], host(), StudioServer.SECRET_HEADER + ": " + SECRET + "x");
            assertTrue(wrong.startsWith("HTTP/1.1 403"), r[1] + " with a wrong secret");
            assertFalse(wrong.contains(SECRET), "the secret isn't repeated");
            assertTrue(send(r[0], r[1], r[2], host(), StudioServer.SECRET_HEADER + ": " + SECRET.substring(1)).startsWith("HTTP/1.1 403"),
                    r[1] + " with a shorter secret");
            assertTrue(send(r[0], r[1], r[2], host(), StudioServer.SECRET_HEADER + ":").startsWith("HTTP/1.1 403"), r[1] + " with an empty one");
        }
        // HEAD of health needs no secret either
        assertTrue(send("HEAD", "/api/health", null, host()).startsWith("HTTP/1.1 200"));
        assertEquals(0, hubs.get());
        String ok = send("GET", "/api/catalogue", null, host(), StudioServer.SECRET_HEADER + ": " + SECRET);
        assertTrue(ok.startsWith("HTTP/1.1 200"), ok);
        // with the secret, an Origin is still refused
        assertTrue(send("GET", "/api/catalogue", null, host(), StudioServer.SECRET_HEADER + ": " + SECRET, "Origin: https://evil.example")
                .startsWith("HTTP/1.1 403"));
    }

    @Test void theTokenCanBeSetButNeverReadBack() throws Exception {
        start(null);
        assertTrue(send("POST", "/api/hub/token", "{\"token\": \"hf_secret_token\"}", host()).startsWith("HTTP/1.1 200"));
        for (String[] r : List.of(new String[]{"GET", "/api/hub"}, new String[]{"GET", "/api/hub/token"}, new String[]{"GET", "/api/catalogue"},
                new String[]{"GET", "/api/models"}, new String[]{"GET", "/api/health"})) {
            String reply = send(r[0], r[1], null, host());
            assertFalse(reply.contains("hf_secret_token"), r[1] + ": " + reply);
        }
        assertTrue(send("GET", "/api/hub/token", null, host()).startsWith("HTTP/1.1 405"), "the token route is POST only");
    }
}
