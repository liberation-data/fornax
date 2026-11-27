package org.liberationdata.fornax.server;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Who may call the server: not a DNS-rebound name (a foreign Host) while it is on loopback, not a web page (an Origin)
 * unless its origin is allowed, and with an API key, only what carries it. Requests are written by hand: Java's
 * HttpClient won't send another Host.
 */
class GuardTest {
    static final String CHAT = "{\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]}";
    private final StubEngines engines = new StubEngines();
    private FornaxServer server;

    FornaxServer start(FornaxServer.Builder b) throws IOException {
        server = b.port(0).engines(engines).log(s -> {}).build().start();
        return server;
    }

    @AfterEach void stop() { if (server != null) server.close(); }

    /** The status line, headers and body of a request written as it is given: method, path, then header lines. */
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
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    String host() { return "Host: 127.0.0.1:" + server.port(); }

    static int status(String reply) { return Integer.parseInt(reply.substring(9, 12)); }

    static String header(String reply, String name) {
        for (String line : reply.substring(0, reply.indexOf("\r\n\r\n")).split("\r\n"))
            if (line.toLowerCase(Locale.ROOT).startsWith(name.toLowerCase(Locale.ROOT) + ":")) return line.substring(name.length() + 1).strip();
        return null;
    }

    @Test void itsOwnLoopbackNamesAreServed() throws Exception {
        start(FornaxServer.builder());
        for (String h : List.of(host(), "Host: localhost:" + server.port(), "Host: LOCALHOST:" + server.port(), "Host: [::1]:" + server.port()))
            assertEquals(200, status(send("POST", "/v1/chat/completions", CHAT, h)), h);
    }

    @Test void aRebindingHostIsRefused() throws Exception {
        start(FornaxServer.builder());
        int p = server.port();
        for (String h : List.of("Host: attacker.example:" + p, "Host: 127.0.0.1", "Host: 127.0.0.1:1", "Host: 127.0.0.1.attacker.example:" + p,
                "Host: 0.0.0.0:" + p, "Host: 127.1:" + p, "Host: localhost.:" + p, "Host: :" + p, "Host:")) {
            String reply = send("POST", "/v1/chat/completions", CHAT, h);
            assertEquals(403, status(reply), h + ": " + reply);
            assertTrue(reply.contains("wrong Host"), reply);
            assertTrue(reply.contains("\"permission_error\""), reply);
        }
        assertEquals(403, status(send("GET", "/v1/models", null, "Host: attacker.example:" + p)));
        assertEquals(403, status(send("POST", "/v1/chat/completions", CHAT)), "no Host at all");
        assertEquals(403, status(send("POST", "/v1/chat/completions", CHAT, host(), "Host: attacker.example:" + p)), "two Hosts");
        assertEquals(403, status(send("DELETE", "/v1/nothing", null, "Host: attacker.example:" + p)), "before routing");
        assertTrue(engines.requests.isEmpty(), "nothing reached a model");
    }

    @Test void onAnotherAddressAnyHostIsServed() throws Exception {
        start(FornaxServer.builder().host("0.0.0.0"));
        assertEquals(200, status(send("POST", "/v1/chat/completions", CHAT, "Host: fornax.lan:" + server.port())));
    }

    @Test void aWebPagesRequestIsRefused() throws Exception {
        start(FornaxServer.builder());
        // a "simple" POST a page can send with no preflight: text/plain, from another origin
        String reply = send("POST", "/v1/chat/completions", CHAT, host(), "Origin: https://evil.example", "Content-Type: text/plain");
        assertEquals(403, status(reply));
        assertTrue(reply.contains("--cors"), reply);
        assertFalse(reply.contains("evil.example"), "what was sent isn't repeated");
        assertNull(header(reply, "Access-Control-Allow-Origin"));
        for (String o : List.of("Origin: null", "Origin:", "Origin: http://localhost:" + server.port()))
            assertEquals(403, status(send("POST", "/v1/chat/completions", CHAT, host(), o)), o);
        assertEquals(403, status(send("OPTIONS", "/v1/chat/completions", null, host(), "Origin: https://evil.example",
                "Access-Control-Request-Method: POST")), "no preflight answer either");
        assertTrue(engines.requests.isEmpty());
    }

    @Test void anAllowedOriginGetsCorsHeadersAndItsPreflightAnswered() throws Exception {
        start(FornaxServer.builder().corsOrigin("https://app.example/"));
        String pre = send("OPTIONS", "/v1/chat/completions", null, host(), "Origin: https://app.example",
                "Access-Control-Request-Method: POST", "Access-Control-Request-Headers: authorization, content-type, x-stainless-os");
        assertEquals(204, status(pre), pre);
        assertEquals("https://app.example", header(pre, "Access-Control-Allow-Origin"));
        assertTrue(header(pre, "Access-Control-Allow-Methods").contains("POST"));
        assertEquals("authorization, content-type, x-stainless-os", header(pre, "Access-Control-Allow-Headers"));
        String reply = send("POST", "/v1/chat/completions", CHAT, host(), "Origin: https://app.example");
        assertEquals(200, status(reply));
        assertEquals("https://app.example", header(reply, "Access-Control-Allow-Origin"));
        assertEquals("Origin", header(reply, "Vary"));
        // another origin, and two Origins, are still refused
        assertEquals(403, status(send("POST", "/v1/chat/completions", CHAT, host(), "Origin: https://other.example")));
        assertEquals(403, status(send("POST", "/v1/chat/completions", CHAT, host(), "Origin: https://app.example", "Origin: https://other.example")));
    }

    @Test void anyOriginWithAStar() throws Exception {
        start(FornaxServer.builder().corsOrigin("*"));
        String reply = send("GET", "/v1/models", null, host(), "Origin: https://anywhere.example");
        assertEquals(200, status(reply));
        assertEquals("https://anywhere.example", header(reply, "Access-Control-Allow-Origin"));
    }

    @Test void anApiKeyIsRequiredWhenSet() throws Exception {
        String key = "sk-fornax-0123456789abcdef";
        start(FornaxServer.builder().apiKey(key).corsOrigin("https://app.example"));
        for (String auth : List.of("", "Authorization: Bearer wrong", "Authorization: Bearer " + key.substring(0, 10),
                "Authorization: Bearer ", "Authorization: Basic " + key, "Authorization: " + key)) {
            String[] headers = auth.isEmpty() ? new String[]{host()} : new String[]{host(), auth};
            String reply = send("POST", "/v1/chat/completions", CHAT, headers);
            assertEquals(401, status(reply), auth + ": " + reply);
            assertTrue(reply.contains("\"invalid_api_key\""), reply);
            assertFalse(reply.contains(key), "the key isn't repeated");
        }
        assertEquals(401, status(send("GET", "/v1/models", null, host())));
        assertEquals(401, status(send("GET", "/health", null, host())));
        assertEquals(200, status(send("POST", "/v1/chat/completions", CHAT, host(), "Authorization: Bearer " + key)));
        assertEquals(200, status(send("POST", "/v1/chat/completions", CHAT, host(), "Authorization: bearer " + key)));
        // a preflight carries no credentials, so it needs none; the request after it does
        assertEquals(204, status(send("OPTIONS", "/v1/chat/completions", null, host(), "Origin: https://app.example",
                "Access-Control-Request-Method: POST")));
        assertEquals(401, status(send("POST", "/v1/chat/completions", CHAT, host(), "Origin: https://app.example")));
        assertEquals(1 + 1, engines.requests.size());
    }

    @Test void theHostIsCheckedBeforeTheKey() throws Exception {
        start(FornaxServer.builder().apiKey("k"));
        assertEquals(403, status(send("POST", "/v1/chat/completions", CHAT, "Host: attacker.example:" + server.port(), "Authorization: Bearer k")));
    }
}
