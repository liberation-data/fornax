package org.liberationdata.fornax.server;

import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.Socket;
import java.net.SocketException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@code sun.net.httpserver.maxReqTime}, as {@code fornax server} sets it: a client trickling its request is dropped,
 * while a stream that runs longer than that is not (the limit is on reading the request, not on answering it). The JDK
 * reads the property once, JVM-wide, so the server runs in a JVM of its own.
 */
class SlowClientsTest {
    /** The server in the child JVM: a chat reply streamed over 3 seconds; its port on stdout; stops when stdin closes. */
    public static void main(String[] args) throws Exception {
        var engines = new StubEngines();
        engines.pieces = Collections.nCopies(30, "word ");
        engines.pieceDelayMillis = 100;
        try (var server = FornaxServer.builder().port(0).engines(engines).log(s -> {}).build().start()) {
            System.out.println(server.port());
            System.out.flush();
            System.in.transferTo(OutputStream.nullOutputStream());
        }
    }

    @Test void aTricklingClientIsDroppedButALongStreamIsNot() throws Exception {
        String java = ProcessHandle.current().info().command().orElse("java");
        Process p = new ProcessBuilder(java, "-Dsun.net.httpserver.maxReqTime=1", "-cp", System.getProperty("java.class.path"),
                SlowClientsTest.class.getName()).redirectError(ProcessBuilder.Redirect.INHERIT).start();
        try {
            int port = Integer.parseInt(new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8)).readLine().strip());
            String host = "Host: 127.0.0.1:" + port + "\r\n";

            var stream = HttpClient.newHttpClient().sendAsync(HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + port + "/v1/chat/completions"))
                    .POST(HttpRequest.BodyPublishers.ofString("{\"stream\": true, \"messages\": [{\"role\": \"user\", \"content\": \"Go on\"}]}")).build(),
                    HttpResponse.BodyHandlers.ofString());

            // headers a byte at a time, and a body a byte at a time: each is dropped about a second in, not served
            for (String start : List.of("POST /v1/chat/completions HTTP/1.1\r\n" + host + "Content-Length: 100\r\n",
                    "POST /v1/chat/completions HTTP/1.1\r\n" + host + "Content-Length: 100\r\n\r\n{")) {
                try (Socket s = new Socket(InetAddress.getLoopbackAddress(), port)) {
                    s.setSoTimeout(100);
                    OutputStream out = s.getOutputStream();
                    InputStream in = s.getInputStream();
                    out.write(start.getBytes(StandardCharsets.UTF_8));
                    out.flush();
                    long t0 = System.nanoTime();
                    boolean dropped = false;
                    while (!dropped && System.nanoTime() - t0 < 6_000_000_000L) {
                        try {
                            out.write(' ');
                            out.flush();
                            dropped = in.read() == -1;
                        } catch (java.net.SocketTimeoutException stillOpen) {
                            // trickle on
                        } catch (SocketException reset) {
                            dropped = true;
                        }
                    }
                    double seconds = (System.nanoTime() - t0) / 1e9;
                    assertTrue(dropped, "a trickling client is dropped: " + start.strip());
                    assertTrue(seconds > 0.5, "not at once: " + seconds + "s");
                }
            }

            var r = stream.get(20, TimeUnit.SECONDS);   // began before the trickles, ends after them
            assertEquals(200, r.statusCode());
            assertTrue(r.body().strip().endsWith("data: [DONE]"), r.body());
            assertEquals(32, FornaxServerTest.events(r.body()).size(), "every piece, though the stream outlasted maxReqTime");
        } finally {
            p.getOutputStream().close();
            if (!p.waitFor(5, TimeUnit.SECONDS)) p.destroyForcibly();
        }
    }
}
