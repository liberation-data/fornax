package org.liberationdata.kiln.studio;

import com.sun.net.httpserver.HttpExchange;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;

/**
 * A reply sent as server-sent events: {@code event: name} and {@code data: json}, flushed as each is sent.
 *
 * <p>{@link #send} never throws. Tokens are sent from inside a native callback (an FFM upcall), where an exception
 * would end the JVM; once the page has gone, sending just stops, and {@link #open} says so.
 */
final class Sse implements AutoCloseable {
    private final HttpExchange exchange;
    private final OutputStream out;
    private volatile boolean open = true;

    Sse(HttpExchange exchange) throws IOException {
        this.exchange = exchange;
        var h = exchange.getResponseHeaders();
        h.set("Content-Type", "text/event-stream; charset=utf-8");
        h.set("Cache-Control", "no-cache, no-transform");
        h.set("X-Accel-Buffering", "no");
        exchange.sendResponseHeaders(200, 0);
        this.out = exchange.getResponseBody();
    }

    boolean open() { return open; }

    synchronized boolean send(String event, Object data) {
        if (!open) return false;
        try {
            out.write(("event: " + event + "\ndata: " + Json.write(data) + "\n\n").getBytes(StandardCharsets.UTF_8));
            out.flush();
            return true;
        } catch (IOException | RuntimeException e) {
            open = false;
            return false;
        }
    }

    @Override public synchronized void close() {
        open = false;
        try { out.close(); } catch (IOException ignored) {}
        exchange.close();
    }
}
