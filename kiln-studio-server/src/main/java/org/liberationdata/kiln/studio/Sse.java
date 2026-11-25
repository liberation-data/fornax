package org.liberationdata.kiln.studio;

import com.sun.net.httpserver.HttpExchange;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/**
 * A reply sent as server-sent events: {@code event: name} and {@code data: json}, flushed as each is sent.
 *
 * <p>{@link #send} never throws. Tokens are sent from inside a native callback (an FFM upcall), where an exception
 * would end the JVM; once the page has gone, sending just stops, and {@link #open} says so.
 *
 * <p>A comment is sent every {@link #keepAliveMillis} while nothing else is: a model loading, or a video's decode, can
 * say nothing for minutes, and the studio's proxy (Node's fetch) gives up on a reply silent for five.
 */
final class Sse implements AutoCloseable {
    private final HttpExchange exchange;
    private final OutputStream out;
    private volatile boolean open = true;
    private volatile long lastSent = System.nanoTime();
    private final ScheduledFuture<?> keepAlive;

    /** How often a silent stream is kept alive (a test shortens it). */
    static volatile long keepAliveMillis = 15_000;
    private static final ScheduledExecutorService TIMER = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "kiln-studio-sse-keepalive");
        t.setDaemon(true);
        return t;
    });

    Sse(HttpExchange exchange) throws IOException {
        this.exchange = exchange;
        var h = exchange.getResponseHeaders();
        h.set("Content-Type", "text/event-stream; charset=utf-8");
        h.set("Cache-Control", "no-cache, no-transform");
        h.set("X-Accel-Buffering", "no");
        exchange.sendResponseHeaders(200, 0);
        this.out = exchange.getResponseBody();
        long every = keepAliveMillis;
        this.keepAlive = TIMER.scheduleAtFixedRate(() -> {
            if (System.nanoTime() - lastSent >= TimeUnit.MILLISECONDS.toNanos(every)) write(": keep-alive\n\n");
        }, every, every, TimeUnit.MILLISECONDS);
    }

    boolean open() { return open; }

    boolean send(String event, Object data) {
        return write("event: " + event + "\ndata: " + Json.write(data) + "\n\n");
    }

    private synchronized boolean write(String block) {
        if (!open) return false;
        try {
            out.write(block.getBytes(StandardCharsets.UTF_8));
            out.flush();
            lastSent = System.nanoTime();
            return true;
        } catch (IOException | RuntimeException e) {
            open = false;
            keepAlive.cancel(false);
            return false;
        }
    }

    @Override public synchronized void close() {
        open = false;
        keepAlive.cancel(false);
        try { out.close(); } catch (IOException ignored) {}
        exchange.close();
    }
}
