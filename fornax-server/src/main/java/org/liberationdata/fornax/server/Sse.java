package org.liberationdata.fornax.server;

import com.sun.net.httpserver.HttpExchange;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CancellationException;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A streamed reply as OpenAI sends one: server-sent events of {@code data: <json>}, ending with {@code data: [DONE]}.
 *
 * <p>The status and headers go out with the first event, so a request that fails before anything is streamed (a model
 * that won't load) still gets an error status. While nothing else is sent (a model loading, a reasoning model
 * thinking), a {@code : keep-alive} comment is sent every {@link #keepAliveMillis}, which clients skip. The one timer
 * thread only decides that one is due; the write itself runs on a virtual thread, so a client that reads slowly holds up
 * its own stream and no one else's keep-alives. A write that
 * fails means the client has gone: {@code onGone} runs once (it stops the generation), and {@link #data} throws a
 * CancellationException, which from a model's callback stops the generation too.
 */
final class Sse implements AutoCloseable {
    private final HttpExchange exchange;
    private final Runnable onGone;
    private OutputStream out;     // null until the first write
    private boolean open = true;
    private volatile long lastSent = System.nanoTime();
    private final ScheduledFuture<?> keepAlive;
    private final AtomicBoolean keepingAlive = new AtomicBoolean();   // a keep-alive write under way: at most one

    /** How often a silent stream is kept alive, and a client that has gone noticed (a test shortens it). */
    static volatile long keepAliveMillis = 3_000;
    private static final ScheduledExecutorService TIMER = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "fornax-server-sse-keepalive");
        t.setDaemon(true);
        return t;
    });

    Sse(HttpExchange exchange, Runnable onGone) {
        this.exchange = exchange;
        this.onGone = onGone;
        long every = keepAliveMillis;
        this.keepAlive = TIMER.scheduleAtFixedRate(() -> {
            if (System.nanoTime() - lastSent < TimeUnit.MILLISECONDS.toNanos(every) || !keepingAlive.compareAndSet(false, true)) return;
            Thread.ofVirtual().name("fornax-server-sse-keepalive-write").start(() -> {
                try { write(": keep-alive\n\n"); } finally { keepingAlive.set(false); }
            });
        }, every, every, TimeUnit.MILLISECONDS);
    }

    /**
     * Gives up the stream before anything has been sent, so that the reply can be an ordinary one (an error with its
     * status). False once something has been sent: then a failure can only be reported in the stream.
     */
    synchronized boolean abandon() {
        if (out != null) return false;
        open = false;
        keepAlive.cancel(false);
        return true;
    }

    /** One event; throws a CancellationException if the client has gone. */
    void data(Object json) {
        if (!write("data: " + Json.write(json) + "\n\n")) throw new CancellationException("the client has gone");
    }

    /** The end of the stream. */
    void done() { write("data: [DONE]\n\n"); }

    private synchronized boolean write(String block) {
        if (!open) return false;
        try {
            if (out == null) {
                var h = exchange.getResponseHeaders();
                h.set("Content-Type", "text/event-stream; charset=utf-8");
                h.set("Cache-Control", "no-cache, no-transform");
                h.set("X-Accel-Buffering", "no");
                exchange.sendResponseHeaders(200, 0);
                out = exchange.getResponseBody();
            }
            out.write(block.getBytes(StandardCharsets.UTF_8));
            out.flush();
            lastSent = System.nanoTime();
            return true;
        } catch (IOException | RuntimeException e) {
            open = false;
            keepAlive.cancel(false);
            onGone.run();
            return false;
        }
    }

    @Override public synchronized void close() {
        keepAlive.cancel(false);
        open = false;
        if (out != null) {
            try { out.close(); } catch (IOException ignored) {}
        }
    }
}
