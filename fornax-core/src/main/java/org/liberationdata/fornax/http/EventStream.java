package org.liberationdata.fornax.http;

import com.sun.net.httpserver.HttpExchange;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * The writer under a server-sent-events reply, shared by fornax-server and fornax-studio-server (each formats its own
 * events): blocks of text, each written and flushed whole. The status and headers go out with the first block, or at
 * once with {@link #start}.
 *
 * <p>While nothing else is sent (a model loading, a reasoning model thinking, a video decoding), a
 * {@code : keep-alive} comment goes out every keepAliveMillis, which clients skip; it keeps proxies and clients that
 * give up on a silent reply waiting. One timer thread serves every stream, and only decides that a keep-alive is due:
 * a virtual thread writes it, at most one per stream at a time, so a client that reads slowly holds up its own stream
 * and no one else's.
 *
 * <p>A write that fails means the client has gone: the stream is closed to writes from then on, and onGone runs, once.
 */
public final class EventStream implements AutoCloseable {
    private static final ScheduledExecutorService TIMER = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "fornax-sse-keepalive");
        t.setDaemon(true);
        return t;
    });

    private final HttpExchange exchange;
    private final Runnable onGone;
    private final ScheduledFuture<?> keepAlive;
    private final AtomicBoolean keepingAlive = new AtomicBoolean();   // a keep-alive write under way: at most one
    private OutputStream out;     // null until the headers are sent
    private boolean open = true;
    private volatile long lastSent = System.nanoTime();

    /**
     * @param keepAliveMillis how long the stream may be silent before a keep-alive goes out
     * @param onGone          runs once if a write fails because the client has gone (for example, to stop the work)
     */
    public EventStream(HttpExchange exchange, long keepAliveMillis, Runnable onGone) {
        this.exchange = exchange;
        this.onGone = onGone;
        this.keepAlive = TIMER.scheduleAtFixedRate(() -> {
            if (System.nanoTime() - lastSent < TimeUnit.MILLISECONDS.toNanos(keepAliveMillis) || !keepingAlive.compareAndSet(false, true)) return;
            Thread.ofVirtual().name("fornax-sse-keepalive-write").start(() -> {
                try { write(": keep-alive\n\n"); } finally { keepingAlive.set(false); }
            });
        }, keepAliveMillis, keepAliveMillis, TimeUnit.MILLISECONDS);
    }

    /** Sends the status and headers now, rather than with the first block. */
    public synchronized void start() throws IOException {
        try {
            headers();
        } catch (IOException | RuntimeException e) {
            open = false;
            keepAlive.cancel(false);
            throw e;
        }
    }

    /**
     * Gives the stream up before anything has been sent, so that the reply can be an ordinary one (an error with its
     * status). False once the headers have gone: then a failure can only be reported in the stream.
     */
    public synchronized boolean abandon() {
        if (out != null) return false;
        open = false;
        keepAlive.cancel(false);
        return true;
    }

    /** Writes and flushes block; false if the stream is closed or the client has gone. */
    public synchronized boolean write(String block) {
        if (!open) return false;
        try {
            headers();
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

    /** Whether writes still go out: false once closed, abandoned, or the client has gone. */
    public synchronized boolean open() { return open; }

    private void headers() throws IOException {
        if (out != null) return;
        var h = exchange.getResponseHeaders();
        h.set("Content-Type", "text/event-stream; charset=utf-8");
        h.set("Cache-Control", "no-cache, no-transform");
        h.set("X-Accel-Buffering", "no");
        exchange.sendResponseHeaders(200, 0);
        out = exchange.getResponseBody();
    }

    /** Ends the stream: no more keep-alives, and the reply's body is closed (the exchange is the caller's). */
    @Override public synchronized void close() {
        keepAlive.cancel(false);
        open = false;
        if (out != null) {
            try { out.close(); } catch (IOException ignored) {}
        }
    }
}
