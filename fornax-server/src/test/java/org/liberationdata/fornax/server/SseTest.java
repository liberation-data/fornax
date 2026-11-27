package org.liberationdata.fornax.server;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpContext;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpPrincipal;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/** Sse's keep-alives, against exchanges whose response body this test controls. */
class SseTest {
    /** An exchange whose response body is the given stream; the rest isn't used by Sse. */
    static HttpExchange exchange(OutputStream body) {
        return new HttpExchange() {
            final Headers response = new Headers();
            @Override public Headers getRequestHeaders() { return new Headers(); }
            @Override public Headers getResponseHeaders() { return response; }
            @Override public URI getRequestURI() { return URI.create("/v1/chat/completions"); }
            @Override public String getRequestMethod() { return "POST"; }
            @Override public HttpContext getHttpContext() { return null; }
            @Override public void close() {}
            @Override public InputStream getRequestBody() { return InputStream.nullInputStream(); }
            @Override public OutputStream getResponseBody() { return body; }
            @Override public void sendResponseHeaders(int code, long length) {}
            @Override public InetSocketAddress getRemoteAddress() { return null; }
            @Override public int getResponseCode() { return 200; }
            @Override public InetSocketAddress getLocalAddress() { return null; }
            @Override public String getProtocol() { return "HTTP/1.1"; }
            @Override public Object getAttribute(String name) { return null; }
            @Override public void setAttribute(String name, Object value) {}
            @Override public void setStreams(InputStream i, OutputStream o) {}
            @Override public HttpPrincipal getPrincipal() { return null; }
        };
    }

    /** A client that has stopped reading: its writes block until released. */
    static final class Stuck extends OutputStream {
        final CountDownLatch writing = new CountDownLatch(1), release = new CountDownLatch(1);
        final java.util.concurrent.atomic.AtomicInteger writes = new java.util.concurrent.atomic.AtomicInteger();
        @Override public void write(int b) { write(new byte[]{(byte) b}, 0, 1); }
        @Override public void write(byte[] b, int off, int len) {
            writes.incrementAndGet();
            writing.countDown();
            try { release.await(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
        }
    }

    @Test void aClientThatStopsReadingDoesntHoldUpAnotherStreamsKeepAlives() throws Exception {
        Sse.keepAliveMillis = 20;
        var stuck = new Stuck();
        var fine = new ByteArrayOutputStream() {
            final CountDownLatch kept = new CountDownLatch(3);
            @Override public synchronized void write(byte[] b, int off, int len) { super.write(b, off, len); kept.countDown(); }
        };
        var a = new Sse(exchange(stuck), () -> {});
        var b = new Sse(exchange(fine), () -> {});
        try {
            assertTrue(stuck.writing.await(5, TimeUnit.SECONDS), "the stuck stream's keep-alive is being written");
            // with the write on the timer thread, b would get no keep-alive until a's returned
            assertTrue(fine.kept.await(5, TimeUnit.SECONDS), "the other stream kept getting keep-alives");
            assertTrue(fine.toString(StandardCharsets.UTF_8).startsWith(": keep-alive\n\n"));
        } finally {
            stuck.release.countDown();   // before close: a's close waits for its write
            a.close();
            b.close();
            Sse.keepAliveMillis = 3_000;
        }
    }

    @Test void aStuckStreamHasOneKeepAliveUnderWayNotOnePerTick() throws Exception {
        Sse.keepAliveMillis = 20;
        var stuck = new Stuck();
        var a = new Sse(exchange(stuck), () -> {});
        try {
            assertTrue(stuck.writing.await(5, TimeUnit.SECONDS));
            Thread.sleep(300);   // some 15 ticks while the first write is stuck
            stuck.release.countDown();
            Thread.sleep(30);
            // a write queued per tick would all go out now, some 15 at once; with one under way, a tick or two's worth
            // (the bound leaves room for a slow machine pausing during the sleep)
            assertTrue(stuck.writes.get() < 8, stuck.writes.get() + " keep-alives");
        } finally {
            stuck.release.countDown();
            a.close();
            Sse.keepAliveMillis = 3_000;
        }
    }
}
