package org.liberationdata.fornax.http;

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

/** EventStream's keep-alives and headers, against exchanges whose response body this test controls. */
class EventStreamTest {
    /** An exchange whose response body is the given stream; the rest isn't used by EventStream. */
    static HttpExchange exchange(OutputStream body) { return exchange(body, null); }

    /** {@link #exchange(OutputStream)}, whose headers fail to go out with headersFail unless that's null. */
    static HttpExchange exchange(OutputStream body, java.io.IOException headersFail) {
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
            @Override public void sendResponseHeaders(int code, long length) throws java.io.IOException { if (headersFail != null) throw headersFail; }
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
        var stuck = new Stuck();
        var fine = new ByteArrayOutputStream() {
            final CountDownLatch kept = new CountDownLatch(3);
            @Override public synchronized void write(byte[] b, int off, int len) { super.write(b, off, len); kept.countDown(); }
        };
        var a = new EventStream(exchange(stuck), 20, () -> {});
        var b = new EventStream(exchange(fine), 20, () -> {});
        try {
            assertTrue(stuck.writing.await(5, TimeUnit.SECONDS), "the stuck stream's keep-alive is being written");
            // with the write on the timer thread, b would get no keep-alive until a's returned
            assertTrue(fine.kept.await(5, TimeUnit.SECONDS), "the other stream kept getting keep-alives");
            assertTrue(fine.toString(StandardCharsets.UTF_8).startsWith(": keep-alive\n\n"));
        } finally {
            stuck.release.countDown();   // before close: a's close waits for its write
            a.close();
            b.close();
        }
    }

    @Test void aKeepAliveThatFailsStillNoticesTheClientHasGone() throws Exception {
        var gone = new CountDownLatch(1);
        var a = new EventStream(exchange(new OutputStream() {
            @Override public void write(int b) throws java.io.IOException { throw new java.io.IOException("broken pipe"); }
        }), 20, gone::countDown);
        try {
            assertTrue(gone.await(5, TimeUnit.SECONDS), "onGone ran from the keep-alive's own thread");
            assertFalse(a.write("data: 1\n\n"));
            assertFalse(a.open());
        } finally {
            a.close();
        }
    }

    @Test void aStuckStreamHasOneKeepAliveUnderWayNotOnePerTick() throws Exception {
        var stuck = new Stuck();
        var a = new EventStream(exchange(stuck), 20, () -> {});
        try {
            assertTrue(stuck.writing.await(5, TimeUnit.SECONDS));
            Thread.sleep(300);   // some 15 ticks while the first write is stuck
            long released = System.nanoTime();
            stuck.release.countDown();
            Thread.sleep(30);
            int writes = stuck.writes.get();
            long since = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - released);
            // a write queued per tick would all go out now, some 15 at once; with one under way, a tick or two's worth
            // plus one per tick since (a slow machine may wake from the sleep late, and the writes don't block now)
            assertTrue(writes < 8 + since / 20, writes + " keep-alives in " + since + " ms");
        } finally {
            stuck.release.countDown();
            a.close();
        }
    }

    @Test void headersThatFailToGoOutThrowFromStartAndEndTheStream() throws Exception {
        var body = new ByteArrayOutputStream();
        var gone = new CountDownLatch(1);
        var a = new EventStream(exchange(body, new java.io.IOException("reset")), 20, gone::countDown);
        try {
            assertThrows(java.io.IOException.class, a::start);   // as the studio's Sse constructor threw before
            assertFalse(a.open());
            Thread.sleep(100);   // some 5 ticks: no keep-alive goes out, and onGone isn't run for this
            assertEquals(0, body.size());
            assertEquals(1, gone.getCount());
        } finally {
            a.close();
        }
    }
}
