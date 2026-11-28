package org.liberationdata.fornax.server;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpContext;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpPrincipal;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Sse over its EventStream (whose keep-alives EventStreamTest covers), against an exchange whose response body this
 * test controls.
 */
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

    @Test void aKeepAliveThatFailsStillNoticesTheClientHasGone() throws Exception {
        Sse.keepAliveMillis = 20;
        var gone = new CountDownLatch(1);
        var a = new Sse(exchange(new OutputStream() {
            @Override public void write(int b) throws java.io.IOException { throw new java.io.IOException("broken pipe"); }
        }), gone::countDown);
        try {
            assertTrue(gone.await(5, TimeUnit.SECONDS), "onGone ran from the keep-alive's own thread");
            assertThrows(java.util.concurrent.CancellationException.class, () -> a.data(Json.obj("x", 1)));
        } finally {
            a.close();
            Sse.keepAliveMillis = 3_000;
        }
    }
}
