package org.liberationdata.fornax.studio;

import com.sun.net.httpserver.HttpExchange;
import org.liberationdata.fornax.http.EventStream;

import java.io.IOException;
import java.util.concurrent.CancellationException;

/**
 * A reply sent as server-sent events: {@code event: name} and {@code data: json}, flushed as each is sent, through an
 * {@link EventStream} (shared with fornax-server, as are its keep-alives).
 *
 * <p>{@link #send} never throws: once the page has gone, sending just stops, and {@link #open} says so. A model's
 * callback sends with {@link #stream} instead, which throws then: Fornax stops the generation and rethrows it from the
 * call, so a page that leaves frees the model for the next request rather than holding it to the end.
 *
 * <p>A comment is sent every {@link #keepAliveMillis} while nothing else is: a model loading, or a video's decode, can
 * say nothing for minutes, and the studio's proxy (Node's fetch) gives up on a reply silent for five.
 */
final class Sse implements AutoCloseable {
    /** How often a silent stream is kept alive (a test shortens it). */
    static volatile long keepAliveMillis = 15_000;

    private final HttpExchange exchange;
    private final EventStream out;

    /** Sends the status and headers at once. */
    Sse(HttpExchange exchange) throws IOException {
        this.exchange = exchange;
        this.out = new EventStream(exchange, keepAliveMillis, () -> {});
        out.start();
    }

    boolean open() { return out.open(); }

    boolean send(String event, Object data) {
        return out.write("event: " + event + "\ndata: " + Json.write(data) + "\n\n");
    }

    /** {@link #send}, from a model's callback: throws a CancellationException once the page has gone. */
    void stream(String event, Object data) {
        if (!send(event, data)) throw new CancellationException("the page has gone");
    }

    @Override public void close() {
        out.close();
        exchange.close();
    }
}
