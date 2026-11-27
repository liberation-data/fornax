package org.liberationdata.fornax.server;

import com.sun.net.httpserver.HttpExchange;
import org.liberationdata.fornax.http.EventStream;

import java.util.concurrent.CancellationException;

/**
 * A streamed reply as OpenAI sends one: server-sent events of {@code data: <json>}, ending with {@code data: [DONE]},
 * written through an {@link EventStream} (shared with fornax-studio-server, as are its keep-alives).
 *
 * <p>The status and headers go out with the first event, so a request that fails before anything is streamed (a model
 * that won't load) still gets an error status. A write that fails means the client has gone: {@code onGone} runs once
 * (it stops the generation), and {@link #data} throws a CancellationException, which from a model's callback stops the
 * generation too.
 */
final class Sse implements AutoCloseable {
    /** How often a silent stream is kept alive, and a client that has gone noticed (a test shortens it). */
    static volatile long keepAliveMillis = 3_000;

    private final EventStream stream;

    Sse(HttpExchange exchange, Runnable onGone) { stream = new EventStream(exchange, keepAliveMillis, onGone); }

    /**
     * Gives up the stream before anything has been sent, so that the reply can be an ordinary one (an error with its
     * status). False once something has been sent: then a failure can only be reported in the stream.
     */
    boolean abandon() { return stream.abandon(); }

    /** One event; throws a CancellationException if the client has gone. */
    void data(Object json) {
        if (!stream.write("data: " + Json.write(json) + "\n\n")) throw new CancellationException("the client has gone");
    }

    /** The end of the stream. */
    void done() { stream.write("data: [DONE]\n\n"); }

    @Override public void close() { stream.close(); }
}
