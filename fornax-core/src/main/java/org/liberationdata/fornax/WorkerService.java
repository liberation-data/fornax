package org.liberationdata.fornax;

import java.util.function.Consumer;

/**
 * What a {@link Worker} hosts: one engine instance (a loaded model) in the worker process, called by method name with
 * {@link Wire}-encoded arguments. The worker creates it from its class name through a public constructor that takes the
 * config bytes passed to {@link Worker#open}, and runs its calls one at a time on a {@link ModelThread} of its own
 * (or up to {@link #lanes()} at once).
 * An engine module provides one for its model class (fornax-llama's {@code LlamaService}); you write one only to isolate
 * something Fornax doesn't.
 */
public interface WorkerService extends AutoCloseable {
    /**
     * Runs one call, in the worker. An unchecked exception goes back to the caller, as the same class where it can be
     * rebuilt there (see {@link Worker.Instance#invoke}).
     * @param call   this call: {@link Call#stopRequested()} turns true when the caller cancels it
     * @param events sends an item (a token, a chunk) to the caller while the call runs; items arrive in order, before
     *               the result
     * @return the result, encoded
     */
    byte[] call(String method, byte[] args, Call call, Consumer<byte[]> events);

    /**
     * How many calls it runs at once: 1 unless it batches them (a language model with slots), and then those calls stop
     * through {@link Call#stopRequested()} rather than {@link #cancel()}. Open its instance with as many lanes
     * ({@link Worker#open(String, byte[], int)}) for them to arrive together.
     */
    default int lanes() { return 1; }

    /** Stops the call in progress, from another thread, for an engine with a cancel of its own. */
    default void cancel() {}

    /** Frees the engine instance. */
    @Override default void close() {}
}
