package org.liberationdata.fornax.llama;

import org.liberationdata.fornax.Call;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.function.Consumer;
import java.util.function.IntSupplier;

/**
 * The loop that drives a model loaded with slots: one platform thread owns the engine, adds the requests that
 * arrive, and steps every running request together until each ends. Callers block in {@link #run} until theirs has.
 *
 * <p>Why one thread: a llama context isn't thread-safe, and a step decodes every slot at once. So requests, and their
 * cancels, are handed to this thread, which applies them between steps; nothing else touches the engine. Pieces of
 * text reach each request's consumer on this thread, so a consumer that blocks holds up every slot (the async and
 * streaming methods hand pieces to a bounded buffer, which doesn't block).
 *
 * <p>{@link org.liberationdata.fornax.ModelThread}'s lanes bound the requests in flight to the engine's slots, so an add always finds one free.
 */
final class Batcher implements AutoCloseable {

    /** The engine, as the loop sees it. Called only from the loop's thread. */
    interface Engine {
        /** Ends request seq at the next step, with what it has. */
        void cancel(int seq);

        /** One step for every running request; their pieces and ends go to events. */
        void step(Events events);
    }

    /** What a step reports. */
    interface Events {
        /** New text for request seq. Returns true to stop it. */
        boolean piece(int seq, String text);

        /** Request seq ended: rc 0 with its response, or an error code and message. Its slot is free again. */
        void done(int seq, int rc, String response);
    }

    private static final class Request {
        final Call call;
        final IntSupplier add;
        final Consumer<String> onPiece;
        final CompletableFuture<String> response = new CompletableFuture<>();
        int seq = -1;

        Request(Call call, IntSupplier add, Consumer<String> onPiece) {
            this.call = call; this.add = add; this.onPiece = onPiece;
        }
    }

    private final String name;
    private final Engine engine;
    private final Runnable onThread;
    private final Object lock = new Object();
    private final ArrayDeque<Request> pending = new ArrayDeque<>();  // guarded by lock
    private final Map<Integer, Request> running = new HashMap<>();   // the loop's own
    private boolean closed;                                          // guarded by lock
    private Throwable failed;                                        // guarded by lock: why the loop died, if it did
    private Thread thread;                                           // guarded by lock

    /** @param onThread runs first on the loop's thread (to mark it as the one that runs callbacks) */
    Batcher(String name, Engine engine, Runnable onThread) {
        this.name = name;
        this.engine = Objects.requireNonNull(engine);
        this.onThread = Objects.requireNonNull(onThread);
    }

    /**
     * Runs one request and waits for its response. {@code add} runs on the loop's thread and puts the request in a
     * free slot, returning the slot, or throws for a request the engine refuses. Stopping the call (cancel, or a
     * consumer that threw) ends it at the next step with what it has.
     * @throws IllegalStateException with the engine's message when the request failed
     */
    String run(Call call, IntSupplier add, Consumer<String> onPiece) {
        Request r = new Request(call, add, onPiece);
        synchronized (lock) {
            if (failed != null) throw new IllegalStateException(name + "'s batch loop failed", failed);
            if (closed) throw new CancellationException(name + " was closed");
            pending.add(r);
            if (thread == null) {
                thread = Thread.ofPlatform().name("fornax-" + name + "-batch").daemon(true).stackSize(8L << 20).unstarted(this::loop);
                thread.start();
            }
            lock.notifyAll();
        }
        try {
            return r.response.join();
        } catch (CompletionException e) {
            if (e.getCause() instanceof RuntimeException re) throw re;
            if (e.getCause() instanceof Error err) throw err;
            throw e;
        }
    }

    /** Stops the loop once the requests in flight have ended (the caller has seen to that), and waits for it. */
    @Override public void close() {
        Thread t;
        synchronized (lock) {
            closed = true;
            t = thread;
            lock.notifyAll();
        }
        if (t != null && t != Thread.currentThread()) {
            boolean interrupted = false;
            while (true) {
                try { t.join(); break; } catch (InterruptedException e) { interrupted = true; }
            }
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    private final Events events = new Events() {
        @Override public boolean piece(int seq, String text) {
            Request r = running.get(seq);
            if (r == null) return true;
            if (r.onPiece != null && !text.isEmpty()) {
                try {
                    r.onPiece.accept(text);
                } catch (Throwable t) {
                    r.call.fail(t);
                }
            }
            return r.call.stopRequested();
        }

        @Override public void done(int seq, int rc, String response) {
            Request r = running.remove(seq);
            if (r == null) return;
            if (rc == 0) r.response.complete(response);
            else r.response.completeExceptionally(new IllegalStateException("llama chat failed (rc=" + rc + "): " + response));
        }
    };

    private void loop() {
        onThread.run();
        try {
            while (true) {
                List<Request> adds;
                synchronized (lock) {
                    while (pending.isEmpty() && running.isEmpty() && !closed) {
                        try { lock.wait(); } catch (InterruptedException ignored) {}
                    }
                    if (closed && pending.isEmpty() && running.isEmpty()) return;
                    adds = new ArrayList<>(pending);
                    pending.clear();
                }
                for (Request r : adds) {
                    try {
                        r.seq = r.add.getAsInt();
                        running.put(r.seq, r);
                    } catch (Throwable t) {
                        r.response.completeExceptionally(t);
                    }
                }
                for (Request r : running.values()) if (r.call.stopRequested()) engine.cancel(r.seq);
                if (!running.isEmpty()) engine.step(events);
            }
        } catch (Throwable t) {
            // The engine threw (it shouldn't: its errors come back through done). Nothing can run on it now.
            List<Request> left;
            synchronized (lock) {
                closed = true;
                failed = t;
                left = new ArrayList<>(pending);
                pending.clear();
            }
            left.addAll(running.values());
            running.clear();
            for (Request r : left) r.response.completeExceptionally(new IllegalStateException(name + "'s batch loop failed", t));
        }
    }
}
