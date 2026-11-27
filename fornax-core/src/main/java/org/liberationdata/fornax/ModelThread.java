package org.liberationdata.fornax;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executor;
import java.util.concurrent.Flow;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.SubmissionPublisher;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * Owns one loaded engine instance: runs its calls one at a time, on the caller's thread ({@link #run}) or on a
 * platform thread of its own ({@link #submit}), cancels them, and frees the instance exactly once.
 * Every Fornax model class keeps one; you use it through the model's {@code ...Async} methods.
 *
 * <h2>Why a thread of its own</h2>
 * A native call pins a virtual thread to its carrier for as long as it runs, which for a generation is seconds to
 * minutes. A few virtual threads generating at once can occupy every carrier (there is about one per core) and
 * stall unrelated virtual-thread work. So the async methods run the native call on this platform thread
 * ({@code fornax-<name>}, started on first use) and hand back a {@link CompletableFuture}; a virtual thread that waits
 * on it unmounts, and its carrier goes on running other work. The thread has an 8 MB stack, a command-line tool's
 * main-thread size, because some GPU backends recurse deeply in native code.
 *
 * <h2>One call at a time</h2>
 * Engine instances aren't thread-safe. Calls from any thread, blocking or async, queue for one lock (fair, so in
 * arrival order) and run one after another; the async ones queue in submission order on the thread.
 *
 * <p>An engine that runs several calls together (a language model with slots, which decodes them in one batch) takes
 * {@code lanes} calls at once instead: as many threads, and up to that many calls running, the rest queued in order.
 * Its calls hand their work to the engine's own loop and wait for it, and its callbacks run on that loop's thread,
 * which {@link #markCallbackThread} tells this class about.
 *
 * <h2>Streaming</h2>
 * {@link #submit(BiFunction, Flow.Subscriber)} streams a call's items (tokens, audio chunks) to a
 * {@link Flow.Subscriber} through a bounded buffer, delivered on the stream executor ({@link #streamOn}; virtual
 * threads by default), so a slow subscriber never slows the GPU. A subscriber that falls a whole buffer behind stops
 * the generation, and gets a {@link StreamOverflowException}; one that cancels its subscription stops it too.
 *
 * <h2>Cancel and close</h2>
 * {@link #cancel()} stops the call in progress; what it returns then is up to the engine (text so far, or a
 * {@link CancellationException}). {@link #close()} is idempotent and safe while a call is in flight: it cancels that
 * call, fails queued async calls with a {@link CancellationException}, waits for the native call to return, and only
 * then frees the instance. An engine that can't stop mid-call makes close wait for it to finish.
 */
public final class ModelThread implements AutoCloseable {
    /** Items a streaming subscriber may fall behind before the generation stops. */
    public static final int DEFAULT_BUFFER = 1024;

    /** Stream delivery by default: a virtual thread per delivery run, so a subscriber may block freely. */
    private static final Executor VIRTUAL = r -> Thread.ofVirtual().name("fornax-stream").start(r);

    private final String name;
    private final Runnable destroy, nativeCancel, nativeReset;
    private final int lanes;
    private final Semaphore engine;                               // fair: calls run in arrival order
    /** Threads inside a call (or running its callbacks), where waiting for a lane would wait for themselves. */
    private final ThreadLocal<Boolean> inside = ThreadLocal.withInitial(() -> false);
    private final Object state = new Object();
    private final ArrayDeque<Job<?>> queue = new ArrayDeque<>();  // guarded by state
    private final CountDownLatch destroyed = new CountDownLatch(1);
    private final Set<Call> current = new HashSet<>();            // guarded by state: the calls running
    private boolean closed;                                       // guarded by state
    private int busy;                                             // guarded by state: workers running a job
    private final List<Thread> workers = new ArrayList<>();       // guarded by state
    private Thread closer;                                        // guarded by state: the thread doing the first close
    private volatile Executor streamExecutor = VIRTUAL;
    private volatile int streamBuffer = DEFAULT_BUFFER;

    /** For an engine whose calls stop through a callback that polls {@link Call#stopRequested()}. */
    public ModelThread(String name, Runnable destroy) { this(name, destroy, null, null); }

    /**
     * @param nativeCancel asks the engine to stop its run, from any thread (null if the upcalls do it). It is only
     *                     ever called while the call it is for is the one running.
     * @param nativeReset  clears that request before each call (null if not needed)
     */
    public ModelThread(String name, Runnable destroy, Runnable nativeCancel, Runnable nativeReset) {
        this(name, 1, destroy, nativeCancel, nativeReset);
    }

    /**
     * For an engine that runs up to {@code lanes} calls at once (see the class doc), whose calls stop through
     * {@link Call#stopRequested()}.
     */
    public ModelThread(String name, int lanes, Runnable destroy) { this(name, lanes, destroy, null, null); }

    private ModelThread(String name, int lanes, Runnable destroy, Runnable nativeCancel, Runnable nativeReset) {
        if (lanes < 1) throw new IllegalArgumentException("lanes must be at least 1");
        this.name = name;
        this.lanes = lanes;
        this.engine = new Semaphore(lanes, true);
        this.destroy = Objects.requireNonNull(destroy);
        this.nativeCancel = nativeCancel;
        this.nativeReset = nativeReset;
    }

    /** How many calls can run at once. */
    public int lanes() { return lanes; }

    /**
     * Marks the calling thread as one that runs this model's callbacks (an engine's own loop): a blocking call or
     * {@link #close()} from it would wait for itself, so they throw instead, as they do from inside a call.
     */
    public void markCallbackThread() { inside.set(true); }

    /**
     * Where streamed items are delivered (and async futures completed), and how many items a subscriber may fall
     * behind before the generation stops (SubmissionPublisher rounds it up to a power of two). Applies to calls
     * submitted afterwards.
     */
    public void streamOn(Executor executor, int bufferSize) {
        if (bufferSize < 1) throw new IllegalArgumentException("bufferSize must be at least 1");
        this.streamExecutor = Objects.requireNonNull(executor);
        this.streamBuffer = bufferSize;
    }

    /**
     * Runs a call on the calling thread, after any call in progress. Blocks (and pins a virtual thread) meanwhile.
     * @throws IllegalStateException if this model is closed, or if called from inside one of its own calls (a
     *         callback), which would wait for itself. An async call made there is queued instead: it runs after.
     */
    public <T> T run(Function<Call, T> body) {
        if (inside.get())
            throw new IllegalStateException("a call to " + name + " was made from inside one of its own callbacks");
        engine.acquireUninterruptibly();
        inside.set(true);
        try {
            return runLocked(new Call(this), body, false);
        } finally {
            inside.set(false);
            engine.release();
        }
    }

    /** Runs a call on this model's thread. Cancelling the future (or failing it, as orTimeout does) cancels the call. */
    public <T> CompletableFuture<T> submit(Function<Call, T> body) {
        Objects.requireNonNull(body);
        return enqueue(new Call(this), body);
    }

    /**
     * Runs a streaming call on this model's thread: body gets a sink for its items, which reach subscriber through a
     * bounded buffer. The subscriber completes after the last item when the call returns, and gets onError if the call
     * fails, overflows ({@link StreamOverflowException}) or the future is cancelled. Cancelling the subscription stops
     * the call; the future then completes with whatever the engine returns for a cancelled call.
     */
    public <T, R> CompletableFuture<R> submit(BiFunction<Call, Consumer<T>, R> body, Flow.Subscriber<? super T> subscriber) {
        Objects.requireNonNull(body);
        Objects.requireNonNull(subscriber);
        Call call = new Call(this);
        // A subscriber method that throws cancels its subscription; the handler makes that the call's failure.
        SubmissionPublisher<T> pub = new SubmissionPublisher<>(streamExecutor, streamBuffer, (s, t) -> call.fail(t));
        pub.subscribe(subscriber);
        int capacity = pub.getMaxBufferCapacity();
        Consumer<T> sink = item -> {
            if (call.stopRequested()) return;
            if (!pub.hasSubscribers()) { call.cancel(); return; }  // the subscriber cancelled
            try {
                pub.offer(item, (s, dropped) -> {
                    call.fail(new StreamOverflowException(capacity));
                    return false;
                });
            } catch (IllegalStateException closedMeanwhile) {      // the future was cancelled, which closes pub
                call.cancel();
            }
        };
        CompletableFuture<R> f = enqueue(call, c -> body.apply(c, sink));
        f.whenComplete((v, e) -> {
            if (e == null) pub.close();
            else pub.closeExceptionally(e instanceof CompletionException && e.getCause() != null ? e.getCause() : e);
        });
        return f;
    }

    /** A cold publisher: each subscription runs the call once, as {@link #submit(BiFunction, Flow.Subscriber)} does. */
    public <T> Flow.Publisher<T> publisher(BiFunction<Call, Consumer<T>, ?> body) {
        Objects.requireNonNull(body);
        return subscriber -> submit(body, subscriber);
    }

    /** Stops the calls in progress, if any (from any thread). Queued calls still run. */
    public void cancel() {
        synchronized (state) {
            for (Call c : current) stop(c);
        }
    }

    /** Stops one call, if it hasn't finished: called by {@link Call#cancel()}. */
    void cancel(Call call) {
        synchronized (state) { stop(call); }
    }

    /**
     * Caller holds state, so a native cancel can't land on the call after this one. Repeating it is allowed: an
     * engine that ignores a cancel before its run has really begun (stable-diffusion.cpp) has it re-asserted from
     * its progress callback.
     */
    private void stop(Call call) {
        call.markStopped();
        if (nativeCancel != null && current.contains(call)) nativeCancel.run();
    }

    /** Whether {@link #close()} has been called. */
    public boolean isClosed() {
        synchronized (state) { return closed; }
    }

    /**
     * Cancels the call in progress and every queued one, waits for the native call to return, then frees the engine
     * instance. Idempotent: a second close (from any thread) waits for the first to finish freeing it and returns;
     * one nested in the first, on its thread (a queued call's dependent stage, which runs as close fails it), returns
     * at once and leaves the freeing to the outer close.
     * @throws IllegalStateException if called from inside one of this model's own callbacks, which would free the
     *         instance under the native call that is running the callback
     */
    @Override public void close() {
        if (inside.get())
            throw new IllegalStateException("close() was called from inside a call to " + name + "; close it after the call returns");
        boolean first;
        List<Job<?>> abandoned;
        List<Thread> ws;
        synchronized (state) {
            // Nested in the first close, on its thread (a queued call's dependent, failed below, closing the model):
            // waiting here would wait for itself, so leave the freeing to the outer close.
            if (closed && closer == Thread.currentThread()) return;
            first = !closed;
            if (first) closer = Thread.currentThread();
            closed = true;
            for (Call c : current) stop(c);
            abandoned = new ArrayList<>(queue);
            queue.clear();
            ws = new ArrayList<>(workers);
            state.notifyAll();
        }
        for (Job<?> j : abandoned) j.future.completeExceptionally(new CancellationException(name + " was closed"));
        if (first) {
            engine.acquireUninterruptibly(lanes);  // the calls in progress return first
            try {
                destroy.run();
            } finally {
                engine.release(lanes);
                destroyed.countDown();
            }
        } else {
            awaitUninterruptibly(destroyed);
        }
        for (Thread w : ws) if (w != Thread.currentThread()) joinUninterruptibly(w);
    }

    // ---- internals ----

    /** @param queued an async call, which close() fails with a CancellationException even once it has left the queue */
    private <T> T runLocked(Call call, Function<Call, T> body, boolean queued) {
        synchronized (state) {
            if (closed) throw queued ? new CancellationException(name + " was closed") : new IllegalStateException(name + " is closed");
            if (call.stopRequested()) throw new CancellationException("cancelled before it started");
            if (nativeReset != null) nativeReset.run();
            current.add(call);
        }
        try {
            T result = body.apply(call);
            call.rethrow();  // a callback failed and the binding didn't check
            return result;
        } finally {
            synchronized (state) { current.remove(call); }
        }
    }

    private <T> CompletableFuture<T> enqueue(Call call, Function<Call, T> body) {
        CompletableFuture<T> f = new CompletableFuture<>();
        f.whenComplete((v, e) -> { if (e != null) call.cancel(); });  // cancel(), orTimeout(), completeExceptionally()
        synchronized (state) {
            if (closed) {
                f.completeExceptionally(new IllegalStateException(name + " is closed"));
                return f;
            }
            queue.add(new Job<>(call, body, f));
            if (workers.size() < Math.min(lanes, queue.size() + busy)) {
                Thread w = Thread.ofPlatform().name(lanes == 1 ? "fornax-" + name : "fornax-" + name + "-" + workers.size())
                        .daemon(true).stackSize(8L << 20).unstarted(this::work);
                workers.add(w);
                w.start();
            }
            state.notifyAll();
        }
        return f;
    }

    private void work() {
        while (true) {
            Job<?> job;
            synchronized (state) {
                while (queue.isEmpty() && !closed) {
                    try { state.wait(); } catch (InterruptedException ignored) {}
                }
                if (queue.isEmpty()) return;
                job = queue.poll();
                busy++;
            }
            try { job.run(); } finally { synchronized (state) { busy--; } }
        }
    }

    private final class Job<T> {
        final Call call;
        final Function<Call, T> body;
        final CompletableFuture<T> future;

        Job(Call call, Function<Call, T> body, CompletableFuture<T> future) {
            this.call = call; this.body = body; this.future = future;
        }

        void run() {
            if (future.isDone()) return;  // cancelled while queued
            T result;
            engine.acquireUninterruptibly();
            inside.set(true);
            try {
                result = runLocked(call, body, true);
            } catch (Throwable t) {
                complete(() -> future.completeExceptionally(t));
                return;
            } finally {
                inside.set(false);
                engine.release();
            }
            complete(() -> future.complete(result));
        }

        /**
         * Completes the future off this thread: callers' dependent stages (thenApply and the rest) would otherwise
         * run here, holding up the next generation.
         */
        private void complete(Runnable r) {
            try { streamExecutor.execute(r); } catch (RejectedExecutionException e) { r.run(); }
        }
    }

    private static void awaitUninterruptibly(CountDownLatch latch) {
        boolean interrupted = false;
        while (true) {
            try { latch.await(); break; } catch (InterruptedException e) { interrupted = true; }
        }
        if (interrupted) Thread.currentThread().interrupt();
    }

    private static void joinUninterruptibly(Thread t) {
        boolean interrupted = false;
        while (true) {
            try { t.join(); break; } catch (InterruptedException e) { interrupted = true; }
        }
        if (interrupted) Thread.currentThread().interrupt();
    }
}
