package org.liberationdata.kiln;

/**
 * One call into a model's engine: whether it has been asked to stop, and the first exception a callback threw during
 * it. {@link ModelThread} makes one for each call; the engine bindings read it from their upcalls.
 *
 * <p>Why it exists: an exception that escapes an FFM upcall can't unwind through the engine's C++ frames, so the JVM
 * prints it and exits. Every Kiln upcall therefore catches {@code Throwable}, hands it to {@link #fail}, and tells the
 * engine to stop; when the downcall returns, {@link #rethrow} throws it on the Java side, where it belongs.
 */
public final class Call {
    private final ModelThread owner;
    private volatile boolean stop;
    private Throwable failure;  // guarded by this

    Call(ModelThread owner) { this.owner = owner; }

    /** True once the call has been cancelled or a callback failed: an upcall returns "stop" to the engine. */
    public boolean stopRequested() { return stop; }

    /** Asks the engine to stop this call (and only this one), from any thread. */
    public void cancel() { owner.cancel(this); }

    /**
     * Records a callback's exception (the first one; later ones are added to it as suppressed) and asks the engine to
     * stop. Call it from an upcall's catch block, then return to the engine normally.
     */
    public void fail(Throwable t) {
        synchronized (this) {
            if (failure == null) failure = t;
            else if (failure != t) failure.addSuppressed(t);
        }
        owner.cancel(this);
    }

    /** The first exception a callback threw, or null. */
    public synchronized Throwable failure() { return failure; }

    /** Throws the callback's exception, if one threw: as itself when unchecked, wrapped otherwise. */
    public void rethrow() {
        Throwable f = failure();
        if (f == null) return;
        if (f instanceof RuntimeException e) throw e;
        if (f instanceof Error e) throw e;
        throw new RuntimeException(f);
    }

    void markStopped() { stop = true; }
}
