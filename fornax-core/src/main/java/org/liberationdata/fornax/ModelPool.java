package org.liberationdata.fornax;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.Semaphore;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

/**
 * Keeps models loaded between uses, the way ComfyUI does, within a memory budget.
 *
 * <pre>{@code
 * var pool = new ModelPool(96L << 30, Duration.ofMinutes(10));   // 96 GB budget, unload after 10 idle minutes
 * try (var lease = pool.acquire("qwen3-4b", ModelPool.sizeOf(gguf), () -> new LlamaModel(gguf.toString()))) {
 *     lease.model().chat(...);
 * }
 * }</pre>
 *
 * <ul>
 *   <li>A model loads on first {@link #acquire} and stays loaded after its lease closes.</li>
 *   <li>Loading a model that would exceed the budget first unloads idle models, least recently used first.
 *       Models in use are never unloaded; if they alone exceed the budget, the load goes ahead anyway. A model
 *       counts against the budget from the moment its load starts, so two loads at once make room for both.</li>
 *   <li>Models idle longer than {@code idleTimeout} are unloaded in the background ({@code null} = never).</li>
 *   <li>A model has one user at a time (engine instances aren't thread-safe): {@code acquire} waits for it. One
 *       that serves several requests at once (a language model with slots) is acquired with its concurrency, and
 *       has that many users at once; it is unloaded only once all of them are done with it.</li>
 * </ul>
 *
 * Sizes are the caller's estimate of memory used, usually the size of the weight files ({@link #sizeOf}).
 */
public final class ModelPool implements AutoCloseable {

    /** A model borrowed from the pool. Close it to give the model back (it stays loaded). */
    public interface Lease<M> extends AutoCloseable {
        M model();

        @Override void close();
    }

    /** A snapshot of one pooled model. */
    public record Entry(String key, long sizeBytes, boolean loaded, boolean inUse, Instant lastUsed) {}

    private final class Slot {
        final String key;
        /** How many leases at once. Unloading takes them all, so a model in use by anyone is never unloaded. */
        final int concurrency;
        // Not a ReentrantLock: a thread holding this model must not be able to "re-lock" it from makeRoom and evict it.
        final Semaphore use;
        /**
         * Held while the model loads, so of several leases taken at once, only the first loads it. A lock, not a
         * monitor: the others wait on it for the whole load, and on JDK 22 and 23 a virtual thread waiting for a
         * monitor pins its carrier.
         */
        final ReentrantLock loading = new ReentrantLock();
        volatile AutoCloseable model;
        volatile long size;
        /** The size of the model while its loader runs, held against the budget so a concurrent load counts it. */
        volatile long reserved;
        volatile Instant lastUsed = Instant.now();

        Slot(String key, int concurrency) {
            this.key = key;
            this.concurrency = concurrency;
            this.use = new Semaphore(concurrency, true);
        }

        boolean tryAcquireAll() { return use.tryAcquire(concurrency); }
        void acquireAll() { use.acquireUninterruptibly(concurrency); }
        void releaseAll() { use.release(concurrency); }
    }

    private final long budget;
    private final Duration idleTimeout;
    private final Map<String, Slot> slots = new LinkedHashMap<>();
    private final ScheduledExecutorService reaper;
    private volatile boolean closed;
    /**
     * Making room and reserving it are one step: two loads at once can't both take the same free space. Held only
     * around the accounting and the unloading of idle models (which other loads wait for), never while a loader runs,
     * and nothing blocks under it: makeRoom only tries other models' locks.
     */
    private final Object budgetLock = new Object();

    public ModelPool(long budgetBytes, Duration idleTimeout) {
        this.budget = budgetBytes;
        this.idleTimeout = idleTimeout;
        if (idleTimeout != null) {
            reaper = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "fornax-model-pool-reaper");
                t.setDaemon(true);
                return t;
            });
            long period = Math.max(1, Math.min(60, idleTimeout.toSeconds() / 2));
            reaper.scheduleWithFixedDelay(this::unloadIdle, period, period, TimeUnit.SECONDS);
        } else {
            reaper = null;
        }
    }

    /** Borrows the model under {@code key}, loading it with {@code loader} if it isn't loaded. */
    public <M extends AutoCloseable> Lease<M> acquire(String key, long sizeBytes, Supplier<M> loader) {
        return acquire(key, sizeBytes, 1, loader);
    }

    /**
     * Borrows the model under {@code key}, which takes up to {@code concurrency} users at once (a language model with
     * that many slots); a key keeps the concurrency it was first acquired with.
     * @throws IllegalArgumentException for a concurrency below 1, or other than the key's
     */
    @SuppressWarnings("unchecked")
    public <M extends AutoCloseable> Lease<M> acquire(String key, long sizeBytes, int concurrency, Supplier<M> loader) {
        if (closed) throw new IllegalStateException("pool is closed");
        if (concurrency < 1) throw new IllegalArgumentException("concurrency must be at least 1");
        Slot slot;
        synchronized (slots) { slot = slots.computeIfAbsent(key, k -> new Slot(k, concurrency)); }
        if (slot.concurrency != concurrency)
            throw new IllegalArgumentException(key + " was pooled with concurrency " + slot.concurrency + ", not " + concurrency);
        slot.use.acquireUninterruptibly();
        try {
            slot.loading.lock();
            try {
                if (slot.model == null) {
                    synchronized (budgetLock) {
                        makeRoom(sizeBytes, slot);
                        slot.reserved = sizeBytes;
                    }
                    slot.size = sizeBytes;
                    AutoCloseable loaded = null;
                    try {
                        loaded = loader.get();
                    } finally {
                        // One step, so makeRoom counts it by its reservation or by its size: never both, never neither.
                        synchronized (budgetLock) {
                            slot.model = loaded;
                            slot.reserved = 0;   // loaded (now counted by its size) or failed (given back)
                        }
                    }
                }
            } finally { slot.loading.unlock(); }
            slot.lastUsed = Instant.now();
        } catch (RuntimeException | Error e) {
            slot.use.release();
            throw e;
        }
        M model = (M) slot.model;
        return new Lease<>() {
            private boolean released;

            @Override public M model() {
                if (released) throw new IllegalStateException("lease already closed");
                return model;
            }

            @Override public void close() {
                if (released) return;
                released = true;
                slot.lastUsed = Instant.now();
                slot.use.release();
            }
        };
    }

    /** Unloads the model under {@code key} once it's free. Returns false if it wasn't loaded. */
    public boolean unload(String key) {
        Slot slot;
        synchronized (slots) { slot = slots.get(key); }
        if (slot == null) return false;
        slot.acquireAll();
        try { return closeModel(slot); } finally { slot.releaseAll(); }
    }

    /** Unloads every model not in use. */
    public void unloadAll() {
        for (Slot s : snapshot()) {
            if (s.tryAcquireAll()) {
                try { closeModel(s); } finally { s.releaseAll(); }
            }
        }
    }

    public List<Entry> entries() {
        List<Entry> out = new ArrayList<>();
        for (Slot s : snapshot()) out.add(new Entry(s.key, s.size, s.model != null, s.use.availablePermits() < s.concurrency, s.lastUsed));
        return out;
    }

    /** Estimated bytes held by loaded models. */
    public long loadedBytes() {
        long n = 0;
        for (Slot s : snapshot()) if (s.model != null) n += s.size;
        return n;
    }

    /** Bytes reserved by models loading now, which {@link #loadedBytes} counts once they have loaded. */
    private long reservedBytes() {
        long n = 0;
        for (Slot s : snapshot()) n += s.reserved;
        return n;
    }

    /** Total size of the given files: a reasonable estimate of what loading them takes. */
    public static long sizeOf(Path... files) {
        long n = 0;
        try {
            for (Path p : files) if (p != null) n += Files.size(p);
        } catch (IOException e) { throw new UncheckedIOException(e); }
        return n;
    }

    @Override public void close() {
        closed = true;
        if (reaper != null) reaper.shutdownNow();
        for (Slot s : snapshot()) {
            s.acquireAll();
            try { closeModel(s); } finally { s.releaseAll(); }
        }
    }

    // ---- internals ----

    private void makeRoom(long needed, Slot requester) {
        List<Slot> idle = new ArrayList<>();
        for (Slot s : snapshot()) if (s != requester && s.model != null) idle.add(s);
        idle.sort(Comparator.comparing(s -> s.lastUsed));
        long loaded = loadedBytes() + reservedBytes();
        for (Slot s : idle) {
            if (loaded + needed <= budget) break;
            if (!s.tryAcquireAll()) continue;  // in use: never evicted
            try {
                long size = s.size;
                if (closeModel(s)) loaded -= size;
            } finally { s.releaseAll(); }
        }
    }

    private void unloadIdle() {
        Instant cutoff = Instant.now().minus(idleTimeout);
        for (Slot s : snapshot()) {
            if (s.model == null || s.lastUsed.isAfter(cutoff) || !s.tryAcquireAll()) continue;
            try {
                if (s.lastUsed.isBefore(cutoff)) closeModel(s);
            } finally { s.releaseAll(); }
        }
    }

    /** Caller holds all of s.use. */
    private static boolean closeModel(Slot s) {
        AutoCloseable m = s.model;
        if (m == null) return false;
        s.model = null;
        try { m.close(); } catch (Exception e) { throw new IllegalStateException("closing " + s.key + " failed", e); }
        return true;
    }

    private List<Slot> snapshot() {
        synchronized (slots) { return new ArrayList<>(slots.values()); }
    }
}
