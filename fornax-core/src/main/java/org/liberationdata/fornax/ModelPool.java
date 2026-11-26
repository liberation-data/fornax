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
 *       Models in use are never unloaded; if they alone exceed the budget, the load goes ahead anyway.</li>
 *   <li>Models idle longer than {@code idleTimeout} are unloaded in the background ({@code null} = never).</li>
 *   <li>A model has one user at a time (engine instances aren't thread-safe): {@code acquire} waits for it.</li>
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
        // Not a ReentrantLock: a thread holding this model must not be able to "re-lock" it from makeRoom and evict it.
        final Semaphore use = new Semaphore(1, true);
        volatile AutoCloseable model;
        volatile long size;
        volatile Instant lastUsed = Instant.now();

        Slot(String key) { this.key = key; }
    }

    private final long budget;
    private final Duration idleTimeout;
    private final Map<String, Slot> slots = new LinkedHashMap<>();
    private final ScheduledExecutorService reaper;
    private volatile boolean closed;

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
    @SuppressWarnings("unchecked")
    public <M extends AutoCloseable> Lease<M> acquire(String key, long sizeBytes, Supplier<M> loader) {
        if (closed) throw new IllegalStateException("pool is closed");
        Slot slot;
        synchronized (slots) { slot = slots.computeIfAbsent(key, Slot::new); }
        slot.use.acquireUninterruptibly();
        try {
            if (slot.model == null) {
                makeRoom(sizeBytes, slot);
                slot.model = loader.get();
                slot.size = sizeBytes;
            }
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
        slot.use.acquireUninterruptibly();
        try { return closeModel(slot); } finally { slot.use.release(); }
    }

    /** Unloads every model not in use. */
    public void unloadAll() {
        for (Slot s : snapshot()) {
            if (s.use.tryAcquire()) {
                try { closeModel(s); } finally { s.use.release(); }
            }
        }
    }

    public List<Entry> entries() {
        List<Entry> out = new ArrayList<>();
        for (Slot s : snapshot()) out.add(new Entry(s.key, s.size, s.model != null, s.use.availablePermits() == 0, s.lastUsed));
        return out;
    }

    /** Estimated bytes held by loaded models. */
    public long loadedBytes() {
        long n = 0;
        for (Slot s : snapshot()) if (s.model != null) n += s.size;
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
            s.use.acquireUninterruptibly();
            try { closeModel(s); } finally { s.use.release(); }
        }
    }

    // ---- internals ----

    private void makeRoom(long needed, Slot requester) {
        List<Slot> idle = new ArrayList<>();
        for (Slot s : snapshot()) if (s != requester && s.model != null) idle.add(s);
        idle.sort(Comparator.comparing(s -> s.lastUsed));
        long loaded = loadedBytes();
        for (Slot s : idle) {
            if (loaded + needed <= budget) break;
            if (!s.use.tryAcquire()) continue;  // in use: never evicted
            try {
                long size = s.size;
                if (closeModel(s)) loaded -= size;
            } finally { s.use.release(); }
        }
    }

    private void unloadIdle() {
        Instant cutoff = Instant.now().minus(idleTimeout);
        for (Slot s : snapshot()) {
            if (s.model == null || s.lastUsed.isAfter(cutoff) || !s.use.tryAcquire()) continue;
            try {
                if (s.lastUsed.isBefore(cutoff)) closeModel(s);
            } finally { s.use.release(); }
        }
    }

    /** Caller holds s.use. */
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
