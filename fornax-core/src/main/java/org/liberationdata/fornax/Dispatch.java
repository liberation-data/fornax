package org.liberationdata.fornax;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;

/**
 * A pool of {@link Worker} processes that serves requests for many models: it routes each request to the worker that
 * already has its model loaded, loads a model on demand within one memory budget (unloading the least recently used
 * ones to make room), and queues requests by {@link QoS} class, with aging so that {@link QoS#BATCH} work can't starve.
 * For servers; {@link ModelPool} stays the simple single-process default.
 *
 * <pre>{@code
 * try (var dispatch = new Dispatch(Dispatch.Options.of(2, 48L << 30))) {     // two worker processes, a 48 GB budget
 *     var qwen = dispatch.register("qwen3-4b", ModelPool.sizeOf(gguf),
 *             worker -> LlamaModel.isolated(worker, gguf.toString(), null, -1, 8192));
 *     Result r = qwen.submit(QoS.INTERACTIVE, llm -> llm.chatAsync(messages, LlamaModel.Options.DEFAULT)).join();
 * }
 * }</pre>
 *
 * <ul>
 *   <li><b>Workers run one request at a time.</b> The GPU is the scarce resource, so a worker is the unit of
 *   concurrency: {@code workers} requests run at once, each with its model to itself. A model is loaded in one worker
 *   at a time; a worker may hold several. The exception is a model registered with a concurrency above 1 (a language
 *   model with slots, which decodes its requests together): once loaded, it takes up to that many requests at once on
 *   its worker, unless a request ranked above them is waiting for a worker (or for room).</li>
 *   <li><b>Affinity.</b> A request for a loaded model waits for that model's worker. A request for a model that isn't
 *   loaded takes an idle worker (one that hasn't just crashed first, then the one with the fewest bytes loaded) and
 *   loads it there.</li>
 *   <li><b>Budget.</b> ModelPool's rules, across the whole pool: loading first unloads idle models, least recently used
 *   first, from any worker. A model in use is never unloaded. A load that won't fit until a model in use finishes waits
 *   for it, and loads for requests ranked below it wait too, so the budget can't be raced away. A model too big for the
 *   budget still loads when nothing else is loaded. Each worker JVM counts {@link Options#workerOverheadBytes}.</li>
 *   <li><b>QoS.</b> Requests are granted in order of class, then arrival. A request's class rises one step for each
 *   {@link Options#aging} it has waited, so a {@code BATCH} request that has waited that long is ranked with
 *   {@code INTERACTIVE} ones by arrival, ahead of newer ones.</li>
 *   <li><b>Cancel and timeouts.</b> Cancelling a queued request (or {@code orTimeout} on it) takes it out of the queue,
 *   and it never runs. Cancelling a running one cancels the future its body returned, which for a Fornax model stops the
 *   generation.</li>
 *   <li><b>Crashes.</b> A worker that dies during a request takes all its models with it: they are dropped here, the
 *   request fails with the {@link WorkerCrashedException}, and the next request for any of them loads it again,
 *   preferring another worker while the crashed one backs off and restarts.</li>
 *   <li><b>Back-pressure.</b> At most {@link Options#maxQueued} requests wait in each class; one more fails at once with
 *   a {@link RejectedExecutionException}.</li>
 * </ul>
 */
public final class Dispatch implements AutoCloseable {

    /** How urgent a request is. Higher classes go first; waiting raises a request's class (see {@link Options#aging}). */
    public enum QoS {
        /** A person is waiting: chat, vision Q&amp;A, tool calls. */
        INTERACTIVE,
        /** Nobody is waiting on this one: renders, bulk extraction, evals. */
        BATCH
    }

    /**
     * @param workers             worker processes: the number of requests that run at once
     * @param budgetBytes         the memory the pool's models and workers may use (the GPU's, on a discrete GPU)
     * @param workerOverheadBytes what each worker JVM costs on top of its models
     * @param aging               a waiting request rises one class per this much waiting (null: never)
     * @param idleTimeout         unload a model idle this long (null: never)
     * @param maxQueued           requests that may wait in each class; one more is rejected
     * @param worker              how the worker processes are launched and watched
     */
    public record Options(int workers, long budgetBytes, long workerOverheadBytes, Duration aging, Duration idleTimeout,
                          int maxQueued, Worker.Options worker) {
        public Options {
            if (workers < 1) throw new IllegalArgumentException("workers must be at least 1");
            if (maxQueued < 0) throw new IllegalArgumentException("maxQueued must be at least 0");
            if (aging != null && (aging.isZero() || aging.isNegative())) throw new IllegalArgumentException("aging must be positive, or null");
            Objects.requireNonNull(worker);
        }

        /** Workers and a budget, with 100 MB per worker JVM, 10 s aging, no idle timeout and 10,000 queued per class. */
        public static Options of(int workers, long budgetBytes) {
            return new Options(workers, budgetBytes, 100L << 20, Duration.ofSeconds(10), null, 10_000, Worker.Options.DEFAULT);
        }

        public Options withWorkerOverhead(long bytes) { return new Options(workers, budgetBytes, bytes, aging, idleTimeout, maxQueued, worker); }
        public Options withAging(Duration d) { return new Options(workers, budgetBytes, workerOverheadBytes, d, idleTimeout, maxQueued, worker); }
        public Options withIdleTimeout(Duration d) { return new Options(workers, budgetBytes, workerOverheadBytes, aging, d, maxQueued, worker); }
        public Options withMaxQueued(int n) { return new Options(workers, budgetBytes, workerOverheadBytes, aging, idleTimeout, n, worker); }
        public Options withWorker(Worker.Options o) { return new Options(workers, budgetBytes, workerOverheadBytes, aging, idleTimeout, maxQueued, o); }
    }

    /** A snapshot of one loaded model. @param worker the index of its worker ({@link #worker(int)}) */
    public record Entry(String key, long sizeBytes, int worker, boolean ready, boolean inUse, Instant lastUsed) {}

    /**
     * Counters since the dispatch started, and the queues now.
     * @param hits      requests granted a model that was already loaded
     * @param loads     requests that loaded their model
     * @param evictions models unloaded to make room or after their idle timeout
     * @param crashes   requests during which their worker died
     * @param rejected  requests refused because their class's queue was full
     */
    public record Stats(long requests, long hits, long loads, long evictions, long crashes, long rejected,
                        int queuedInteractive, int queuedBatch, long loadedBytes) {}

    private final Options options;
    private final List<Lane> lanes = new ArrayList<>();
    private final ReentrantLock lock = new ReentrantLock();
    private final Map<String, Model<?>> models = new HashMap<>();          // guarded by lock
    private final Map<String, Loaded> loaded = new LinkedHashMap<>();      // guarded by lock
    private final List<Request> queue = new ArrayList<>();                 // guarded by lock
    private final ScheduledExecutorService reaper;
    private long seq, closingBytes;                                        // guarded by lock
    private long requests, hits, loads, evictions, crashes, rejected;      // guarded by lock
    private boolean closed;                                                // guarded by lock

    public Dispatch(Options options) {
        this.options = Objects.requireNonNull(options);
        for (int i = 0; i < options.workers; i++) lanes.add(new Lane(i, new Worker("dispatch-" + i, options.worker)));
        if (options.idleTimeout != null) {
            reaper = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "fornax-dispatch-reaper");
                t.setDaemon(true);
                return t;
            });
            long period = Math.max(10, Math.min(60_000, options.idleTimeout.toMillis() / 2));
            reaper.scheduleWithFixedDelay(this::unloadIdle, period, period, TimeUnit.MILLISECONDS);
        } else {
            reaper = null;
        }
    }

    /**
     * Makes a model available under {@code key}. Nothing loads until a request for it is granted; then
     * {@code loader} loads it into the worker chosen for it, typically {@code w -> LlamaModel.isolated(w, gguf, ...)}.
     * @param sizeBytes the memory it takes once loaded, as for {@link ModelPool#acquire} ({@link ModelPool#sizeOf})
     */
    public <M extends AutoCloseable> Model<M> register(String key, long sizeBytes, Function<Worker, M> loader) {
        return register(key, sizeBytes, 1, loader);
    }

    /**
     * {@link #register(String, long, Function)} for a model that runs up to {@code concurrency} requests at once, such
     * as {@code w -> LlamaModel.isolated(w, gguf, -1, 4096, 4)} with 4: requests for it share its worker instead of
     * queueing for it. Count its slots' context in {@code sizeBytes} ({@code LlamaModel.kvCacheBytes}).
     */
    public <M extends AutoCloseable> Model<M> register(String key, long sizeBytes, int concurrency, Function<Worker, M> loader) {
        if (concurrency < 1) throw new IllegalArgumentException("concurrency must be at least 1");
        Model<M> m = new Model<>(Objects.requireNonNull(key), sizeBytes, concurrency, Objects.requireNonNull(loader));
        lock.lock();
        try {
            if (closed) throw new IllegalStateException("dispatch is closed");
            if (models.putIfAbsent(key, m) != null) throw new IllegalArgumentException("a model is already registered as " + key);
        } finally { lock.unlock(); }
        return m;
    }

    /** The worker process at {@code index} (0 to workers - 1), as {@link Entry#worker()} names it. */
    public Worker worker(int index) { return lanes.get(index).worker; }

    /** The loaded models (and those loading), least recently used first. */
    public List<Entry> entries() {
        lock.lock();
        try {
            List<Entry> out = new ArrayList<>();
            for (Loaded l : loaded.values()) out.add(new Entry(l.spec.key, l.spec.size, l.lane.index, l.ready, l.inUse > 0, l.lastUsed));
            out.sort(Comparator.comparing(Entry::lastUsed));
            return out;
        } finally { lock.unlock(); }
    }

    public Stats stats() {
        lock.lock();
        try {
            int interactive = 0, batch = 0;
            for (Request r : queue) if (r.qos == QoS.INTERACTIVE) interactive++; else batch++;
            return new Stats(requests, hits, loads, evictions, crashes, rejected, interactive, batch, usedBytes());
        } finally { lock.unlock(); }
    }

    /**
     * Fails every queued request with a {@link CancellationException}, unloads every model (which cancels the calls in
     * flight, as closing a model does) and stops the workers. Idempotent.
     */
    @Override public void close() {
        List<Request> pending;
        List<Loaded> all;
        lock.lock();
        try {
            if (closed) return;
            closed = true;
            pending = new ArrayList<>(queue);
            queue.clear();
            all = new ArrayList<>(loaded.values());
            loaded.clear();
        } finally { lock.unlock(); }
        if (reaper != null) reaper.shutdownNow();
        for (Request r : pending) r.grant.completeExceptionally(new CancellationException("dispatch was closed"));
        for (Loaded l : all) closeQuietly(l.model);
        for (Lane lane : lanes) lane.worker.close();
    }

    /**
     * A model registered with a dispatch: what you make requests for. Each request waits in the dispatch's queue until
     * it is granted the model, which it then has to itself (or shares with up to {@link #concurrency()} - 1 others).
     */
    public final class Model<M extends AutoCloseable> {
        private final String key;
        private final long size;
        private final int concurrency;
        private final Function<Worker, M> loader;

        private Model(String key, long size, int concurrency, Function<Worker, M> loader) {
            this.key = key; this.size = size; this.concurrency = concurrency; this.loader = loader;
        }

        public String key() { return key; }

        /** How many requests it runs at once. */
        public int concurrency() { return concurrency; }

        /**
         * Runs {@code body} with the model once the request is granted, and holds the model until the future it
         * returns completes: use the model's async methods ({@code llm -> llm.chatAsync(...)}). The returned future
         * completes with that future's result. Cancelling it (or {@code orTimeout}) while the request is queued takes
         * it out of the queue; while it runs, it cancels the body's future, which stops a Fornax model's generation
         * (return the model's own future: cancelling one derived from it doesn't). A future that never completes holds
         * the worker for good, and a body that waits on another request to this dispatch can deadlock it.
         * @throws RejectedExecutionException (in the future) if the class's queue is full
         */
        public <T> CompletableFuture<T> submit(QoS qos, Function<? super M, ? extends CompletionStage<T>> body) {
            Objects.requireNonNull(body);
            CompletableFuture<T> out = new CompletableFuture<>();
            CompletableFuture<ModelPool.Lease<M>> grant = acquireAsync(qos);
            AtomicReference<CompletableFuture<T>> inner = new AtomicReference<>();
            out.whenComplete((v, e) -> {             // a no-op once the body's future is done; else as a cancel
                grant.cancel(false);
                CompletableFuture<T> i = inner.get();
                if (i != null) i.cancel(true);
            });
            grant.whenComplete((lease, e) -> {
                if (e != null) { out.completeExceptionally(unwrap(e)); return; }
                Grant<M> g = (Grant<M>) lease;
                CompletableFuture<T> f;
                try {
                    f = Objects.requireNonNull(body.apply(g.model()), "the body returned null").toCompletableFuture();
                } catch (Throwable t) {
                    g.release(t);
                    out.completeExceptionally(t);
                    return;
                }
                inner.set(f);
                if (out.isDone()) f.cancel(true);
                f.whenComplete((v, x) -> {
                    g.release(x);
                    if (x == null) out.complete(v); else out.completeExceptionally(unwrap(x));
                });
            });
            return out;
        }

        /**
         * Waits for the model (a virtual thread unmounts meanwhile), runs {@code body} on this thread with it, and gives
         * it back: for the blocking methods. Interrupting the wait cancels the request.
         */
        public <T> T call(QoS qos, Function<? super M, T> body) {
            Objects.requireNonNull(body);
            Grant<M> g = (Grant<M>) acquire(qos);
            Throwable failure = null;
            try {
                return body.apply(g.model());
            } catch (Throwable t) {
                failure = t;
                throw t;
            } finally {
                g.release(failure);
            }
        }

        /** {@link #acquireAsync}, waited for. Close the lease to give the model back. */
        public ModelPool.Lease<M> acquire(QoS qos) {
            CompletableFuture<ModelPool.Lease<M>> grant = acquireAsync(qos);
            try {
                return grant.get();
            } catch (InterruptedException e) {
                if (!grant.cancel(false)) grant.join().close();
                Thread.currentThread().interrupt();
                throw new CancellationException("interrupted waiting for " + key);
            } catch (ExecutionException e) {
                throw asUnchecked(e.getCause());
            }
        }

        /**
         * Queues a request for the model, completing with a lease on it when granted (the model is loaded by then).
         * Close the lease to give it back; until you do, nothing else runs on its worker. Cancelling the future while
         * the request is queued takes it out of the queue. If the worker isn't running when the lease closes (it died
         * while the lease was held), its models are dropped, as for {@link #submit}.
         */
        @SuppressWarnings({"unchecked", "rawtypes"})
        public CompletableFuture<ModelPool.Lease<M>> acquireAsync(QoS qos) {
            Objects.requireNonNull(qos);
            Request r;
            lock.lock();
            try {
                if (closed) return CompletableFuture.failedFuture(new IllegalStateException("dispatch is closed"));
                if (models.get(key) != this) return CompletableFuture.failedFuture(new IllegalStateException(key + " isn't registered with this dispatch"));
                int waiting = 0;
                for (Request q : queue) if (q.qos == qos) waiting++;
                if (waiting >= options.maxQueued) {
                    rejected++;
                    return CompletableFuture.failedFuture(new RejectedExecutionException(
                            options.maxQueued + " " + qos + " requests are already queued"));
                }
                r = new Request(this, qos, System.nanoTime(), seq++);
                queue.add(r);
                requests++;
            } finally { lock.unlock(); }
            r.grant.whenComplete((l, e) -> {         // granted, or ended by cancel, orTimeout or completeOnTimeout
                boolean wasQueued;
                lock.lock();
                try { wasQueued = queue.remove(r); } finally { lock.unlock(); }
                if (wasQueued) reschedule();         // it may have been holding back loads ranked below it
            });
            reschedule();
            return (CompletableFuture) r.grant;
        }
    }

    // ---- scheduling ----

    /** One worker process, which runs one request at a time (or several, for one model with a concurrency). */
    private static final class Lane {
        final int index;
        final Worker worker;
        Loaded running;           // guarded by lock: the model of the requests running here (or loading for one)
        boolean crashed;          // guarded by lock: it died during the last request here

        Lane(int index, Worker worker) { this.index = index; this.worker = worker; }
    }

    /** A model loaded (or loading) in a lane. */
    private static final class Loaded {
        final Model<?> spec;
        final Lane lane;
        volatile AutoCloseable model;
        boolean ready;            // guarded by lock: loaded
        int inUse;                // guarded by lock: requests granted it and not yet released
        long lastUsedNanos = System.nanoTime();
        Instant lastUsed = Instant.now();
        volatile boolean started; // the worker had started when the current request was granted it

        Loaded(Model<?> spec, Lane lane) { this.spec = spec; this.lane = lane; }
    }

    private static final class Request {
        final Model<?> spec;
        final QoS qos;
        final long enqueued, seq;
        final CompletableFuture<Object> grant = new CompletableFuture<>();

        Request(Model<?> spec, QoS qos, long enqueued, long seq) {
            this.spec = spec; this.qos = qos; this.enqueued = enqueued; this.seq = seq;
        }
    }

    /** Grants what can be granted now, then starts each grant on a virtual thread of its own. */
    private void reschedule() {
        List<Runnable> starts;
        lock.lock();
        try { starts = plan(); } finally { lock.unlock(); }
        for (Runnable s : starts) Thread.ofVirtual().name("fornax-dispatch").start(s);
    }

    /** Caller holds lock. */
    private List<Runnable> plan() {
        List<Runnable> starts = new ArrayList<>();
        if (closed) return starts;
        queue.removeIf(r -> r.grant.isDone());
        long now = System.nanoTime();
        List<Request> order = new ArrayList<>(queue);
        order.sort(Comparator.<Request>comparingLong(r -> rank(r, now)).thenComparingLong(r -> r.seq));
        boolean roomBlocked = false, workerWanted = false;
        java.util.Set<Lane> laneWanted = new java.util.HashSet<>();
        for (Request r : order) {
            Loaded l = loaded.get(r.spec.key);
            if (l != null) {                                     // affinity: wait for the model's own worker
                if (l.lane.running != null) {
                    // a model with a concurrency takes more requests while it runs, unless one ranked above them is
                    // waiting for this worker (or any, or for room), which would otherwise wait for as long as they
                    // keep coming
                    boolean join = l.lane.running == l && l.ready && l.inUse < l.spec.concurrency
                            && !workerWanted && !roomBlocked && !laneWanted.contains(l.lane);
                    if (!join) {
                        if (l.lane.running != l) laneWanted.add(l.lane);
                        continue;
                    }
                }
                queue.remove(r);
                l.lane.running = l;
                l.inUse++;
                hits++;
                starts.add(() -> start(r, l, List.of()));
                continue;
            }
            if (roomBlocked) continue;                           // a request ranked higher is waiting for room
            Lane lane = idleLane();
            if (lane == null) {
                workerWanted = true;
                continue;
            }
            long free = capacity() - usedBytes();
            List<Loaded> victims = new ArrayList<>();
            if (free < r.spec.size) {
                List<Loaded> idle = new ArrayList<>();
                for (Loaded c : loaded.values()) if (c.ready && c.lane.running != c) idle.add(c);
                idle.sort(Comparator.comparingLong(c -> c.lastUsedNanos));
                for (Loaded c : idle) {
                    if (free >= r.spec.size) break;
                    victims.add(c);
                    free += c.spec.size;
                }
                long held = usedBytes();
                for (Loaded v : victims) held -= v.spec.size;
                if (free < r.spec.size && held > 0) {            // it fits once a model in use is done: wait for that
                    roomBlocked = true;
                    continue;
                }
            }
            for (Loaded v : victims) {
                loaded.remove(v.spec.key);
                closingBytes += v.spec.size;
                evictions++;
            }
            Loaded nl = new Loaded(r.spec, lane);
            loaded.put(r.spec.key, nl);
            lane.running = nl;
            nl.inUse = 1;
            queue.remove(r);
            loads++;
            starts.add(() -> start(r, nl, victims));
        }
        return starts;
    }

    /** Lower goes first: the class's ordinal, less one for each aging period waited. */
    private long rank(Request r, long now) {
        long base = r.qos.ordinal();
        if (options.aging == null) return base;
        return Math.max(0, base - (now - r.enqueued) / options.aging.toNanos());
    }

    /** Caller holds lock. An idle worker: not one that just crashed (it backs off), then the least loaded, then the first. */
    private Lane idleLane() {
        Lane best = null;
        long bestBytes = 0;
        for (Lane lane : lanes) {
            if (lane.running != null) continue;
            long bytes = 0;
            for (Loaded l : loaded.values()) if (l.lane == lane) bytes += l.spec.size;
            if (best == null || (best.crashed && !lane.crashed) || (best.crashed == lane.crashed && bytes < bestBytes)) {
                best = lane;
                bestBytes = bytes;
            }
        }
        return best;
    }

    /** Caller holds lock. */
    private long capacity() { return options.budgetBytes - options.workers * options.workerOverheadBytes; }

    /** Caller holds lock: models loaded or loading, and those still being unloaded. */
    private long usedBytes() {
        long n = closingBytes;
        for (Loaded l : loaded.values()) n += l.spec.size;
        return n;
    }

    /** On a virtual thread: unloads the victims, loads the model if need be, and hands the request its lease. */
    private void start(Request r, Loaded l, List<Loaded> victims) {
        for (Loaded v : victims) {
            closeQuietly(v.model);
            lock.lock();
            try { closingBytes -= v.spec.size; } finally { lock.unlock(); }
        }
        if (!victims.isEmpty()) reschedule();       // a load may have been waiting for the room they freed
        Lane lane = l.lane;
        if (l.model == null) {
            try {
                l.model = l.spec.loader.apply(lane.worker);
            } catch (Throwable t) {
                List<Loaded> dropped = List.of();
                lock.lock();
                try {
                    loaded.remove(l.spec.key, l);
                    l.inUse = 0;
                    lane.running = null;
                    if (isCrash(t)) dropped = crashedLocked(lane);
                } finally { lock.unlock(); }
                for (Loaded d : dropped) closeQuietly(d.model);
                r.grant.completeExceptionally(t);
                reschedule();
                return;
            }
        }
        boolean closedMeanwhile;
        lock.lock();
        try {
            closedMeanwhile = closed;
            l.ready = true;
            l.started = lane.worker.starts() > 0;
        } finally { lock.unlock(); }
        if (closedMeanwhile) {                       // close() cleared the models before this one finished loading
            closeQuietly(l.model);
            r.grant.completeExceptionally(new CancellationException("dispatch was closed"));
            return;
        }
        Grant<?> g = new Grant<>(l);
        if (!r.grant.complete(g)) g.release(null);  // cancelled meanwhile: the model stays loaded
        else if (l.spec.concurrency > 1) reschedule();   // ready now: requests waiting for it may share it
    }

    /** Gives a model back, after a request; drops the worker's models if it died meanwhile. */
    private void release(Loaded l, Throwable failure) {
        Lane lane = l.lane;
        Worker w = lane.worker;
        // The worker is running after any call that didn't crash it (one that died while idle restarts for the call).
        boolean crash = isCrash(failure) || (l.started && !w.isClosed() && !w.isRunning());
        List<Loaded> dropped = List.of();
        lock.lock();
        try {
            if (l.inUse > 0) l.inUse--;
            if (lane.running == l && l.inUse == 0) lane.running = null;
            l.lastUsedNanos = System.nanoTime();
            l.lastUsed = Instant.now();
            if (crash && !closed) dropped = crashedLocked(lane);
            else if (failure == null) lane.crashed = false;
        } finally { lock.unlock(); }
        for (Loaded d : dropped) closeQuietly(d.model);
        reschedule();
    }

    /** Caller holds lock: the lane's worker died, and its models with it. Returns them, to close (which is quick now). */
    private List<Loaded> crashedLocked(Lane lane) {
        crashes++;
        lane.crashed = true;
        List<Loaded> dropped = new ArrayList<>();
        for (var it = loaded.values().iterator(); it.hasNext(); ) {
            Loaded l = it.next();
            if (l.lane == lane && lane.running != l) {
                dropped.add(l);
                it.remove();
            }
        }
        return dropped;
    }

    private void unloadIdle() {
        long cutoff = System.nanoTime() - options.idleTimeout.toNanos();
        List<Loaded> idle = new ArrayList<>();
        lock.lock();
        try {
            for (var it = loaded.values().iterator(); it.hasNext(); ) {
                Loaded l = it.next();
                if (l.ready && l.lane.running != l && l.lastUsedNanos - cutoff < 0) {
                    idle.add(l);
                    it.remove();
                    closingBytes += l.spec.size;
                    evictions++;
                }
            }
        } finally { lock.unlock(); }
        if (idle.isEmpty()) return;
        for (Loaded l : idle) {
            closeQuietly(l.model);
            lock.lock();
            try { closingBytes -= l.spec.size; } finally { lock.unlock(); }
        }
        reschedule();
    }

    /** A granted request's hold on its model. */
    private final class Grant<M> implements ModelPool.Lease<M> {
        private final Loaded loaded;
        private boolean released;   // guarded by this

        Grant(Loaded loaded) { this.loaded = loaded; }

        @SuppressWarnings("unchecked")
        @Override public M model() {
            synchronized (this) { if (released) throw new IllegalStateException("lease already closed"); }
            return (M) loaded.model;
        }

        @Override public void close() { release(null); }

        void release(Throwable failure) {
            synchronized (this) {
                if (released) return;
                released = true;
            }
            Dispatch.this.release(loaded, failure);
        }
    }

    private static boolean isCrash(Throwable t) {
        for (Throwable c = t; c != null; c = c.getCause()) if (c instanceof WorkerCrashedException) return true;
        return false;
    }

    private static Throwable unwrap(Throwable e) {
        return e instanceof CompletionException && e.getCause() != null ? e.getCause() : e;
    }

    private static RuntimeException asUnchecked(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new CompletionException(t);
    }

    private static void closeQuietly(AutoCloseable c) {
        if (c == null) return;
        try { c.close(); } catch (Exception ignored) {
            // unloading is best effort: a model whose worker died closes with nothing left to free
        }
    }
}
