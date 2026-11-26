package org.liberationdata.fornax;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.StandardProtocolFamily;
import java.net.UnixDomainSocketAddress;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Consumer;
import java.util.stream.Stream;

/**
 * A child JVM that hosts engine instances ({@link WorkerService}s) so that a native crash in one of them (an abort, a
 * segfault, a failed {@code GGML_ASSERT}) kills only the worker. The calls in flight then fail with a
 * {@link WorkerCrashedException}; the next call starts a new worker and loads its models again. You mostly use one
 * through an engine's isolated model ({@code LlamaModel.isolated(...)}), which owns a worker of its own.
 *
 * <ul>
 *   <li><b>Process.</b> {@code java} from this JVM's {@code java.home}, with the same class path, every {@code -Dfornax.*}
 *   property, the same environment (so {@code FORNAX_*} too), and {@code --enable-native-access=ALL-UNNAMED}: it finds
 *   the same natives this JVM would. It starts on first use.</li>
 *   <li><b>Transport.</b> A Unix domain socket in a private temporary directory, for calls, results, cancels and each
 *   streamed item. Payloads over {@link Options#sharedMemoryThreshold} (images, audio) go through a memory-mapped file
 *   on {@code /dev/shm} instead.</li>
 *   <li><b>Health.</b> A ping every {@link Options#heartbeat}; a worker that answers nothing for
 *   {@link Options#heartbeatTimeout} is killed, and counts as crashed. {@link #ping()} measures a round trip.</li>
 *   <li><b>Restart.</b> After a crash the next call starts a new worker, waiting first for a backoff that doubles with
 *   each crash in a row (from {@link Options#backoff} to {@link Options#maxBackoff}) and resets after a call succeeds.
 *   A model that crashes the worker as it loads can't make it spin.</li>
 *   <li><b>Reaping.</b> The worker exits when its standard input or its socket closes: when this JVM closes it, or
 *   dies.</li>
 * </ul>
 */
public final class Worker implements AutoCloseable {
    /**
     * How a worker is launched and watched.
     * @param jvmArgs               more JVM options for the worker ({@code -Xmx2g}, ...), after Fornax's own
     * @param classPath             the worker's class path: this JVM's {@code java.class.path} by default
     * @param heartbeat             how often to ping the worker
     * @param heartbeatTimeout      how long the worker may stay silent before it is killed
     * @param startTimeout          how long the worker has to start and connect
     * @param sharedMemoryThreshold payloads bigger than this many bytes go through shared memory
     * @param backoff               the wait before the first restart after a crash; it doubles with each crash in a row
     * @param maxBackoff            the longest wait before a restart
     * @param echoOutput            copy the worker's output (the engines' logs) to this JVM's standard error
     */
    public record Options(List<String> jvmArgs, String classPath, Duration heartbeat, Duration heartbeatTimeout,
                          Duration startTimeout, int sharedMemoryThreshold, Duration backoff, Duration maxBackoff,
                          boolean echoOutput) {
        public static final Options DEFAULT = new Options(List.of(), null, Duration.ofSeconds(2), Duration.ofSeconds(30),
                Duration.ofSeconds(60), 64 << 10, Duration.ofMillis(250), Duration.ofSeconds(30), true);

        public Options {
            jvmArgs = List.copyOf(jvmArgs);
            Objects.requireNonNull(heartbeat);
            Objects.requireNonNull(heartbeatTimeout);
            Objects.requireNonNull(startTimeout);
            Objects.requireNonNull(backoff);
            Objects.requireNonNull(maxBackoff);
            if (sharedMemoryThreshold < 0) throw new IllegalArgumentException("sharedMemoryThreshold must be at least 0");
        }

        public Options withJvmArgs(String... args) { return new Options(List.of(args), classPath, heartbeat, heartbeatTimeout, startTimeout, sharedMemoryThreshold, backoff, maxBackoff, echoOutput); }
        public Options withClassPath(String cp) { return new Options(jvmArgs, cp, heartbeat, heartbeatTimeout, startTimeout, sharedMemoryThreshold, backoff, maxBackoff, echoOutput); }
        public Options withHeartbeat(Duration every, Duration timeout) { return new Options(jvmArgs, classPath, every, timeout, startTimeout, sharedMemoryThreshold, backoff, maxBackoff, echoOutput); }
        public Options withSharedMemoryThreshold(int bytes) { return new Options(jvmArgs, classPath, heartbeat, heartbeatTimeout, startTimeout, bytes, backoff, maxBackoff, echoOutput); }
        public Options withBackoff(Duration first, Duration max) { return new Options(jvmArgs, classPath, heartbeat, heartbeatTimeout, startTimeout, sharedMemoryThreshold, first, max, echoOutput); }
        public Options withEchoOutput(boolean echo) { return new Options(jvmArgs, classPath, heartbeat, heartbeatTimeout, startTimeout, sharedMemoryThreshold, backoff, maxBackoff, echo); }
    }

    private static final int TAIL_LINES = 40;
    private static final ScheduledExecutorService HEARTBEATS = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread t = new Thread(r, "fornax-worker-heartbeat");
        t.setDaemon(true);
        return t;
    });

    private final String name;
    private final Options options;
    private final ReentrantLock lock = new ReentrantLock();
    private final java.util.concurrent.locks.Condition changed = lock.newCondition();
    private final AtomicLong ids = new AtomicLong();
    private final Map<Long, Instance> instances = new ConcurrentHashMap<>();
    private Incarnation current;          // guarded by lock
    private int generation;               // guarded by lock: the number of workers started
    private int crashesInARow;            // guarded by lock
    private long lastCrashNanos;          // guarded by lock
    private boolean closed;               // guarded by lock

    public Worker() { this("worker", Options.DEFAULT); }

    /** @param name for the worker's threads and messages (an engine passes its model file's name) */
    public Worker(String name, Options options) {
        this.name = Objects.requireNonNull(name);
        this.options = Objects.requireNonNull(options);
    }

    /**
     * Creates a service in the worker (starting the worker if it isn't running) and returns a handle to it. The
     * worker builds it with {@code new serviceClass(config)}; after a crash, the next call builds it again in the new
     * worker from the same config.
     * @throws WorkerCrashedException if the worker died while creating it (a model that crashes its engine as it loads)
     */
    public Instance open(String serviceClass, byte[] config) {
        Instance i = new Instance(ids.incrementAndGet(), serviceClass, config == null ? new byte[0] : config);
        instances.put(i.id, i);
        try {
            i.ensureOpen();
        } catch (RuntimeException | Error e) {
            instances.remove(i.id);
            throw e;
        }
        return i;
    }

    /** The round trip of a ping to the worker, starting it if it isn't running: a health check. */
    public Duration ping() {
        Incarnation inc = running();
        long id = ids.incrementAndGet();
        Pending p = inc.register(id);
        long start = System.nanoTime();
        try {
            inc.send(Frames.PING, id, 0, null, null);
            p.await(null, null, inc);
            return Duration.ofNanos(System.nanoTime() - start);
        } finally {
            inc.pending.remove(id);
        }
    }

    /** Whether a worker process is running now. */
    public boolean isRunning() {
        lock.lock();
        try { return current != null && !current.dead; } finally { lock.unlock(); }
    }

    /** The running worker's process id, or -1. */
    public long pid() {
        lock.lock();
        try { return current != null && !current.dead ? current.process.pid() : -1; } finally { lock.unlock(); }
    }

    /** How many worker processes have been started, counting the first: one more than the restarts. */
    public int starts() {
        lock.lock();
        try { return generation; } finally { lock.unlock(); }
    }

    /** The running worker's shared-memory directory, or null: for tests. */
    Path sharedDir() {
        lock.lock();
        try { return current != null && !current.dead ? current.sharedDir : null; } finally { lock.unlock(); }
    }

    /** Whether {@link #close()} has been called. */
    public boolean isClosed() {
        lock.lock();
        try { return closed; } finally { lock.unlock(); }
    }

    /**
     * Stops the worker: its models are freed with the process, and calls in flight fail with an
     * {@link IllegalStateException}. Idempotent.
     */
    @Override public void close() {
        Incarnation inc;
        lock.lock();
        try {
            closed = true;
            inc = current;
            current = null;
            changed.signalAll();
        } finally { lock.unlock(); }
        if (inc != null) inc.stop();
    }

    // ---- one worker process ----

    /** The running worker, starting one (after the backoff, if the last one crashed) if there is none. */
    private Incarnation running() {
        lock.lock();
        try {
            while (true) {
                if (closed) throw new IllegalStateException("worker " + name + " is closed");
                if (current != null && !current.dead) return current;
                long wait = crashesInARow == 0 ? 0 : backoffNanos() - (System.nanoTime() - lastCrashNanos);
                if (wait <= 0) break;
                changed.awaitNanos(wait);
            }
            generation++;
            try {
                current = new Incarnation(generation);
            } catch (WorkerCrashedException e) {
                crashed();
                throw e;
            }
            return current;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted waiting to restart worker " + name, e);
        } finally { lock.unlock(); }
    }

    private long backoffNanos() {
        long first = options.backoff.toNanos(), max = options.maxBackoff.toNanos();
        int doublings = Math.min(crashesInARow - 1, 30);
        return Math.min(max, first << doublings < 0 ? max : first << doublings);
    }

    /** Caller holds lock. */
    private void crashed() {
        crashesInARow++;
        lastCrashNanos = System.nanoTime();
    }

    private void succeeded() {
        lock.lock();
        try { crashesInARow = 0; } finally { lock.unlock(); }
    }

    private final class Incarnation {
        final int gen;
        final Process process;
        final SocketChannel channel;
        final Path dir, sharedDir;
        final Frames.Writer writer;
        final Map<Long, Pending> pending = new ConcurrentHashMap<>();
        final ArrayDeque<String> tail = new ArrayDeque<>();     // guarded by itself
        final CountDownLatch gone = new CountDownLatch(1);
        final Thread pump;
        final ScheduledFuture<?> heartbeat;
        volatile long lastHeard = System.nanoTime();
        volatile boolean dead, stopping, unresponsive;
        volatile RuntimeException death;

        Incarnation(int gen) {
            this.gen = gen;
            Path dir = null, shared = null;
            Process p = null;
            try {
                dir = Files.createTempDirectory("fornax-worker-");  // owner-only on POSIX file systems
                Path dev = Path.of("/dev/shm");
                shared = Files.isDirectory(dev) && Files.isWritable(dev) ? Files.createTempDirectory(dev, "fornax-worker-") : dir;
                Path sock = dir.resolve("s");
                if (sock.toString().length() > 100) {  // sun_path is 104 to 108 bytes
                    dir = Files.createTempDirectory(Path.of("/tmp"), "fornax-worker-");
                    sock = dir.resolve("s");
                }
                try (ServerSocketChannel server = ServerSocketChannel.open(StandardProtocolFamily.UNIX)) {
                    server.bind(UnixDomainSocketAddress.of(sock));
                    p = new ProcessBuilder(command(sock, shared)).redirectErrorStream(true).start();
                    this.process = p;
                    this.dir = dir;
                    this.sharedDir = shared;
                    this.pump = Thread.ofPlatform().name("fornax-worker-output-" + name).daemon(true).start(this::pumpOutput);
                    this.channel = accept(server, p);
                } finally {
                    Files.deleteIfExists(sock);
                }
            } catch (IOException e) {
                if (p != null) p.destroyForcibly();
                deleteQuietly(dir);
                deleteQuietly(shared);
                throw new UncheckedIOException("could not start worker " + name, e);
            }
            this.writer = new Frames.Writer(channel, sharedDir, options.sharedMemoryThreshold, "p");
            long every = Math.max(1, options.heartbeat.toMillis());
            this.heartbeat = HEARTBEATS.scheduleWithFixedDelay(this::beat, every, every, TimeUnit.MILLISECONDS);
            // after the heartbeat is set: a worker that dies at once has the reader call died(), which cancels it
            Thread.ofPlatform().name("fornax-worker-reader-" + name).daemon(true).start(this::readFrames);
        }

        private List<String> command(Path sock, Path shared) {
            List<String> cmd = new ArrayList<>();
            cmd.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
            cmd.add("--enable-native-access=ALL-UNNAMED");
            // A crash report, not a core file: a core of a process holding a model's weights is gigabytes.
            cmd.add("-XX:-CreateCoredumpOnCrash");
            cmd.add("-XX:ErrorFile=" + Path.of(System.getProperty("java.io.tmpdir"), "fornax-worker-hs_err_pid%p.log"));
            for (String key : System.getProperties().stringPropertyNames().stream().sorted().toList())
                if (key.startsWith("fornax.") && !key.equals("fornax.isolation")) cmd.add("-D" + key + "=" + System.getProperty(key));
            cmd.add("-Dfornax.isolation=inprocess");
            cmd.addAll(options.jvmArgs);
            cmd.add("-cp");
            cmd.add(options.classPath != null ? options.classPath : System.getProperty("java.class.path"));
            cmd.add(WorkerMain.class.getName());
            cmd.add(sock.toString());
            cmd.add(shared.toString());
            cmd.add(Integer.toString(options.sharedMemoryThreshold));
            return cmd;
        }

        /** The worker's connection, or a WorkerCrashedException if it exits or times out first. */
        private SocketChannel accept(ServerSocketChannel server, Process p) throws IOException {
            server.configureBlocking(false);
            long deadline = System.nanoTime() + options.startTimeout.toNanos();
            try (Selector sel = Selector.open()) {
                server.register(sel, SelectionKey.OP_ACCEPT);
                while (true) {
                    sel.select(50);
                    SocketChannel ch = server.accept();
                    if (ch != null) {
                        ch.configureBlocking(true);
                        return ch;
                    }
                    if (!p.isAlive() || System.nanoTime() > deadline) {
                        boolean exited = !p.isAlive();
                        if (!exited) p.destroyForcibly();
                        waitFor(p, Duration.ofSeconds(5));
                        joinQuietly(pump, 2000);
                        deleteQuietly(dir);
                        if (sharedDir != dir) deleteQuietly(sharedDir);
                        throw new WorkerCrashedException(name, p.pid(), exitCode(p),
                                exited ? "exited as it started" : "did not connect within " + options.startTimeout, tail());
                    }
                }
            }
        }

        private void pumpOutput() {
            try (BufferedReader r = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = r.readLine()) != null) {
                    synchronized (tail) {
                        tail.addLast(line);
                        if (tail.size() > TAIL_LINES) tail.removeFirst();
                    }
                    if (options.echoOutput) System.err.println(line);
                }
            } catch (IOException ignored) {
                // the process is gone
            }
        }

        List<String> tail() {
            synchronized (tail) { return new ArrayList<>(tail); }
        }

        private void readFrames() {
            Throwable failure = null;
            try (Frames.Reader reader = new Frames.Reader(channel, sharedDir)) {
                while (true) {
                    Frames.Frame f = reader.read();
                    lastHeard = System.nanoTime();
                    Pending p = pending.get(f.id());
                    if (p != null) p.queue.add(f);
                }
            } catch (Throwable e) {  // anything that ends this thread must fail the calls waiting on it
                failure = e;
            }
            died(failure);
        }

        private void beat() {
            if (dead) return;
            if (System.nanoTime() - lastHeard > options.heartbeatTimeout.toNanos()) {
                unresponsive = true;
                process.destroyForcibly();  // the reader sees the socket close, and reports the crash
                return;
            }
            try { writer.trySend(Frames.PING, 0); } catch (IOException ignored) { /* the reader reports it */ }
        }

        void send(byte type, long id, long ref, String method, byte[] payload) {
            try {
                writer.send(type, id, ref, method, payload);
            } catch (IOException e) {
                // the worker is gone: report its crash, which the reader is about to establish
                awaitGone();
                throw death instanceof WorkerCrashedException w ? w.copy() : new IllegalStateException("worker " + name + " is closed", e);
            }
        }

        Pending register(long id) {
            Pending p = new Pending();
            pending.put(id, p);
            if (dead) p.queue.add(death);  // it died just now: the reader has already failed the others
            return p;
        }

        /** The process is gone (or the socket broke): fail everything in flight with why. */
        private void died(Throwable cause) {
            heartbeat.cancel(false);
            try { channel.close(); } catch (IOException ignored) {}
            if (!waitFor(process, Duration.ofSeconds(stopping ? 5 : 10))) {
                process.destroyForcibly();
                waitFor(process, Duration.ofSeconds(5));
            }
            joinQuietly(pump, 2000);
            writer.close();
            RuntimeException why;
            if (stopping) {
                why = new IllegalStateException("worker " + name + " was closed");
            } else {
                String reason = unresponsive ? "stopped answering for " + options.heartbeatTimeout.toMillis() + " ms and was killed"
                        : process.isAlive() ? "closed its connection (" + cause + ")" : "crashed";
                why = new WorkerCrashedException(name, process.pid(), exitCode(process), reason, tail());
            }
            death = why;
            dead = true;
            lock.lock();
            try {
                if (current == this) current = null;
                if (!stopping) crashed();
                changed.signalAll();
            } finally { lock.unlock(); }
            for (Pending p : pending.values()) p.queue.add(why);
            gone.countDown();
            deleteQuietly(dir);
            if (sharedDir != dir) deleteQuietly(sharedDir);
        }

        void awaitGone() {
            try {
                if (!gone.await(30, TimeUnit.SECONDS)) throw new IllegalStateException("worker " + name + " did not finish dying");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(e);
            }
        }

        /** Closes the worker: EOF on its standard input and its socket makes it halt. */
        void stop() {
            stopping = true;
            try { OutputStream in = process.getOutputStream(); in.close(); } catch (IOException ignored) {}
            try { channel.close(); } catch (IOException ignored) {}
            awaitGone();
        }
    }

    /** The frames (or the crash) for one call, which its caller waits for. */
    private final class Pending {
        final LinkedBlockingQueue<Object> queue = new LinkedBlockingQueue<>();

        /**
         * Waits for the result, handing events to the caller's consumer on the caller's thread. A consumer that
         * throws stops the call: the exception goes to call (which ModelThread rethrows) or, without one, is thrown
         * once the worker has answered.
         */
        byte[] await(Consumer<byte[]> events, Call call, Incarnation inc) {
            Throwable failed = null;
            boolean interrupted = false;
            try {
                while (true) {
                    Object o;
                    try { o = queue.take(); }
                    catch (InterruptedException e) {
                        interrupted = true;
                        if (call != null) call.cancel();
                        continue;
                    }
                    if (o instanceof WorkerCrashedException w) throw w.copy();
                    if (o instanceof RuntimeException r) throw r;
                    Frames.Frame f = (Frames.Frame) o;
                    switch (f.type()) {
                        case Frames.EVENT -> {
                            if (failed != null || events == null) continue;
                            try { events.accept(f.payload()); }
                            catch (Throwable t) {
                                failed = t;
                                if (call != null) call.fail(t);
                                else inc.send(Frames.CANCEL, f.id(), 0, null, null);
                            }
                        }
                        case Frames.RESULT, Frames.PONG -> {
                            if (failed != null && call == null) throwUnchecked(failed);
                            return f.payload();
                        }
                        case Frames.ERROR -> {
                            if (failed != null && call == null) throwUnchecked(failed);
                            throw Frames.decodeError(f.payload());
                        }
                        default -> throw new IllegalStateException("unexpected frame " + f.type());
                    }
                }
            } finally {
                if (interrupted) Thread.currentThread().interrupt();
            }
        }
    }

    private static void throwUnchecked(Throwable t) {
        if (t instanceof RuntimeException r) throw r;
        if (t instanceof Error e) throw e;
        throw new RuntimeException(t);
    }

    /**
     * One service in the worker: a loaded model. Calls run one at a time, in arrival order. After a worker crash, the
     * next call creates the service again in the new worker (reloading its model) before it runs.
     */
    public final class Instance implements AutoCloseable {
        private final long id;
        private final String serviceClass;
        private final byte[] config;
        private final ReentrantLock calls = new ReentrantLock(true);
        private int openedIn = -1;              // guarded by calls: the worker generation it was created in
        private volatile long inFlight = -1;
        private volatile Incarnation inFlightOn;
        private volatile boolean closed;

        private Instance(long id, String serviceClass, byte[] config) {
            this.id = id; this.serviceClass = serviceClass; this.config = config;
        }

        /** The worker this instance lives in. */
        public Worker worker() { return Worker.this; }

        /** {@link #invoke(String, byte[], Call, Consumer)} with no events and no Call. */
        public byte[] invoke(String method, byte[] args) { return invoke(method, args, null, null); }

        /**
         * Runs a call in the worker and waits for its result, blocking (a virtual thread unmounts meanwhile).
         * Events are handed to {@code events} on this thread, in order. A consumer that throws stops the call.
         * @param call the ModelThread call this runs in, if any: a consumer's exception is recorded there, and its
         *             cancel arrives through {@link #cancel()} (pass {@code instance::cancel} as the ModelThread's
         *             native cancel)
         * @throws WorkerCrashedException if the worker died during the call (or while restarting for it)
         * @throws RuntimeException the worker's exception, as the same class where it can be rebuilt here
         *         ({@link IllegalStateException}, {@link java.util.concurrent.CancellationException}, ...)
         */
        public byte[] invoke(String method, byte[] args, Call call, Consumer<byte[]> events) {
            calls.lock();
            try {
                if (closed) throw new IllegalStateException("this instance is closed");
                Incarnation inc = ensureOpen();
                long callId = ids.incrementAndGet();
                Pending p = inc.register(callId);
                inFlightOn = inc;
                inFlight = callId;
                try {
                    inc.send(Frames.CALL, callId, id, method, args);
                    if (call != null && call.stopRequested()) cancel();  // a cancel that came before inFlight was set
                    byte[] result = p.await(events, call, inc);
                    succeeded();
                    return result;
                } finally {
                    inFlight = -1;
                    inc.pending.remove(callId);
                }
            } finally {
                calls.unlock();
            }
        }

        /** Stops the call in flight, if any, as the engine's own cancel does (an LLM returns the text so far). */
        public void cancel() {
            long c = inFlight;
            Incarnation inc = inFlightOn;
            if (c < 0 || inc == null || inc.dead) return;
            try { inc.send(Frames.CANCEL, c, 0, null, null); } catch (RuntimeException ignored) { /* it died: the call fails anyway */ }
        }

        /** The running worker, with this service created in it. Caller holds calls (or is open()). */
        private Incarnation ensureOpen() {
            Incarnation inc = running();
            if (openedIn == inc.gen) return inc;
            long callId = ids.incrementAndGet();
            Pending p = inc.register(callId);
            try {
                inc.send(Frames.OPEN, callId, id, serviceClass, config);
                p.await(null, null, inc);
                openedIn = inc.gen;
                return inc;
            } finally {
                inc.pending.remove(callId);
            }
        }

        /** Frees the service in the worker (the worker keeps running). Idempotent. */
        @Override public void close() {
            if (closed) return;
            cancel();
            calls.lock();
            try {
                if (closed) return;
                closed = true;
                instances.remove(id);
                Incarnation inc;
                lock.lock();
                try { inc = current; } finally { lock.unlock(); }
                if (inc == null || inc.dead || openedIn != inc.gen) return;
                long callId = ids.incrementAndGet();
                Pending p = inc.register(callId);
                try {
                    inc.send(Frames.FREE, callId, id, null, null);
                    p.await(null, null, inc);
                } catch (WorkerCrashedException | IllegalStateException gone) {
                    // the worker went with it
                } finally {
                    inc.pending.remove(callId);
                }
            } finally {
                calls.unlock();
            }
        }
    }

    // ---- helpers ----

    private static int exitCode(Process p) {
        try { return p.exitValue(); } catch (IllegalThreadStateException running) { return -1; }
    }

    private static boolean waitFor(Process p, Duration d) {
        try { return p.waitFor(d.toMillis(), TimeUnit.MILLISECONDS); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); return !p.isAlive(); }
    }

    private static void joinQuietly(Thread t, long millis) {
        if (t == null) return;
        try { t.join(millis); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }

    private static void deleteQuietly(Path dir) {
        if (dir == null || !Files.exists(dir)) return;
        try (Stream<Path> s = Files.walk(dir)) {
            for (Path p : s.sorted(Comparator.reverseOrder()).toList()) Files.deleteIfExists(p);
        } catch (IOException ignored) {
            // a temp directory; best effort
        }
    }
}
