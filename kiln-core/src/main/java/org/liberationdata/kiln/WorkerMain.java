package org.liberationdata.kiln;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.net.UnixDomainSocketAddress;
import java.nio.channels.SocketChannel;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The worker process's main class, which {@link Worker} launches: it connects back to the parent's socket and hosts
 * {@link WorkerService}s. Not for direct use.
 *
 * <p>It exits as soon as its standard input or its socket closes, which is when the parent closes the worker or dies
 * (the OS closes a dead process's pipes and sockets): a worker never outlives its parent. It halts rather than exits,
 * so a native call still running can't hold the process up.
 */
public final class WorkerMain {
    private record Hosted(WorkerService service, ModelThread thread) {}

    /** A call the parent made: cancelled marks it whether or not it has started here yet. */
    private static final class Running {
        volatile Call call;           // set as it starts
        volatile boolean cancelled;
    }

    private final SocketChannel channel;
    private final Path sharedDir;
    private final Frames.Writer out;
    private final Map<Long, Hosted> instances = new ConcurrentHashMap<>();
    private final Map<Long, Running> calls = new ConcurrentHashMap<>();

    private WorkerMain(SocketChannel channel, Path sharedDir, int threshold) {
        this.channel = channel;
        this.sharedDir = sharedDir;
        this.out = new Frames.Writer(channel, sharedDir, threshold, "w");
    }

    /** args: the parent's socket, the shared-memory directory, the shared-memory threshold in bytes. */
    public static void main(String[] args) throws IOException {
        dirs = new Path[]{Path.of(args[0]).getParent(), Path.of(args[1])};
        Thread.ofPlatform().name("kiln-worker-watchdog").daemon(true).start(() -> {
            try { while (System.in.read() >= 0) { /* the parent writes nothing; EOF means it closed us or died */ } }
            catch (IOException ignored) {}
            exit();
        });
        SocketChannel ch = SocketChannel.open(UnixDomainSocketAddress.of(Path.of(args[0])));
        new WorkerMain(ch, Path.of(args[1]), Integer.parseInt(args[2])).serve();
    }

    private void serve() {
        try {
            Frames.Reader reader = new Frames.Reader(channel, sharedDir);
            while (true) {
                Frames.Frame f = reader.read();
                switch (f.type()) {
                    case Frames.PING -> out.send(Frames.PONG, f.id(), 0, null, null);
                    case Frames.OPEN -> open(f);
                    case Frames.CALL -> call(f);
                    case Frames.CANCEL -> cancel(f.id());
                    case Frames.FREE -> free(f);
                    default -> throw new IOException("unknown frame type " + f.type());
                }
            }
        } catch (Throwable e) {
            // the parent is gone (EOF), sent something we can't read, or a pong couldn't be written: either way the
            // parent must see this worker go, not a worker that has stopped reading
        }
        exit();
    }

    private static volatile Path[] dirs = new Path[0];

    /**
     * Halts, after removing the temporary directories (which the parent removes too, unless it died first). Halting
     * skips shutdown hooks and any native call still running; the OS frees the models' memory with the process.
     */
    private static void exit() {
        for (Path d : dirs) {
            if (d == null || !d.getFileName().toString().startsWith("kiln-worker-")) continue;
            try (var s = java.nio.file.Files.walk(d)) {
                for (Path p : s.sorted(java.util.Comparator.reverseOrder()).toList()) java.nio.file.Files.deleteIfExists(p);
            } catch (IOException | RuntimeException ignored) {
                // best effort
            }
        }
        Runtime.getRuntime().halt(0);
    }

    /** Loading a model takes seconds: off the reading thread, so pings are still answered meanwhile. */
    private void open(Frames.Frame f) {
        Thread.ofPlatform().name("kiln-worker-open").stackSize(8L << 20).start(() -> {
            try {
                WorkerService s = create(f.name(), f.payload());
                String name = f.name().substring(f.name().lastIndexOf('.') + 1) + "-" + f.ref();
                Hosted h = new Hosted(s, new ModelThread(name, s::close, s::cancel, null));
                instances.put(f.ref(), h);
                reply(f.id(), null, null);
            } catch (Throwable t) {
                reply(f.id(), null, t);
            }
        });
    }

    private static WorkerService create(String className, byte[] config) throws Throwable {
        try {
            Class<? extends WorkerService> c = Class.forName(className).asSubclass(WorkerService.class);
            return c.getConstructor(byte[].class).newInstance((Object) config);
        } catch (InvocationTargetException e) {
            throw e.getCause();
        }
    }

    private void call(Frames.Frame f) {
        Hosted h = instances.get(f.ref());
        if (h == null) {
            reply(f.id(), null, new IllegalStateException("no instance " + f.ref() + " in this worker"));
            return;
        }
        long id = f.id();
        Running r = new Running();
        calls.put(id, r);
        CompletableFuture<byte[]> future = h.thread().submit(c -> {
            r.call = c;
            if (r.cancelled) c.cancel();
            return h.service().call(f.name(), f.payload(), c, item -> {
                try { out.send(Frames.EVENT, id, 0, null, item); }
                catch (IOException e) { throw new java.io.UncheckedIOException(e); }
            });
        });
        future.whenComplete((v, e) -> {
            calls.remove(id);
            reply(id, v, e);
        });
    }

    /**
     * The call stops the way the engine's own cancel stops it (an LLM returns the text so far), as in-process. The
     * parent counts it as running once sent, so one that hasn't started here yet starts already cancelled.
     */
    private void cancel(long id) {
        Running r = calls.get(id);
        if (r == null) return;
        r.cancelled = true;
        Call c = r.call;
        if (c != null) c.cancel();
    }

    private void free(Frames.Frame f) {
        Hosted h = instances.remove(f.ref());
        Thread.ofPlatform().name("kiln-worker-free").start(() -> {
            try {
                if (h != null) h.thread().close();
                reply(f.id(), null, null);
            } catch (Throwable t) {
                reply(f.id(), null, t);
            }
        });
    }

    private void reply(long id, byte[] result, Throwable failure) {
        if (failure instanceof CompletionException && failure.getCause() != null) failure = failure.getCause();
        try {
            if (failure == null) out.send(Frames.RESULT, id, 0, null, result);
            else out.error(id, failure instanceof CancellationException ? new CancellationException(failure.getMessage()) : failure);
        } catch (IOException e) {
            exit();  // the parent is gone
        } catch (Throwable t) {
            // the reply couldn't be written, e.g. a fault writing a shared-memory slot on a full /dev/shm: say so
            try { out.error(id, t); }
            catch (Throwable u) {
                // nor that: the parent sees the worker die and fails its calls, rather than waiting for this one
                try { channel.close(); } catch (IOException ignored) {}
                exit();
            }
        }
    }
}
