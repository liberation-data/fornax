package org.liberationdata.kiln;

import java.io.EOFException;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.VarHandle;
import java.lang.reflect.Constructor;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.SocketChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * The worker protocol: length-prefixed frames over a Unix domain socket, with payloads above a threshold passed
 * through shared memory (memory-mapped files on {@code /dev/shm} where there is one) instead of through the socket.
 *
 * <p>Shared memory comes in <b>slots</b>: a few files per sender, each mapped once by each side and reused, so a
 * transfer is one copy in and one copy out, with no file to create and no fresh pages to fault in. A slot's first int
 * says whether it is free (0) or holds a payload not yet read (1): the sender claims a free one with a
 * compare-and-set, and the receiver frees it once it has copied the payload out. With every slot busy, or a payload
 * bigger than {@link #SLOT_MAX}, the sender writes a file for that payload alone, which the receiver deletes.
 *
 * <p>A frame: {@code int length, byte type, long id, long ref, string name, payload}, where a payload is
 * {@code int 0, int n, n bytes} inline, {@code int 1, string file, int n} in a slot, or {@code int 2, string file, int n}
 * in a file of its own.
 */
final class Frames {
    // parent to worker
    static final byte OPEN = 1, CALL = 2, CANCEL = 3, FREE = 4, PING = 5;
    // worker to parent
    static final byte RESULT = 11, ERROR = 12, EVENT = 13, PONG = 14;

    static final int MAX_FRAME = 64 << 20;  // inline frames only; a bigger payload goes through shared memory
    static final long SLOT_MAX = 256L << 20;
    static final int MAX_SLOTS = 4;
    private static final int INLINE = 0, SLOT = 1, ONE_OFF = 2;
    private static final long HEADER = 64;  // the state int, padded to a cache line
    private static final VarHandle STATE = JAVA_INT.varHandle();
    private static final byte[] EMPTY = new byte[0];
    /** Payloads this JVM has sent through shared memory: for tests. */
    static final AtomicLong SHARED_PAYLOADS = new AtomicLong();

    /** @param id the call (or open) it belongs to. @param ref the instance, for OPEN, CALL and FREE. */
    record Frame(byte type, long id, long ref, String name, byte[] payload) {}

    /** A shared file, mapped. */
    private record Mapped(String name, FileChannel channel, Arena arena, MemorySegment segment) {
        static Mapped open(Path dir, String name, long size, boolean create) throws IOException {
            FileChannel fc = create
                    ? FileChannel.open(dir.resolve(name), StandardOpenOption.CREATE_NEW, StandardOpenOption.READ, StandardOpenOption.WRITE)
                    : FileChannel.open(dir.resolve(name), StandardOpenOption.READ, StandardOpenOption.WRITE);
            Arena arena = Arena.ofShared();
            try {
                long n = create ? size : fc.size();
                return new Mapped(name, fc, arena, fc.map(FileChannel.MapMode.READ_WRITE, 0, n, arena));
            } catch (IOException | RuntimeException e) {
                arena.close();
                fc.close();
                throw e;
            }
        }

        void close() {
            try { arena.close(); } catch (RuntimeException ignored) { /* still in use as the process goes */ }
            try { channel.close(); } catch (IOException ignored) {}
        }
    }

    /** Sends frames on one channel; safe from any thread. */
    static final class Writer implements AutoCloseable {
        private final SocketChannel channel;
        private final Path sharedDir;
        private final int threshold;
        private final String prefix;
        private final AtomicLong files = new AtomicLong();
        private final java.util.concurrent.locks.ReentrantLock sending = new java.util.concurrent.locks.ReentrantLock();
        private final List<Mapped> slots = new ArrayList<>();  // guarded by itself
        private boolean closed;                                  // guarded by slots

        Writer(SocketChannel channel, Path sharedDir, int threshold, String prefix) {
            this.channel = channel; this.sharedDir = sharedDir; this.threshold = threshold; this.prefix = prefix;
        }

        /** Sends a frame, waiting for any other sender on this channel. */
        void send(byte type, long id, long ref, String name, byte[] payload) throws IOException {
            send(type, id, ref, name, payload, true);
        }

        /**
         * Sends a small frame unless another sender is mid-write: for the heartbeat, which must never block behind a
         * worker that has stopped reading. Returns whether it sent.
         */
        boolean trySend(byte type, long id) throws IOException { return send(type, id, 0, null, null, false); }

        private boolean send(byte type, long id, long ref, String name, byte[] payload, boolean wait) throws IOException {
            if (payload == null) payload = EMPTY;
            Wire.Out head = new Wire.Out();
            head.integer(type).longValue(id).longValue(ref).string(name == null ? "" : name);
            ByteBuffer body;
            if (payload.length > threshold) {
                Mapped slot = claim(payload.length);
                if (slot != null) {
                    MemorySegment.copy(payload, 0, slot.segment(), JAVA_BYTE, HEADER, payload.length);
                    head.integer(SLOT).string(slot.name()).integer(payload.length);
                } else {
                    String file = prefix + "x" + files.incrementAndGet();
                    writeShared(sharedDir.resolve(file), payload);
                    head.integer(ONE_OFF).string(file).integer(payload.length);
                }
                SHARED_PAYLOADS.incrementAndGet();
                body = ByteBuffer.allocate(0);
            } else {
                head.integer(INLINE).integer(payload.length);
                body = ByteBuffer.wrap(payload);
            }
            byte[] h = head.toBytes();
            ByteBuffer[] frame = {ByteBuffer.allocate(4 + h.length).putInt(h.length + body.remaining()).put(h).flip(), body};
            if (wait) sending.lock();
            else if (!sending.tryLock()) return false;
            try {
                while (frame[0].hasRemaining() || frame[1].hasRemaining()) channel.write(frame);
            } finally {
                sending.unlock();
            }
            return true;
        }

        void error(long id, Throwable t) throws IOException { send(ERROR, id, 0, null, encodeError(t)); }

        /** A free slot that fits, marked busy: an existing one, a new one, or a bigger one in place of a small free one. */
        private Mapped claim(int length) throws IOException {
            long need = HEADER + length;
            if (need > SLOT_MAX) return null;
            synchronized (slots) {
                if (closed) throw new IOException("closed");
                Mapped small = null;
                for (Mapped m : slots) {
                    if (m.segment().byteSize() < need && small != null) continue;
                    if ((boolean) STATE.compareAndSet(m.segment(), 0L, 0, 1)) {
                        if (m.segment().byteSize() >= need) {
                            if (small != null) STATE.setVolatile(small.segment(), 0L, 0);
                            return m;
                        }
                        small = m;  // free but too small: replaced below unless a big enough one turns up
                    }
                }
                if (small == null && slots.size() >= MAX_SLOTS) return null;
                if (small != null) {
                    slots.remove(small);
                    small.close();
                    Files.deleteIfExists(sharedDir.resolve(small.name()));
                }
                long size = Math.min(SLOT_MAX, Math.max(1L << 20, Long.highestOneBit(need - 1) << 1));
                Mapped m = Mapped.open(sharedDir, prefix + files.incrementAndGet(), size, true);
                STATE.setVolatile(m.segment(), 0L, 1);
                slots.add(m);
                return m;
            }
        }

        /** Unmaps and deletes the slots. */
        @Override public void close() {
            synchronized (slots) {
                closed = true;
                for (Mapped m : slots) {
                    m.close();
                    try { Files.deleteIfExists(sharedDir.resolve(m.name())); } catch (IOException ignored) {}
                }
                slots.clear();
            }
        }
    }

    /** Reads frames from one channel, on one thread, keeping the sender's slots mapped. */
    static final class Reader implements AutoCloseable {
        private final SocketChannel channel;
        private final Path sharedDir;
        private final Map<String, Mapped> mapped = new LinkedHashMap<>(16, 0.75f, true);
        private final ByteBuffer length = ByteBuffer.allocate(4);

        Reader(SocketChannel channel, Path sharedDir) { this.channel = channel; this.sharedDir = sharedDir; }

        /** The next frame, blocking; EOFException when the other side has closed. */
        Frame read() throws IOException {
            length.clear();
            readFully(length);
            int n = length.flip().getInt();
            if (n < 0 || n > MAX_FRAME) throw new IOException("bad frame length " + n);
            ByteBuffer body = ByteBuffer.allocate(n);
            readFully(body);
            Wire.In in = new Wire.In(body.array());
            byte type = (byte) in.integer();
            long id = in.longValue(), ref = in.longValue();
            String name = in.string();
            int kind = in.integer();
            byte[] payload;
            if (kind == INLINE) {
                int size = in.integer();
                if (size < 0 || size > n) throw new IOException("bad payload length " + size);
                payload = new byte[size];
                System.arraycopy(body.array(), n - size, payload, 0, size);
            } else {
                String file = in.string();
                int size = in.integer();
                if (file == null || file.isEmpty() || file.contains("/") || file.contains("\\") || file.startsWith("."))
                    throw new IOException("bad shared file " + file);
                payload = kind == SLOT ? fromSlot(file, size) : readShared(sharedDir.resolve(file), size);
            }
            return new Frame(type, id, ref, name, payload);
        }

        private byte[] fromSlot(String file, int size) throws IOException {
            Mapped m = mapped.get(file);
            if (m == null || m.segment().byteSize() < HEADER + size) {
                if (m != null) m.close();
                m = Mapped.open(sharedDir, file, 0, false);
                mapped.put(file, m);
                if (mapped.size() > 2 * MAX_SLOTS) {  // slots the sender has since replaced with bigger ones
                    Iterator<Mapped> eldest = mapped.values().iterator();
                    eldest.next().close();
                    eldest.remove();
                }
            }
            if (m.segment().byteSize() < HEADER + size) throw new IOException(file + " is smaller than " + size + " bytes");
            byte[] out = m.segment().asSlice(HEADER, size).toArray(JAVA_BYTE);
            STATE.setVolatile(m.segment(), 0L, 0);
            return out;
        }

        private void readFully(ByteBuffer buf) throws IOException {
            while (buf.hasRemaining()) if (channel.read(buf) < 0) throw new EOFException();
        }

        @Override public void close() {
            mapped.values().forEach(Mapped::close);
            mapped.clear();
        }
    }

    /** A payload in a file of its own. */
    static void writeShared(Path file, byte[] payload) throws IOException {
        try (FileChannel fc = FileChannel.open(file, StandardOpenOption.CREATE_NEW, StandardOpenOption.READ, StandardOpenOption.WRITE);
             Arena arena = Arena.ofConfined()) {
            MemorySegment m = fc.map(FileChannel.MapMode.READ_WRITE, 0, payload.length, arena);
            MemorySegment.copy(payload, 0, m, JAVA_BYTE, 0, payload.length);
        } catch (IOException | RuntimeException e) {
            Files.deleteIfExists(file);
            throw e;
        }
    }

    /** Reads a payload from a file of its own, and deletes the file. */
    static byte[] readShared(Path file, int size) throws IOException {
        try (FileChannel fc = FileChannel.open(file, StandardOpenOption.READ); Arena arena = Arena.ofConfined()) {
            if (fc.size() < size) throw new IOException(file + " has " + fc.size() + " bytes, expected " + size);
            MemorySegment m = fc.map(FileChannel.MapMode.READ_ONLY, 0, size, arena);
            return m.toArray(JAVA_BYTE);
        } finally {
            Files.deleteIfExists(file);
        }
    }

    // ---- exceptions across the boundary: class, message, stack, and causes ----

    static byte[] encodeError(Throwable t) {
        Wire.Out out = new Wire.Out();
        int depth = 0;
        for (Throwable c = t; c != null && depth < 8; c = c.getCause() == c ? null : c.getCause()) depth++;
        out.integer(depth);
        Throwable c = t;
        for (int i = 0; i < depth; i++, c = c.getCause()) {
            out.string(c.getClass().getName()).string(c.getMessage());
            StackTraceElement[] st = c.getStackTrace();
            out.integer(st.length);
            for (StackTraceElement e : st)
                out.string(e.getClassName()).string(e.getMethodName()).string(e.getFileName()).integer(e.getLineNumber());
        }
        return out.toBytes();
    }

    /**
     * The worker's exception rebuilt in this JVM: the same class when it is an unchecked one with a (String)
     * constructor on this class path (IllegalStateException, CancellationException, ...), else a RuntimeException
     * naming it. It carries the worker's stack trace, and its causes.
     */
    static RuntimeException decodeError(byte[] payload) {
        Wire.In in = new Wire.In(payload);
        int depth = in.integer();
        Throwable[] chain = new Throwable[depth];
        for (int i = 0; i < depth; i++) {
            String cls = in.string(), msg = in.string();
            StackTraceElement[] st = new StackTraceElement[in.integer()];
            for (int j = 0; j < st.length; j++) st[j] = new StackTraceElement(in.string(), in.string(), in.string(), in.integer());
            Throwable t = rebuild(cls, msg, i == 0);
            t.setStackTrace(st);
            chain[i] = t;
        }
        for (int i = 0; i + 1 < depth; i++) {
            try { chain[i].initCause(chain[i + 1]); } catch (IllegalStateException alreadySet) { /* keep its own */ }
        }
        Throwable top = depth == 0 ? new RuntimeException("the worker reported an error with no detail") : chain[0];
        return top instanceof RuntimeException r ? r : new RuntimeException(top.toString(), top);
    }

    private static Throwable rebuild(String cls, String msg, boolean top) {
        try {
            Class<?> c = Class.forName(cls, false, Frames.class.getClassLoader());
            if (top ? RuntimeException.class.isAssignableFrom(c) : Throwable.class.isAssignableFrom(c)) {
                Constructor<?> k = c.getConstructor(String.class);
                return (Throwable) k.newInstance(msg);
            }
        } catch (ReflectiveOperationException | LinkageError | RuntimeException ignored) {
            // not here, or no (String) constructor: fall through
        }
        return new RuntimeException(cls + (msg == null ? "" : ": " + msg));
    }

    private Frames() {}
}
