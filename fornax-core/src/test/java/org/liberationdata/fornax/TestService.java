package org.liberationdata.fornax;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.function.Consumer;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * A service for the worker tests: echoes, streams, spins until cancelled, throws, and crashes the worker for real,
 * through libc (abort, and a write to an unmapped address), with no engine or model.
 */
public final class TestService implements WorkerService {
    private static final Linker LINKER = Linker.nativeLinker();

    public TestService(byte[] config) {
        String mode = config.length == 0 ? "" : new Wire.In(config).string();
        if (mode.equals("fail")) throw new IllegalStateException("bad config");
        if (mode.equals("abort")) abort();  // a model that crashes its engine as it loads
    }

    @Override public byte[] call(String method, byte[] args, Call call, Consumer<byte[]> events) {
        return switch (method) {
            case "echo" -> args;
            case "pid" -> new Wire.Out().longValue(ProcessHandle.current().pid()).toBytes();
            case "sharedPayloads" -> new Wire.Out().longValue(Frames.SHARED_PAYLOADS.get()).toBytes();
            case "big" -> pattern(new Wire.In(args).integer());
            case "stream" -> {
                int n = new Wire.In(args).integer();
                for (int i = 0; i < n && !call.stopRequested(); i++) events.accept(new Wire.Out().integer(i).toBytes());
                yield new Wire.Out().string("done").toBytes();
            }
            case "spin" -> {
                events.accept(new Wire.Out().string("started").toBytes());
                long start = System.nanoTime();
                while (!call.stopRequested()) {
                    if (System.nanoTime() - start > 60_000_000_000L) throw new IllegalStateException("never cancelled");
                    Thread.onSpinWait();
                }
                yield new Wire.Out().string("stopped").toBytes();
            }
            case "throw" -> throw new IllegalArgumentException("bad argument: " + new Wire.In(args).string());
            case "abort" -> { abort(); yield null; }
            case "segv" -> { segfault(); yield null; }
            case "exit" -> { Runtime.getRuntime().halt(3); yield null; }
            default -> throw new UnsupportedOperationException(method);
        };
    }

    /** n bytes of a pattern a test can check. */
    static byte[] pattern(int n) {
        byte[] b = new byte[n];
        for (int i = 0; i < n; i++) b[i] = (byte) (i * 31 + 7);
        return b;
    }

    /** libc abort(): SIGABRT, as a failed GGML_ASSERT does. */
    static void abort() {
        MethodHandle abort = LINKER.downcallHandle(LINKER.defaultLookup().find("abort").orElseThrow(), FunctionDescriptor.ofVoid());
        try { abort.invokeExact(); } catch (Throwable t) { throw new RuntimeException(t); }
    }

    /** libc memset on address 16: a SIGSEGV in native code, which the JVM can't recover from. */
    static void segfault() {
        MethodHandle memset = LINKER.downcallHandle(LINKER.defaultLookup().find("memset").orElseThrow(),
                FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, JAVA_LONG));
        try { MemorySegment r = (MemorySegment) memset.invokeExact(MemorySegment.ofAddress(16), 0, 64L); }
        catch (Throwable t) { throw new RuntimeException(t); }
    }
}
