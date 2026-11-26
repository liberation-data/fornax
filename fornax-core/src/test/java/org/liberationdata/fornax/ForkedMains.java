package org.liberationdata.fornax;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.ArrayList;
import java.util.List;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/** Programs the forked-JVM tests run. They use libc rather than an engine, so they need no models or Fornax natives. */
final class ForkedMains {
    private static final Linker LINKER = Linker.nativeLinker();

    private static MethodHandle libc(String name, FunctionDescriptor fd) {
        return LINKER.downcallHandle(LINKER.defaultLookup().find(name).orElseThrow(), fd);
    }

    /**
     * qsort with a comparator upcall that throws, the way a throwing token consumer did. "raw": the exception escapes
     * the upcall, and the JVM dies. "guarded": the comparator does what Fornax's upcalls do (catch, record in the Call,
     * return normally), and the exception is thrown after qsort returns.
     */
    static final class ThrowingUpcall {
        static final IllegalStateException BOOM = new IllegalStateException("consumer failed");
        static volatile Call call;

        static int raw(MemorySegment a, MemorySegment b) { throw BOOM; }

        static int guarded(MemorySegment a, MemorySegment b) {
            try {
                throw BOOM;
            } catch (Throwable t) {
                call.fail(t);
                return 0;
            }
        }

        public static void main(String[] args) throws Throwable {
            MethodHandle qsort = libc("qsort", FunctionDescriptor.ofVoid(ADDRESS, JAVA_LONG, JAVA_LONG, ADDRESS));
            MethodHandle cmp = MethodHandles.lookup().findStatic(ThrowingUpcall.class, args[0],
                    MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class));
            try (var model = new ModelThread("qsort", () -> {}); Arena arena = Arena.ofConfined()) {
                MemorySegment stub = LINKER.upcallStub(cmp, FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS), arena);
                MemorySegment ints = arena.allocateFrom(JAVA_INT, 3, 1, 2);
                try {
                    model.run(c -> {
                        call = c;
                        try { qsort.invokeExact(ints, 3L, 4L, stub); } catch (Throwable t) { throw new RuntimeException(t); }
                        return null;
                    });
                    System.out.println("no exception");
                } catch (IllegalStateException e) {
                    System.out.println("surfaced=" + (e == BOOM ? 1 : 0));
                }
            }
            System.out.println("alive=1");
        }
    }

    /**
     * Virtual threads making native calls that each take 300 ms (usleep, standing in for a generation), with one
     * carrier thread, while a heartbeat virtual thread wakes every 5 ms. "blocking": the calls run on the virtual
     * threads (ModelThread.run, as the blocking model methods do) and pin the carrier. "async": they go through
     * ModelThread.submit and the virtual threads wait on the futures. Prints the heartbeat's longest gap.
     */
    static final class Pinning {
        public static void main(String[] args) throws Exception {
            MethodHandle usleep = libc("usleep", FunctionDescriptor.of(JAVA_INT, JAVA_INT));
            java.util.function.Function<Call, Integer> generation = c -> {
                try { return (int) usleep.invokeExact(300_000); } catch (Throwable t) { throw new RuntimeException(t); }
            };
            boolean async = args[0].equals("async");
            try (var model = new ModelThread("fake", () -> {})) {
                if (async) model.submit(generation).join();  // start the model thread outside the measurement
                var stop = new java.util.concurrent.atomic.AtomicBoolean();
                var maxGap = new java.util.concurrent.atomic.AtomicLong();
                Thread heartbeat = Thread.ofVirtual().start(() -> {
                    long last = System.nanoTime();
                    while (!stop.get()) {
                        try { Thread.sleep(5); } catch (InterruptedException e) { return; }
                        long now = System.nanoTime();
                        maxGap.accumulateAndGet((now - last) / 1_000_000, Math::max);
                        last = now;
                    }
                });
                Thread.sleep(100);
                long start = System.nanoTime();
                List<Thread> callers = new ArrayList<>();
                for (int i = 0; i < 4; i++)
                    callers.add(Thread.ofVirtual().start(() -> {
                        if (async) model.submit(generation).join();
                        else model.run(generation);
                    }));
                for (Thread t : callers) t.join();
                long total = (System.nanoTime() - start) / 1_000_000;
                stop.set(true);
                heartbeat.join();
                System.out.println("parallelism=" + System.getProperty("jdk.virtualThreadScheduler.parallelism"));
                System.out.println("totalMs=" + total);
                System.out.println("maxGapMs=" + maxGap.get());
            }
        }
    }

    /** Starts a worker, prints its pid, and dies without closing it: the worker must go too. */
    static final class WorkerParent {
        public static void main(String[] args) {
            var w = new Worker("orphan", Worker.Options.DEFAULT);
            w.open(TestService.class.getName(), null);
            System.out.println("worker=" + w.pid());
            System.out.flush();
            Runtime.getRuntime().halt(0);
        }
    }

    private ForkedMains() {}
}
