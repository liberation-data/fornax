package org.liberationdata.fornax;

import java.util.Arrays;

/**
 * What isolation costs per call, with {@link TestService} (no engine): a trivial echo in-process vs in a worker, both
 * through a ModelThread as a model's calls go; streamed items; and an 8 MB payload through shared memory vs the
 * socket. Run it with {@code ./gradlew :fornax-core:workerBench}.
 */
public final class WorkerBenchmark {
    public static void main(String[] args) {
        int calls = args.length > 0 ? Integer.parseInt(args[0]) : 20_000;
        byte[] small = new byte[16];
        var local = new TestService(new byte[0]);
        try (var inProcess = new ModelThread("echo", () -> {});
             var worker = new Worker("bench", Worker.Options.DEFAULT.withEchoOutput(false));
             var socketOnly = new Worker("bench-socket", Worker.Options.DEFAULT.withEchoOutput(false).withSharedMemoryThreshold(Integer.MAX_VALUE))) {
            long start = System.nanoTime();
            var remote = worker.open(TestService.class.getName(), null);
            System.out.printf("worker start (JVM up, connected, service open): %.0f ms%n", (System.nanoTime() - start) / 1e6);
            System.out.println("worker resident memory, idle: " + rss(worker.pid()));
            var remoteSocket = socketOnly.open(TestService.class.getName(), null);
            try (var isolated = new ModelThread("echo-isolated", remote::close, remote::cancel, null)) {
                System.out.println("per call, " + calls + " calls after warm-up (median / p99, microseconds):");
                report("in-process, blocking", latencies(calls, () -> inProcess.run(c -> local.call("echo", small, c, null))));
                report("isolated,   blocking", latencies(calls, () -> isolated.run(c -> remote.invoke("echo", small, c, null))));
                report("in-process, async   ", latencies(calls, () -> inProcess.submit(c -> local.call("echo", small, c, null)).join()));
                report("isolated,   async   ", latencies(calls, () -> isolated.submit(c -> remote.invoke("echo", small, c, null)).join()));
                report("isolated,   ping    ", latencies(calls, worker::ping));

                int items = 200_000;
                byte[] n = new Wire.Out().integer(items).toBytes();
                for (int i = 0; i < 3; i++) remote.invoke("stream", n, null, b -> {});
                long t0 = System.nanoTime();
                inProcess.run(c -> local.call("stream", n, c, b -> {}));
                double inNs = (System.nanoTime() - t0) / (double) items;
                t0 = System.nanoTime();
                isolated.run(c -> remote.invoke("stream", n, c, b -> {}));
                double isoNs = (System.nanoTime() - t0) / (double) items;
                System.out.printf("%nstreamed items (%d): in-process %.3f us each, isolated %.3f us each%n", items, inNs / 1000, isoNs / 1000);

                byte[] big = TestService.pattern(8 << 20);
                System.out.println("\n8 MB echo, there and back (median ms):");
                System.out.printf("  isolated, shared memory  %.2f%n", median(payload(40, () -> remote.invoke("echo", big))) / 1e6);
                System.out.printf("  isolated, socket only    %.2f%n", median(payload(40, () -> remoteSocket.invoke("echo", big))) / 1e6);
                System.out.printf("  in-process (no copy)     %.4f%n", median(payload(40, () -> local.call("echo", big, null, null))) / 1e6);
                System.out.println("\nworker resident memory after the run: " + rss(worker.pid()));
                java.nio.file.Path dir = worker.sharedDir();
                java.util.concurrent.atomic.AtomicInteger k = new java.util.concurrent.atomic.AtomicInteger();
                System.out.printf("  a file per payload, write+read (what slots avoid)  %.2f%n", median(payload(40, () -> {
                    try {
                        var f = dir.resolve("probe" + k.incrementAndGet());
                        Frames.writeShared(f, big);
                        Frames.readShared(f, big.length);
                    } catch (java.io.IOException e) { throw new java.io.UncheckedIOException(e); }
                })) / 1e6);
            }
        }
    }

    private static String rss(long pid) {
        try {
            for (String l : java.nio.file.Files.readAllLines(java.nio.file.Path.of("/proc/" + pid + "/status")))
                if (l.startsWith("VmRSS:")) return l.substring(6).trim();
        } catch (java.io.IOException e) { /* not Linux */ }
        return "unknown";
    }

    private static long[] latencies(int n, Runnable r) {
        for (int i = 0; i < Math.max(2000, n / 4); i++) r.run();
        long[] t = new long[n];
        for (int i = 0; i < n; i++) {
            long s = System.nanoTime();
            r.run();
            t[i] = System.nanoTime() - s;
        }
        Arrays.sort(t);
        return t;
    }

    private static long[] payload(int n, Runnable r) {
        for (int i = 0; i < 5; i++) r.run();
        long[] t = new long[n];
        for (int i = 0; i < n; i++) {
            long s = System.nanoTime();
            r.run();
            t[i] = System.nanoTime() - s;
        }
        Arrays.sort(t);
        return t;
    }

    private static double median(long[] sorted) { return sorted[sorted.length / 2]; }

    private static void report(String what, long[] sorted) {
        System.out.printf("  %s  %8.1f  %8.1f%n", what, sorted[sorted.length / 2] / 1000.0, sorted[(int) (sorted.length * 0.99)] / 1000.0);
    }

    private WorkerBenchmark() {}
}
