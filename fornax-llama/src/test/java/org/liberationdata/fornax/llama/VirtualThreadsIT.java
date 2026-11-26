package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Qwen3 4B on virtual threads with a single carrier thread, in a forked JVM (the scheduler reads its settings once,
 * at start): blocking chat calls pin the carrier and stall an unrelated heartbeat; chatAsync calls don't.
 */
class VirtualThreadsIT {
    static final String REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF", FILE = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf";
    static String gguf;

    @BeforeAll
    static void find() {
        var path = HfCache.find(REPO, FILE);
        assumeTrue(path.isPresent(), "model not downloaded: hf download " + REPO + " " + FILE);
        gguf = path.get().toString();
    }

    @Test
    void chatAsyncKeepsTheCarrierFreeAndBlockingChatDoesNot() throws Exception {
        List<String> cmd = List.of(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--enable-native-access=ALL-UNNAMED",
                "-Dfornax.library.path=" + System.getProperty("fornax.library.path"),
                "-Djdk.virtualThreadScheduler.parallelism=1", "-Djdk.virtualThreadScheduler.maxPoolSize=1",
                "-cp", System.getProperty("java.class.path"), Child.class.getName(), gguf);
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        String out = new String(p.getInputStream().readAllBytes());
        assertTrue(p.waitFor(5, TimeUnit.MINUTES));
        System.out.println(out.lines().filter(l -> l.contains("=")).reduce("", (a, b) -> a + b + "\n"));
        assertEquals(0, p.exitValue(), out);
        long blockingGap = value(out, "blocking.maxGapMs"), blockingGen = value(out, "blocking.totalMs");
        long asyncGap = value(out, "async.maxGapMs"), asyncOk = value(out, "async.ok"), asyncCalls = value(out, "async.calls");
        assertTrue(blockingGap > blockingGen / 3, "blocking calls didn't stall the heartbeat (" + blockingGap + " ms of " + blockingGen + ")");
        assertEquals(asyncCalls, asyncOk, "every async call returned the expected reply");
        assertTrue(asyncGap < 150, "the heartbeat stalled for " + asyncGap + " ms during async calls");
    }

    static long value(String out, String key) {
        return out.lines().filter(l -> l.startsWith(key + "=")).mapToLong(l -> Long.parseLong(l.substring(key.length() + 1).trim()))
                .findFirst().orElseThrow(() -> new AssertionError("no " + key + " in:\n" + out));
    }

    /** Runs in the forked JVM. */
    public static final class Child {
        static final List<Message> FORNAX = List.of(Message.user("Reply with exactly the word: fornax"));
        static final Options GREEDY = Options.DEFAULT.withTemperature(0).withSeed(1).withMaxTokens(24);

        public static void main(String[] args) throws Exception {
            try (var model = new LlamaModel(args[0], -1, 2048)) {
                model.chatAsync(FORNAX, GREEDY).join();  // warm up and start the model's thread
                measure("blocking", 2, () -> model.chat(FORNAX, GREEDY).text());
                measure("async", 8, () -> model.chatAsync(FORNAX, GREEDY).join().text());
            }
        }

        interface Chat { String reply() throws Exception; }

        static void measure(String mode, int calls, Chat chat) throws Exception {
            var stop = new AtomicBoolean();
            var maxGap = new AtomicLong();
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
            List<String> replies = Collections.synchronizedList(new ArrayList<>());
            List<Thread> callers = new ArrayList<>();
            for (int i = 0; i < calls; i++)
                callers.add(Thread.ofVirtual().start(() -> {
                    try { replies.add(chat.reply()); } catch (Exception e) { replies.add("error: " + e); }
                }));
            for (Thread t : callers) t.join();
            long total = (System.nanoTime() - start) / 1_000_000;
            stop.set(true);
            heartbeat.join();
            System.out.println(mode + ".calls=" + calls);
            System.out.println(mode + ".ok=" + replies.stream().filter(r -> r.toLowerCase().contains("fornax")).count());
            System.out.println(mode + ".totalMs=" + total);
            System.out.println(mode + ".maxGapMs=" + maxGap.get());
        }
    }
}
