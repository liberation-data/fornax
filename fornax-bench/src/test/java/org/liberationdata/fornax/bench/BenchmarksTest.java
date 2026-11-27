package org.liberationdata.fornax.bench;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** The harness's parsing, timing maths, formatting and memory sampling: no GPU, no model, no reference tool. */
class BenchmarksTest {
    @TempDir Path dir;

    // A trimmed llama-bench -o json for -p 553 -n 128 -d 0,553: a prompt and a generation test at each depth.
    static final String LLAMA_BENCH = """
            [
              {"n_prompt": 553, "n_gen": 0, "n_depth": 0, "avg_ts": 2000.0},
              {"n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": 80.0},
              {"n_prompt": 553, "n_gen": 0, "n_depth": 553, "avg_ts": 1500.0},
              {"n_prompt": 0, "n_gen": 128, "n_depth": 553, "avg_ts": 75.0}
            ]""";

    @Test
    void llamaBenchTakesPromptOnEmptyContextAndGenerationAtDepth() throws Exception {
        Map<String, Double> m = Benchmarks.llamaBench(LLAMA_BENCH, 553);
        assertEquals(2000.0, m.get("promptTokensPerSecond"));
        assertEquals(75.0, m.get("genTokensPerSecond"));
        assertEquals(553 / 2000.0 + 1 / 75.0, m.get("timeToFirstTokenSeconds"), 1e-12);
    }

    @Test
    void llamaBenchWithoutTheTestAtDepthFails() {
        // Run without -d, there's no generation test at the prompt's depth: not an infinite time to the first token.
        var e = assertThrows(IllegalStateException.class, () -> Benchmarks.llamaBench(LLAMA_BENCH, 600));
        assertTrue(e.getMessage().contains("depth 600"), e.getMessage());
        // Run with only -n, there's no prompt test on an empty context to time the prompt by.
        assertThrows(IllegalStateException.class, () -> Benchmarks.llamaBench("""
                [{"n_prompt": 0, "n_gen": 128, "n_depth": 553, "avg_ts": 75.0}]""", 553));
    }

    @Test
    void replyTimesFirstTokenAndTheRestSeparately() {
        // 1 s to the first token, then 9 more in 0.5 s.
        var r = Benchmarks.Reply.of(1_000_000_000L, 2_000_000_000L, 2_500_000_000L, 10);
        assertEquals(1.0, r.ttftSeconds(), 1e-9);
        assertEquals(18.0, r.tokensPerSecond(), 1e-9);
    }

    @Test
    void replyOfNoneOrOneTokenFails() {
        assertThrows(IllegalStateException.class, () -> Benchmarks.Reply.of(1, 0, 5, 0));
        assertThrows(IllegalStateException.class, () -> Benchmarks.Reply.of(1, 3, 5, 1));
    }

    @Test
    void percentilesAreByNearestRank() {
        assertEquals(2.0, Benchmarks.pct(List.of(3.0, 1.0, 2.0), 50));    // the median of three
        assertEquals(3.0, Benchmarks.pct(List.of(3.0, 1.0, 2.0), 95));    // of three, the 95th is the largest
        List<Double> twenty = new java.util.ArrayList<>();
        for (int i = 20; i >= 1; i--) twenty.add((double) i);
        assertEquals(10.0, Benchmarks.pct(twenty, 50));
        assertEquals(19.0, Benchmarks.pct(twenty, 95));
        assertEquals(7.0, Benchmarks.pct(List.of(7.0), 95));
    }

    /** A server answering every chat with {@code status} and these SSE lines; its base URL, until closed. */
    static com.sun.net.httpserver.HttpServer sse(int status, String... lines) throws Exception {
        var s = com.sun.net.httpserver.HttpServer.create(new java.net.InetSocketAddress(java.net.InetAddress.getLoopbackAddress(), 0), 0);
        s.createContext("/", ex -> {
            byte[] b = String.join("\n\n", lines).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            ex.getRequestBody().readAllBytes();
            ex.sendResponseHeaders(status, b.length == 0 ? -1 : b.length);
            try (var o = ex.getResponseBody()) { o.write(b); }
        });
        s.start();
        return s;
    }

    static String base(com.sun.net.httpserver.HttpServer s) { return "http://127.0.0.1:" + s.getAddress().getPort(); }

    @Test
    void aServingRequestTakesItsTokensFromTheUsageChunkAndFailsWithoutContentOrUsage() throws Exception {
        var http = java.net.http.HttpClient.newHttpClient();
        String role = "data: {\"choices\": [{\"delta\": {\"role\": \"assistant\", \"content\": \"\"}}]}";
        String content = "data: {\"choices\": [{\"delta\": {\"content\": \"Once\"}}]}";
        String usage = "data: {\"choices\": [], \"usage\": {\"prompt_tokens\": 260, \"completion_tokens\": 7}}";
        var ok = sse(200, role, content, usage, "data: [DONE]");
        var noUsage = sse(200, role, content, "data: [DONE]");
        var noContent = sse(200, role, usage, "data: [DONE]");   // an empty first delta is not the first token
        var busy = sse(503);
        try {
            var t = Benchmarks.request(http, base(ok), 1);
            assertEquals(7, t.tokens());
            assertTrue(t.ttft() > 0 && t.ttft() <= t.latency(), t.toString());
            for (var s : List.of(noUsage, noContent))
                assertTrue(assertThrows(IllegalStateException.class, () -> Benchmarks.request(http, base(s), 2)).getMessage()
                        .contains("streamed no content or no usage"));
            assertTrue(assertThrows(IllegalStateException.class, () -> Benchmarks.request(http, base(busy), 3)).getMessage()
                    .contains("answered 503"));
        } finally { for (var s : List.of(ok, noUsage, noContent, busy)) s.stop(0); }
    }

    @Test
    void waitingForLlamaServerFailsOnceItHasExitedAndReturnsWhenHealthy() throws Exception {
        Process gone = new ProcessBuilder("true").start();
        gone.waitFor();
        var e = assertThrows(IllegalStateException.class, () -> Benchmarks.awaitHealthy("http://127.0.0.1:1", gone, dir.resolve("log")));
        assertTrue(e.getMessage().contains("exited"), e.getMessage());
        var healthy = sse(200, "ok");
        Process alive = new ProcessBuilder("sleep", "30").start();
        try { Benchmarks.awaitHealthy(base(healthy), alive, dir.resolve("log")); }
        finally { healthy.stop(0); alive.destroy(); }
    }

    @Test
    void aServingSideSetsItsIsolationAndPutsTheOldOneBack() throws Exception {
        String before = System.getProperty("fornax.isolation");
        try {
            System.clearProperty("fornax.isolation");
            assertEquals("process", Benchmarks.withIsolation(true, () -> System.getProperty("fornax.isolation")));
            assertNull(System.getProperty("fornax.isolation"), "an isolated side doesn't leak into the next");
            // started isolated, an in-process side is still in-process, and the setting comes back after it, even on a throw
            System.setProperty("fornax.isolation", "process");
            assertEquals("inprocess", Benchmarks.withIsolation(false, () -> System.getProperty("fornax.isolation")));
            assertThrows(IllegalStateException.class, () -> Benchmarks.withIsolation(false, () -> { throw new IllegalStateException(); }));
            assertEquals("process", System.getProperty("fornax.isolation"));
        } finally {
            if (before == null) System.clearProperty("fornax.isolation");
            else System.setProperty("fornax.isolation", before);
        }
    }

    @Test
    void casesDefaultToAllAndAnUnknownOneFailsBeforeAnyRuns() {
        assertEquals(List.of("chat", "image", "serving"), Benchmarks.cases(new String[]{"out"}));
        assertEquals(List.of("image"), Benchmarks.cases(new String[]{"out", "image"}));
        // A typo after "chat" fails now, not after the chat case has run and its results are lost.
        var e = assertThrows(IllegalArgumentException.class, () -> Benchmarks.cases(new String[]{"out", "chat", "imgae"}));
        assertTrue(e.getMessage().contains("imgae"), e.getMessage());
    }

    @Test
    void runReturnsStandardOutputAndFailsWithTheTailOfStandardError() throws Exception {
        assertEquals("out\n", Benchmarks.run(List.of("sh", "-c", "echo out; echo noise >&2")));
        var e = assertThrows(IllegalStateException.class, () -> Benchmarks.run(List.of("sh", "-c",
                "for i in $(seq 1 20); do echo line$i >&2; done; exit 3")));
        assertTrue(e.getMessage().startsWith("sh failed:"), e.getMessage());
        assertTrue(e.getMessage().contains("line20") && e.getMessage().contains("line6"), e.getMessage());
        assertFalse(e.getMessage().contains("line5\n"), e.getMessage());
    }

    @Test
    void quotedLeavesPlainArgumentsAndQuotesTheRest() {
        assertEquals(List.of("sd-cli", "-W", "768", "/a/b.gguf", "'a kiln, at dusk'", "'it'\\''s'", "''"),
                Benchmarks.quoted(List.of("sd-cli", "-W", "768", "/a/b.gguf", "a kiln, at dusk", "it's", "")));
    }

    @Test
    void markdownHasOneRowEachAndADashWithoutMemory() {
        Map<String, Double> res = new LinkedHashMap<>();
        res.put("coldSeconds", 132.118);
        res.put("warmSeconds", 130.47);
        String md = Benchmarks.markdown(List.of(
                new Benchmarks.Row("image", "Fornax", "DiffusionModel", "flux2-klein", Map.of(), res, 12_733_706_240L, "x"),
                new Benchmarks.Row("image", "sd-cli", "sd.cpp", "flux2-klein", Map.of(), Map.of("coldSeconds", 1.0), null, "y")));
        List<String> lines = md.lines().toList();
        assertEquals(4, lines.size());
        assertEquals("| image | Fornax | DiffusionModel | coldSeconds 132.1, warmSeconds 130.5 | 12.7 GB |", lines.get(2));
        assertEquals("| image | sd-cli | sd.cpp | coldSeconds 1.0 | — |", lines.get(3));
        String serving = Benchmarks.markdown(List.of(new Benchmarks.Row("serving", "Fornax", "fornax server", "qwen3-4b",
                Map.of("clients", 4), Map.of("tokensPerSecond", 189.04), null, "z"))).lines().toList().get(2);
        assertEquals("| serving | Fornax | fornax server | clients 4, tokensPerSecond 189.0 | — |", serving, "a serving row says its clients");
    }

    @Test
    void pinIsTheEngineCommitShortened() throws Exception {
        Path props = dir.resolve("engines.properties");
        Files.writeString(props, """
                llama.cpp.url=https://github.com/ggml-org/llama.cpp
                llama.cpp.commit=22b99122e1dfe1d8a5175bfa2eaa8389699c0b92
                stable-diffusion.cpp.commit=3b12781
                """);
        assertEquals("22b99122e1", Benchmarks.Machine.pin(props, "llama.cpp"));
        assertEquals("3b12781", Benchmarks.Machine.pin(props, "stable-diffusion.cpp"));
        assertEquals("?", Benchmarks.Machine.pin(props, "whisper.cpp"));
        assertEquals("?", Benchmarks.Machine.pin(dir.resolve("missing.properties"), "llama.cpp"));
    }

    @Test
    void memoryIsInBytesAndMissingWithoutTheFile() throws Exception {
        Path meminfo = Files.writeString(dir.resolve("meminfo"), "MemTotal:       127386092 kB\nMemFree: 1 kB\n");
        assertEquals(127386092L * 1024, Benchmarks.Machine.firstNumber(meminfo, "MemTotal:\\s*(\\d+) kB", 1024));
        assertNull(Benchmarks.Machine.firstNumber(dir.resolve("missing"), "MemTotal:\\s*(\\d+) kB", 1024));
    }

    @Test
    void aCounterThatIsGoneOrGarbledCountsAsNothing() throws Exception {
        Path device = Files.createDirectories(dir.resolve("card0/device"));
        Path vram = Files.writeString(device.resolve("mem_info_vram_used"), "100\n");
        Path gtt = Files.writeString(device.resolve("mem_info_gtt_used"), "1000\n");
        List<Path> counters = Benchmarks.GpuMemory.counters(dir);
        Files.writeString(vram, "busy");
        assertEquals(1000L, Benchmarks.GpuMemory.used(counters));
        Files.delete(gtt);
        assertEquals(0L, Benchmarks.GpuMemory.used(counters));
    }

    @Test
    void noCountersMeansNoPeakButTheWorkStillRuns() throws Exception {
        assertEquals(List.of(), Benchmarks.GpuMemory.counters(dir));                     // a card-less /sys/class/drm
        assertEquals(List.of(), Benchmarks.GpuMemory.counters(dir.resolve("missing")));
        Files.createDirectories(dir.resolve("card0/device"));                              // a card without amdgpu's counters
        assertEquals(List.of(), Benchmarks.GpuMemory.counters(dir));
        var m = Benchmarks.GpuMemory.peakDuring(List.of(), () -> "done");
        assertEquals("done", m.value());
        assertNull(m.peakBytes());
    }

    @Test
    void peakIsAboveTheBaselineAcrossVramAndGtt() throws Exception {
        Path device = Files.createDirectories(dir.resolve("card1/device"));
        Path vram = Files.writeString(device.resolve("mem_info_vram_used"), "100\n");
        Path gtt = Files.writeString(device.resolve("mem_info_gtt_used"), "1000\n");
        List<Path> counters = Benchmarks.GpuMemory.counters(dir);
        assertEquals(2, counters.size());
        var m = Benchmarks.GpuMemory.peakDuring(counters, () -> {
            Files.writeString(gtt, "1300\n");
            Thread.sleep(200);
            Files.writeString(gtt, "1000\n");
            return 1;
        });
        assertEquals(300L, m.peakBytes());
        assertEquals("100\n", Files.readString(vram));
    }

    @Test
    void peakSeesMemoryStillHeldWhenTheWorkEnds() throws Exception {
        // Work that raises use just before it returns: the sampler, asleep for up to 50 ms, hasn't seen it yet.
        Path device = Files.createDirectories(dir.resolve("card0/device"));
        Path vram = Files.writeString(device.resolve("mem_info_vram_used"), "100");
        var m = Benchmarks.GpuMemory.peakDuring(Benchmarks.GpuMemory.counters(dir), () -> {
            Thread.sleep(120);
            Files.writeString(vram, "500");
            return 1;
        });
        assertEquals(400L, m.peakBytes());
    }
}
