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
