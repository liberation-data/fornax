package org.liberationdata.fornax.bench;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.PullListener;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.sd.DiffusionModel;

import javax.imageio.ImageIO;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Callable;

/**
 * Fornax against each engine's own command-line tool, built from the same pinned commit: the same model files and
 * settings, so a difference is Fornax's (Java, FFM, how it drives the engine) and not the engine's.
 *
 * <pre>
 * ./gradlew :fornax-bench:benchmarks --args='out/benchmarks chat image'
 *     -Pllama.bench=/path/to/llama-bench -Psd.cli=/path/to/sd-cli
 * </pre>
 *
 * Build the two tools from the checkouts {@code native/build.sh} leaves under {@code engines-src/} (the pins, with
 * Fornax's patches) and the same backend; the commands are under "Running them yourself" on the site's Benchmarks page
 * ({@code website/src/content/docs/docs/benchmarks.mdx}).
 *
 * Cases: {@code chat} (qwen3-4b: time to the first token of a ~512-token prompt, and tokens per second over 128
 * generated after it) and
 * {@code image} (FLUX.2 [klein], 768x768, 12 steps: seconds, cold with the load and warm). Each side's peak GPU memory
 * is sampled from the amdgpu driver's sysfs counters (VRAM and GTT, above what was in use before it started), where
 * there are any. Writes {@code results.json} (with the machine, the engines' pins and the command behind every row) and
 * {@code results.md}. One side runs at a time, so the two never share the GPU.
 */
public class Benchmarks {
    static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    /** One measurement: what ran, how, what it took, and the command that reproduces it. */
    record Row(String bench, String side, String engine, String model, Map<String, Object> settings,
               Map<String, Double> results, Long peakGpuBytes, String command) {}

    public static void main(String[] args) throws Exception {
        Path out = Path.of(args.length > 0 ? args[0] : "out/benchmarks");
        List<String> cases = args.length > 1 ? List.of(args).subList(1, args.length) : List.of("chat", "image");
        Files.createDirectories(out);
        List<Row> rows = new ArrayList<>();
        for (String c : cases) {
            switch (c) {
                case "chat" -> chat(out, rows);
                case "image" -> image(out, rows);
                default -> throw new IllegalArgumentException("unknown case " + c + " (chat, image)");
            }
        }
        Map<String, Object> report = new LinkedHashMap<>();
        report.put("date", Instant.now().toString());
        report.put("machine", Machine.describe());
        report.put("rows", rows);
        JSON.writeValue(out.resolve("results.json").toFile(), report);
        String md = markdown(rows);
        Files.writeString(out.resolve("results.md"), md);
        System.out.println(md);
    }

    // ---- chat: LlamaModel against llama-bench ----

    static final String CHAT_MODEL = "qwen3-4b";
    static final int PROMPT_TOKENS = 512, GEN_TOKENS = 128, REPS = 3;

    static void chat(Path out, List<Row> rows) throws Exception {
        Path gguf = FornaxHub.standard().fetch(CHAT_MODEL, PullListener.logging()).get("gguf");
        String bench = tool("llama.bench", "llama-bench");

        // Fornax first: the same file, all layers on the GPU, llama.cpp's default batch sizes and flash attention.
        // Its prompt's exact length sets llama-bench's depth below, so both generate after the same context.
        var fornax = GpuMemory.peakDuring(() -> {
            long t0 = System.nanoTime();
            try (var llm = new LlamaModel(gguf.toString(), -1, 4096, LlamaModel.Mode.CHAT)) {
                double load = (System.nanoTime() - t0) / 1e9;
                String prompt = promptOf(PROMPT_TOKENS);
                double ttft = 0, tg = 0;
                int ppTokens = 0, tgTokens = 0;
                for (int r = 0; r < REPS + 1; r++) {   // the first is a warm-up, as llama-bench's is
                    long start = System.nanoTime();
                    long[] first = {0};
                    var res = llm.complete(prompt, new LlamaModel.Options(GEN_TOKENS, 0f, 1f, 0f, r),
                            piece -> { if (first[0] == 0) first[0] = System.nanoTime(); });
                    long end = System.nanoTime();
                    if (r == 0) continue;
                    var reply = Reply.of(start, first[0], end, res.generatedTokens());
                    ttft += reply.ttftSeconds();
                    tg += reply.tokensPerSecond();
                    ppTokens = res.promptTokens();
                    tgTokens = res.generatedTokens();
                }
                Map<String, Double> m = new LinkedHashMap<>();
                m.put("timeToFirstTokenSeconds", ttft / REPS);
                m.put("genTokensPerSecond", tg / REPS);
                m.put("loadSeconds", load);
                m.put("promptTokens", (double) ppTokens);
                m.put("genTokens", (double) tgTokens);
                return m;
            }
        });
        int depth = fornax.value().get("promptTokens").intValue();
        Map<String, Object> settings = Map.of("promptTokens", depth, "genTokens", GEN_TOKENS, "gpuLayers", "all", "reps", REPS);
        rows.add(new Row("chat", "Fornax", "LlamaModel (llama.cpp " + Machine.pin("llama.cpp") + ")", CHAT_MODEL, settings,
                fornax.value(), fornax.peakBytes(),
                "./gradlew :fornax-bench:benchmarks --args='" + out + " chat' -Pllama.bench=" + quoted(List.of(bench)).get(0)));

        // The reference: llama.cpp's own benchmark at the pin Fornax builds. Its prompt test runs on an empty context
        // and its generation test after `depth` tokens, as Fornax's did. Time to first token, as Fornax measures it,
        // is the prompt plus one generated token.
        List<String> cmd = List.of(bench, "-m", gguf.toString(), "-p", String.valueOf(depth), "-n", String.valueOf(GEN_TOKENS),
                "-d", "0," + depth, "-ngl", "99", "-r", String.valueOf(REPS), "-o", "json");
        var ref = GpuMemory.peakDuring(() -> run(cmd));
        rows.add(new Row("chat", "llama-bench", "llama.cpp " + Machine.pin("llama.cpp"), CHAT_MODEL, settings,
                llamaBench(ref.value(), depth), ref.peakBytes(), String.join(" ", quoted(cmd))));
    }

    /**
     * llama-bench's JSON as Fornax's figures: its prompt test on an empty context, and its generation test after
     * {@code depth} tokens. Fails if either is missing, rather than report an infinite time.
     */
    static Map<String, Double> llamaBench(String json, int depth) throws IOException {
        double pp = 0, tgAtDepth = 0;
        for (JsonNode test : JSON.readTree(json)) {
            int nPrompt = test.path("n_prompt").asInt(), nDepth = test.path("n_depth").asInt();
            if (nPrompt > 0 && nDepth == 0) pp = test.path("avg_ts").asDouble();
            if (nPrompt == 0 && nDepth == depth) tgAtDepth = test.path("avg_ts").asDouble();
        }
        if (pp <= 0 || tgAtDepth <= 0)
            throw new IllegalStateException("llama-bench gave no prompt test at depth 0 or no generation test at depth " + depth + ":\n" + json);
        Map<String, Double> m = new LinkedHashMap<>();
        m.put("timeToFirstTokenSeconds", depth / pp + 1 / tgAtDepth);
        m.put("genTokensPerSecond", tgAtDepth);
        m.put("promptTokensPerSecond", pp);
        return m;
    }

    /** One timed reply: seconds to its first token, and tokens per second after it (the first token's time is in ttft). */
    record Reply(double ttftSeconds, double tokensPerSecond) {
        /** Fails on a reply of fewer than two tokens, which has no generation to time, rather than average in a zero. */
        static Reply of(long startNanos, long firstNanos, long endNanos, int generated) {
            if (generated < 2 || firstNanos == 0)
                throw new IllegalStateException("a reply of " + generated + " tokens has no generation to time");
            return new Reply((firstNanos - startNanos) / 1e9, (generated - 1) / ((endNanos - firstNanos) / 1e9));
        }
    }

    /** About n tokens of plain English: the prompt Fornax times its prompt processing on. */
    static String promptOf(int n) {
        String sentence = "The kiln was fired at dawn, and the potter watched the cones bend as the heat rose. ";
        return sentence.repeat(Math.max(1, n / 19)) + "Tell the rest of the story.";
    }

    // ---- image: DiffusionModel against sd-cli ----

    static final String IMAGE_MODEL = "flux2-klein";
    static final int SIZE = 768, STEPS = 12;
    static final double CFG = 4.0;
    static final long SEED = 7;
    static final String PROMPT = "a brick kiln at dusk, warm light, a potter loading bowls";

    static void image(Path out, List<Row> rows) throws Exception {
        Map<String, Path> files = FornaxHub.standard().fetch(IMAGE_MODEL, PullListener.logging());
        Map<String, Object> settings = Map.of("width", SIZE, "height", SIZE, "steps", STEPS, "cfg", CFG, "seed", SEED);

        List<String> cmd = List.of(tool("sd.cli", "sd-cli"), "--diffusion-model", files.get("diffusion_model").toString(),
                "--vae", files.get("vae").toString(), "--llm", files.get("llm").toString(), "-p", PROMPT,
                "-W", String.valueOf(SIZE), "-H", String.valueOf(SIZE), "--steps", String.valueOf(STEPS),
                "--cfg-scale", String.valueOf(CFG), "-s", String.valueOf(SEED), "-o", out.resolve("sd-cli.png").toString());
        var ref = GpuMemory.peakDuring(() -> {
            long t0 = System.nanoTime();
            run(cmd);
            return (System.nanoTime() - t0) / 1e9;
        });
        rows.add(new Row("image", "sd-cli", "stable-diffusion.cpp " + Machine.pin("stable-diffusion.cpp"), IMAGE_MODEL, settings,
                Map.of("coldSeconds", ref.value()), ref.peakBytes(), String.join(" ", quoted(cmd))));

        var fornax = GpuMemory.peakDuring(() -> {
            long t0 = System.nanoTime();
            try (var model = new DiffusionModel(DiffusionModel.Config.builder().model(IMAGE_MODEL).build())) {
                var req = new DiffusionModel.Request(PROMPT, SIZE, SIZE, STEPS, CFG, SEED);
                ImageIO.write(model.generate(req).get(0), "png", out.resolve("fornax.png").toFile());
                double cold = (System.nanoTime() - t0) / 1e9;
                long w0 = System.nanoTime();
                model.generate(req);
                double warm = (System.nanoTime() - w0) / 1e9;
                Map<String, Double> m = new LinkedHashMap<>();
                m.put("coldSeconds", cold);
                m.put("warmSeconds", warm);
                return m;
            }
        });
        rows.add(new Row("image", "Fornax", "DiffusionModel (stable-diffusion.cpp " + Machine.pin("stable-diffusion.cpp") + ")",
                IMAGE_MODEL, settings, fornax.value(), fornax.peakBytes(),
                "./gradlew :fornax-bench:benchmarks --args='" + out + " image' -Psd.cli=" + quoted(List.of(cmd.get(0))).get(0)));
    }

    // ---- helpers ----

    /** A reference tool's path: -D{property}, else the name on the PATH. */
    static String tool(String property, String name) { return System.getProperty(property, name); }

    /** Runs a command to the end; its standard output, or a failure with the tail of what it printed. */
    static String run(List<String> cmd) throws IOException, InterruptedException {
        Path log = Files.createTempFile("bench-", ".log");
        try {
            Process p = new ProcessBuilder(cmd).redirectError(log.toFile()).start();
            String stdout = new String(p.getInputStream().readAllBytes());
            if (p.waitFor() != 0) {
                List<String> lines = Files.readAllLines(log);
                throw new IllegalStateException(cmd.get(0) + " failed:\n" + String.join("\n", lines.subList(Math.max(0, lines.size() - 15), lines.size())));
            }
            return stdout;
        } finally {
            Files.deleteIfExists(log);
        }
    }

    static List<String> quoted(List<String> cmd) {
        return cmd.stream().map(a -> a.matches("[\\w./:=+-]+") ? a : "'" + a.replace("'", "'\\''") + "'").toList();
    }

    static String markdown(List<Row> rows) {
        StringBuilder b = new StringBuilder("| Bench | Side | Engine | Results | Peak GPU memory |\n|---|---|---|---|---|\n");
        for (Row r : rows) {
            StringBuilder res = new StringBuilder();
            r.results().forEach((k, v) -> res.append(res.isEmpty() ? "" : ", ").append(k).append(' ').append(String.format(Locale.ROOT, "%.1f", v)));
            b.append("| ").append(r.bench()).append(" | ").append(r.side()).append(" | ").append(r.engine()).append(" | ").append(res)
                    .append(" | ").append(r.peakGpuBytes() == null ? "—" : String.format(Locale.ROOT, "%.1f GB", r.peakGpuBytes() / 1e9)).append(" |\n");
        }
        return b.toString();
    }

    /** What a run returned, and the most GPU memory in use above where it started (null where it can't be read). */
    record Measured<T>(T value, Long peakBytes) {}

    /** Peak GPU memory from amdgpu's sysfs counters (VRAM and GTT), sampled every 50 ms. */
    static final class GpuMemory {
        static <T> Measured<T> peakDuring(Callable<T> work) throws Exception {
            return peakDuring(counters(Path.of("/sys/class/drm")), work);
        }

        static <T> Measured<T> peakDuring(List<Path> counters, Callable<T> work) throws Exception {
            if (counters.isEmpty()) return new Measured<>(work.call(), null);
            long base = used(counters);
            long[] peak = {base};
            var running = new java.util.concurrent.atomic.AtomicBoolean(true);
            Thread sampler = Thread.ofPlatform().daemon().start(() -> {
                while (running.get()) {
                    peak[0] = Math.max(peak[0], used(counters));
                    try { Thread.sleep(50); } catch (InterruptedException e) { return; }
                }
            });
            T value;
            try {
                value = work.call();
            } finally {
                running.set(false);
                sampler.join();
            }
            // Read after the join, so the sampler's last write is seen, and sample once more for work shorter than 50 ms.
            return new Measured<>(value, Math.max(0, Math.max(peak[0], used(counters)) - base));
        }

        static List<Path> counters(Path drm) {
            List<Path> out = new ArrayList<>();
            try (var cards = Files.newDirectoryStream(drm, "card[0-9]*")) {
                for (Path c : cards) for (String f : List.of("mem_info_vram_used", "mem_info_gtt_used")) {
                    Path p = c.resolve("device").resolve(f);
                    if (Files.isReadable(p)) out.add(p);
                }
            } catch (IOException | RuntimeException e) { return List.of(); }
            return out;
        }

        static long used(List<Path> counters) {
            long n = 0;
            for (Path p : counters) {
                try { n += Long.parseLong(Files.readString(p).strip()); } catch (IOException | NumberFormatException e) { /* gone */ }
            }
            return n;
        }
    }

    /** The machine the numbers come from: OS, kernel, CPU, GPU, its driver, Java, and Fornax's engine pins. */
    static final class Machine {
        static Map<String, Object> describe() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("os", System.getProperty("os.name") + " " + System.getProperty("os.version"));
            m.put("arch", System.getProperty("os.arch"));
            m.put("java", Runtime.version().toString());
            m.put("cpu", firstMatch(Path.of("/proc/cpuinfo"), "model name\\s*:\\s*(.*)"));
            m.put("memoryBytes", firstNumber(Path.of("/proc/meminfo"), "MemTotal:\\s*(\\d+) kB", 1024));
            m.put("gpu", command(List.of("sh", "-c", "lspci | grep -iE 'vga|display|3d' | head -1 | cut -d: -f3-")));
            m.put("vulkanDriver", command(List.of("sh", "-c", "(rpm -q --qf '%{NAME} %{VERSION}\\n' mesa-vulkan-drivers 2>/dev/null "
                    + "|| dpkg-query -W -f '${Package} ${Version}\\n' mesa-vulkan-drivers 2>/dev/null) | head -1")));
            m.put("backend", System.getProperty("fornax.backend", "vulkan"));
            m.put("fornaxCommit", command(List.of("git", "rev-parse", "--short", "HEAD")));
            m.put("llama.cpp", pin("llama.cpp"));
            m.put("stable-diffusion.cpp", pin("stable-diffusion.cpp"));
            return m;
        }

        /** The engine's commit from native/engines.properties, short. */
        static String pin(String engine) { return pin(Path.of("native/engines.properties"), engine); }

        static String pin(Path properties, String engine) {
            String v = firstMatch(properties, java.util.regex.Pattern.quote(engine) + "\\.commit=(\\w+)");
            return v == null ? "?" : v.substring(0, Math.min(10, v.length()));
        }

        static String firstMatch(Path file, String regex) {
            try {
                var m = java.util.regex.Pattern.compile(regex, java.util.regex.Pattern.MULTILINE).matcher(Files.readString(file));
                return m.find() ? m.group(1).strip() : null;
            } catch (IOException e) { return null; }
        }

        static Long firstNumber(Path file, String regex, long scale) {
            String v = firstMatch(file, regex);
            return v == null ? null : Long.parseLong(v) * scale;
        }

        static String command(List<String> cmd) {
            try {
                Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
                String s = new String(p.getInputStream().readAllBytes()).strip();
                return p.waitFor() == 0 && !s.isEmpty() ? s : null;
            } catch (IOException | InterruptedException e) { return null; }
        }
    }
}
