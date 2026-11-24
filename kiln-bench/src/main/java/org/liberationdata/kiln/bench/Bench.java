package org.liberationdata.kiln.bench;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.liberationdata.kiln.HfCache;
import org.liberationdata.kiln.acestep.AceStepModel;
import org.liberationdata.kiln.sd.DiffusionModel;
import org.liberationdata.kiln.yue2.Yue2Model;

import javax.imageio.ImageIO;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A/B: the same job through ComfyUI's HTTP API and through Kiln in-process, cold then warm.
 * Usage: Bench [out-dir] [case...]   cases: acestep yue2 qwen-image (default: acestep yue2)
 * Needs ComfyUI on -Dcomfy.url (default http://127.0.0.1:8188) with the matching models installed.
 * Only one side holds the GPU at a time: ComfyUI is told to unload before Kiln runs, and Kiln closes its model
 * before ComfyUI runs.
 */
public class Bench {
    static final ObjectMapper JSON = new ObjectMapper();

    record Row(String job, String side, String weights, double cold, double warm) {}

    interface Side { void run(long seed, Path out) throws Exception; }

    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "out/bench");
        List<String> cases = args.length > 1 ? List.of(args).subList(1, args.length) : List.of("acestep", "yue2");
        var comfy = new ComfyClient(System.getProperty("comfy.url", "http://127.0.0.1:8188"));
        if (!comfy.available()) throw new IllegalStateException("ComfyUI is not reachable; start it first");
        List<Row> rows = new ArrayList<>();
        for (String c : cases) {
            Path d = Files.createDirectories(dir.resolve(c));
            switch (c) {
                case "acestep" -> acestep(comfy, d, rows);
                case "yue2" -> yue2(comfy, d, rows);
                case "qwen-image" -> qwenImage(comfy, d, rows);
                default -> throw new IllegalArgumentException("unknown case " + c);
            }
        }
        String table = table(rows);
        System.out.println(table);
        Files.writeString(dir.resolve("results.md"), "# Kiln vs ComfyUI, "
                + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")) + "\n\n" + table
                + "\nCold = first run with nothing loaded (includes loading weights). Warm = second run, weights resident, "
                + "seed + 1 (ComfyUI would otherwise return its cached result).\n");
    }

    // ---- cases ----

    static void acestep(ComfyClient comfy, Path d, List<Row> rows) throws Exception {
        JsonNode req = JSON.readTree(resource("/requests/acestep-rusty.json"));
        String job = "ACE-Step 1.5 turbo + 4B LM, " + req.path("duration").asInt() + " s song";
        rows.add(comfySide(comfy, job, "safetensors (bf16 DiT)", d, "wav", (p, seed) -> {
            node(p, "TextEncodeAceStepAudio1.5").put("tags", req.path("caption").asText()).put("lyrics", req.path("lyrics").asText())
                    .put("bpm", req.path("bpm").asInt()).put("keyscale", req.path("keyscale").asText()).put("seed", seed);
            node(p, "KSampler").put("seed", seed);
            node(p, "PrimitiveFloat").put("value", req.path("duration").asDouble());
        }, "acestep-turbo"));
        String repo = "Serveurperso/ACE-Step-1.5-GGUF";
        rows.add(kilnSide(job, "GGUF Q8_0", d, "wav", () -> {
            var m = new AceStepModel(hf(repo, "acestep-5Hz-lm-4B-Q8_0.gguf"), hf(repo, "Qwen3-Embedding-0.6B-Q8_0.gguf"),
                    hf(repo, "acestep-v15-turbo-Q8_0.gguf"), hf(repo, "vae-BF16.gguf"), true);
            return new Kiln(m, (seed, out) -> m.generate(withSeed(req, seed)).audio().writeWav(out.toFile()));
        }));
    }

    static void yue2(ComfyClient comfy, Path d, List<Row> rows) throws Exception {
        JsonNode req = JSON.readTree(resource("/requests/yue2-rusty.json"));
        String job = "YuE2 3B with ABC plan, " + req.path("duration").asInt() + " s song";
        rows.add(comfySide(comfy, job, "safetensors (int8)", d, "wav", (p, seed) -> {
            ((ObjectNode) p.path("studio.36").path("inputs")).put("value", req.path("style").asText());
            ((ObjectNode) p.path("studio.37").path("inputs")).put("value", req.path("lyrics").asText());
            ((ObjectNode) p.path("studio.31").path("inputs")).put("value", true);  // plan melody (ABC) first
            ((ObjectNode) p.path("studio.34").path("inputs")).put("seed", seed);
            node(p, "YuE2GenerateMusic").put("max_duration", req.path("duration").asDouble());
        }, "yue2"));
        String repo = "Serveurperso/YuE2-GGUF";
        rows.add(kilnSide(job, "GGUF Q8_0", d, "wav", () -> {
            var m = new Yue2Model(hf(repo, "YuE2-3B-Q8_0.gguf"), hf(repo, "YuE2-Vae-F32.gguf"), true, 0);
            return new Kiln(m, (seed, out) -> m.generate(withSeed(req, seed)).audio().writeWav(out.toFile()));
        }));
    }

    static void qwenImage(ComfyClient comfy, Path d, List<Row> rows) throws Exception {
        String prompt = "A neon sign in a dark honky-tonk bar window that says MACHINE HEART, rain on the glass";
        int size = Integer.getInteger("size", 768), steps = Integer.getInteger("steps", 12);
        double cfg = 2.5;
        String job = "Qwen-Image 2512, " + size + "² " + steps + " steps";
        rows.add(comfySide(comfy, job, "safetensors (fp8)", d, "png", (p, seed) -> {
            ((ObjectNode) p.path("studio.249").path("inputs")).put("text", prompt);
            ((ObjectNode) p.path("studio.252").path("inputs")).put("width", size).put("height", size);
            ((ObjectNode) p.path("studio.253").path("inputs")).put("seed", seed);
            ((ObjectNode) p.path("studio.254").path("inputs")).put("value", steps);
            ((ObjectNode) p.path("studio.255").path("inputs")).put("value", cfg);
            ((ObjectNode) p.path("studio.256").path("inputs")).put("value", false);  // no Lightning LoRA
        }, "qwen-image"));
        String f = "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors";
        rows.add(kilnSide(job, "safetensors (fp8)", d, "png", () -> {
            var m = new DiffusionModel(DiffusionModel.Config.builder()
                    .diffusionModel(hf("Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"))
                    .vae(hf("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"))
                    .llm(HfCache.requireAny("Comfy-Org/Qwen-Image_ComfyUI", f, "Comfy-Org/HunyuanVideo_1.5_repackaged", f).toString())
                    .build());
            return new Kiln(m, (seed, out) -> ImageIO.write(
                    m.generate(new DiffusionModel.Request(prompt, size, size, steps, cfg, seed, List.of(),
                            Map.of("sample.flow_shift", 3.1))).get(0),
                    "png", out.toFile()));
        }));
    }

    // ---- harness ----

    interface Patch { void apply(ObjectNode prompt, long seed); }

    static Row comfySide(ComfyClient comfy, String job, String weights, Path d, String ext, Patch patch, String template) throws Exception {
        ObjectNode base = (ObjectNode) JSON.readTree(resource("/comfy/" + template + ".json"));
        comfy.free();
        double[] t = new double[2];
        for (int i = 0; i < 2; i++) {
            ObjectNode p = base.deepCopy();
            patch.apply(p, 11 + i);
            long t0 = System.nanoTime();
            JsonNode h = comfy.run(p);
            t[i] = (System.nanoTime() - t0) / 1e9;
            if (i == 0) comfy.download(h, d.resolve("comfyui." + ext));
            System.out.printf("[bench] %s | ComfyUI %s: %.1fs%n", job, i == 0 ? "cold" : "warm", t[i]);
        }
        comfy.free();
        return new Row(job, "ComfyUI", weights, t[0], t[1]);
    }

    record Kiln(AutoCloseable model, Side side) {}

    interface KilnFactory { Kiln open() throws Exception; }

    static Row kilnSide(String job, String weights, Path d, String ext, KilnFactory factory) throws Exception {
        double[] t = new double[2];
        long t0 = System.nanoTime();
        Kiln k = factory.open();
        try {
            for (int i = 0; i < 2; i++) {
                k.side().run(11 + i, i == 0 ? d.resolve("kiln." + ext) : d.resolve("kiln-warm." + ext));
                t[i] = (System.nanoTime() - t0) / 1e9;
                System.out.printf("[bench] %s | Kiln %s: %.1fs%n", job, i == 0 ? "cold" : "warm", t[i]);
                t0 = System.nanoTime();
            }
        } finally {
            k.model().close();
        }
        Files.deleteIfExists(d.resolve("kiln-warm." + ext));
        return new Row(job, "Kiln", weights, t[0], t[1]);
    }

    static String table(List<Row> rows) {
        StringBuilder b = new StringBuilder("| Job | Runner | Weights | Cold | Warm |\n|---|---|---|---|---|\n");
        for (Row r : rows) b.append(String.format("| %s | %s | %s | %.1f s | %.1f s |%n", r.job(), r.side(), r.weights(), r.cold(), r.warm()));
        return b.toString();
    }

    static ObjectNode node(ObjectNode prompt, String classType) {
        for (JsonNode n : prompt) if (classType.equals(n.path("class_type").asText())) return (ObjectNode) n.path("inputs");
        throw new IllegalStateException("no " + classType + " node");
    }

    static String withSeed(JsonNode req, long seed) throws Exception {
        ObjectNode r = req.deepCopy();
        r.put("seed", seed).put("lm_seed", seed);
        return JSON.writeValueAsString(r);
    }

    static String hf(String repo, String file) { return HfCache.require(repo, file).toString(); }

    static String resource(String name) throws Exception {
        try (var in = Bench.class.getResourceAsStream(name)) {
            if (in == null) throw new IllegalStateException("missing resource " + name);
            return new String(in.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        }
    }
}
