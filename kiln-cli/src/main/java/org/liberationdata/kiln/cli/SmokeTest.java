package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.acestep.AceStepModel;
import org.liberationdata.kiln.cli.Args.UsageException;
import org.liberationdata.kiln.cli.ModelSets.ModelSet;
import org.liberationdata.kiln.llama.LlamaModel;
import org.liberationdata.kiln.llama.LlamaModel.Message;
import org.liberationdata.kiln.llama.LlamaModel.Options;
import org.liberationdata.kiln.sd.DiffusionModel;
import org.liberationdata.kiln.yue2.Yue2Model;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** {@code kiln test}: load a model, generate a little, report timings. */
final class SmokeTest {
    static final Set<String> OPTIONS = Set.of("max-tokens", "ctx", "gpu-layers", "out", "prompt");

    static final String USAGE = """
            usage: kiln test <model> [options]

            Loads the model, generates a little and reports the load time and speed.
            Language models: a short greedy reply, with time to first token and tokens/s.
            Image aliases: one 256x256 image in 4 steps. Song aliases: a 10-second song.
            These load the whole model set onto the GPU (see kiln list for sizes).

              --prompt TEXT        what to ask (or draw / sing about)
              --max-tokens N       language models: tokens to generate (default 64)
              --ctx N              language models: context size (default 4096)
              --gpu-layers N       language models: layers to offload (default -1 = all)
              --out FILE           image/song output (default: kiln-test-<alias>.png/.wav in the temp folder)""";

    static int run(Args a, HubCache cache, PrintStream out) throws IOException {
        String spec = a.positional(0);
        if (spec == null) throw new UsageException("which model? (see kiln list)");
        if (!a.rest(1).isEmpty()) throw new UsageException("unexpected " + a.rest(1) + " (use --prompt)");
        return switch (Models.resolve(spec, cache)) {
            case Models.Llm llm -> llm(llm, a, out);
            case Models.Set set -> set(set.set(), a, cache, out);
        };
    }

    private static int llm(Models.Llm llm, Args a, PrintStream out) {
        String prompt = a.option("prompt", "List three colours of a kiln's fire, one per line.");
        Options o = new Options(a.intOption("max-tokens", 64), 0f, 1f, 0f, 1);
        out.printf("Model     %s (%s)%n", llm.path(), Fmt.bytes(llm.path().toFile().length()));
        long t0 = System.nanoTime();
        try (var model = new LlamaModel(llm.path().toString(), a.intOption("gpu-layers", -1), a.intOption("ctx", 4096))) {
            double load = secs(t0);
            out.printf("Load      %s%n", Fmt.seconds(load));
            long[] first = {0};
            long g0 = System.nanoTime();
            var r = model.chat(List.of(Message.user(prompt)), o, p -> { if (first[0] == 0) first[0] = System.nanoTime(); });
            long g1 = System.nanoTime();
            double total = (g1 - g0) / 1e9;
            double ttft = first[0] == 0 ? total : (first[0] - g0) / 1e9;
            double decode = first[0] == 0 ? 0 : (g1 - first[0]) / 1e9;
            out.printf(Locale.ROOT, "Prompt    %d tokens, first token after %s%n", r.promptTokens(), Fmt.seconds(ttft));
            out.printf(Locale.ROOT, "Generate  %d tokens in %s (%.1f tok/s)%n", r.generatedTokens(), Fmt.seconds(total),
                    decode > 0 && r.generatedTokens() > 1 ? (r.generatedTokens() - 1) / decode : r.generatedTokens() / total);
            String text = r.text().strip().replaceAll("\\s+", " ");
            out.printf("Reply     %s%n", text.length() > 160 ? text.substring(0, 160) + "..." : text);
            if (r.generatedTokens() == 0) { out.println("FAILED: no tokens generated"); return 1; }
            out.println("OK");
            return 0;
        }
    }

    private static int set(ModelSet m, Args a, HubCache cache, PrintStream out) throws IOException {
        long total = 0;
        for (var c : m.components()) {
            Path p = c.locate(cache).orElseThrow(() -> new Main.Failure(m.name() + " is missing " + c.primary().file()
                    + ". Fetch it with: kiln pull " + m.name()));
            total += p.toFile().length();
        }
        out.printf("Model     %s: %s (%s of weights)%n", m.name(), m.description(), Fmt.bytes(total));
        String ext = m.kind() == ModelSets.Kind.IMAGE ? ".png" : ".wav";
        Path file = Path.of(a.option("out", Path.of(System.getProperty("java.io.tmpdir"), "kiln-test-" + m.name() + ext).toString()));
        long t0 = System.nanoTime();
        switch (m.kind()) {
            case IMAGE -> image(m, cache, a.option("prompt", "a small brick kiln at dusk, warm light"), file, t0, out);
            case SONG -> song(m, cache, a.option("prompt", "a kiln that sings while it fires the clay"), file, t0, out);
            case LLM -> throw new IllegalStateException();
        }
        out.println("OK -> " + file);
        return 0;
    }

    private static void image(ModelSet m, HubCache cache, String prompt, Path file, long t0, PrintStream out) throws IOException {
        var cfg = DiffusionModel.Config.builder().diffusionModel(path(m, "diffusion", cache))
                .vae(path(m, "vae", cache)).llm(path(m, "llm", cache)).build();
        double guidance = m.name().startsWith("qwen") ? 2.5 : 4.0;
        int size = 256, steps = 4;
        List<BufferedImage> refs = m.name().equals("qwen-image-edit") ? List.of(gray(size)) : List.of();
        try (var model = new DiffusionModel(cfg)) {
            out.printf("Load      %s (%s)%n", Fmt.seconds(secs(t0)), model.modelVersion());
            long g0 = System.nanoTime();
            var img = model.generate(new DiffusionModel.Request(prompt, size, size, steps, guidance, 7, refs, Map.of())).get(0);
            double s = secs(g0);
            ImageIO.write(img, "png", file.toFile());
            out.printf(Locale.ROOT, "Generate  %dx%d, %d steps in %s (%.2fs/step)%n", size, size, steps, Fmt.seconds(s), s / steps);
        }
    }

    private static void song(ModelSet m, HubCache cache, String about, Path file, long t0, PrintStream out) throws IOException {
        String lyrics = "[Verse]\nThe kiln is warm, the clay is new\nIt sings all night the whole way through\n";
        if (m.name().startsWith("ace-step")) {
            try (var model = new AceStepModel(path(m, "lm", cache), path(m, "text-encoder", cache), path(m, "dit", cache),
                    path(m, "vae", cache), true)) {
                out.printf("Load      %s%n", Fmt.seconds(secs(t0)));
                String req = "{\"caption\": " + Json.quote("acoustic folk, warm male vocal, " + about) + ", \"lyrics\": "
                        + Json.quote(lyrics) + ", \"bpm\": 100, \"duration\": 10, \"keyscale\": \"G major\", "
                        + "\"timesignature\": \"4\", \"vocal_language\": \"en\", \"seed\": 7, \"lm_seed\": 7, "
                        + "\"inference_steps\": 8, \"guidance_scale\": 1.0, \"shift\": 3.0}";
                long g0 = System.nanoTime();
                var r = model.generate(req);
                r.audio().writeWav(file.toFile());
                out.printf("Generate  %s of audio in %s%n", Fmt.seconds(r.audio().seconds()), Fmt.seconds(secs(g0)));
            }
        } else {
            try (var model = new Yue2Model(path(m, "model", cache), path(m, "vae", cache), true, 0)) {
                out.printf("Load      %s%n", Fmt.seconds(secs(t0)));
                String req = "{\"style\": " + Json.quote("acoustic folk, warm male vocal, " + about) + ", \"lyrics\": "
                        + Json.quote(lyrics) + ", \"duration\": 10, \"seed\": 7, \"lm_seed\": 7}";
                long g0 = System.nanoTime();
                var r = model.generate(req);
                r.audio().writeWav(file.toFile());
                out.printf("Generate  %s of audio in %s%n", Fmt.seconds(r.audio().seconds()), Fmt.seconds(secs(g0)));
            }
        }
    }

    private static String path(ModelSet m, String role, HubCache cache) {
        return m.component(role).locate(cache).orElseThrow().toString();
    }

    private static BufferedImage gray(int size) {
        var img = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);
        var g = img.createGraphics();
        g.setColor(Color.GRAY);
        g.fillRect(0, 0, size, size);
        g.dispose();
        return img;
    }

    private static double secs(long t) { return (System.nanoTime() - t) / 1e9; }
}
