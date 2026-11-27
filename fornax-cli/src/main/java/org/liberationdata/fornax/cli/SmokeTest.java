package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.cli.Args.UsageException;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.llama.Json;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.sd.DiffusionModel;

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

/** {@code fornax test}: load a model, generate a little, report timings. */
final class SmokeTest {
    static final Set<String> OPTIONS = Set.of("max-tokens", "ctx", "gpu-layers", "out", "prompt", "seconds");

    static final String USAGE = """
            usage: fornax test <model> [options]

            Loads the model, generates a little and reports the load time and speed.
            Language models: a short greedy reply, with time to first token and tokens/s.
            Image aliases: one 256x256 image in 4 steps. Song aliases: a 10-second song (--seconds).
            These load the whole model set onto the GPU (see fornax list for sizes).

              --prompt TEXT        what to ask (or draw / sing about)
              --max-tokens N       language models: tokens to generate, reasoning included (default 512)
              --ctx N              language models: context size (default 4096)
              --gpu-layers N       language models: layers to offload (default -1 = all)
              --seconds N          songs: length in seconds (default 10)
              --out FILE           image/song output (default: fornax-test-<alias>.png/.wav in the temp folder)""";

    static int run(Args a, FornaxHub hub, PrintStream out) throws IOException {
        String spec = a.positional(0);
        if (spec == null) throw new UsageException("which model? (see fornax list)");
        if (!a.rest(1).isEmpty()) throw new UsageException("unexpected " + a.rest(1) + " (use --prompt)");
        return switch (Models.resolve(spec, hub)) {
            case Models.Llm llm -> llm(llm, a, out);
            case Models.Set set -> set(set.set(), set.files(), a, out);
        };
    }

    private static int llm(Models.Llm llm, Args a, PrintStream out) {
        String prompt = a.option("prompt", "List three colours of a kiln's fire, one per line.");
        Options o = new Options(a.intOption("max-tokens", 512), 0f, 1f, 0f, 1);
        out.printf("Model     %s (%s)%n", llm.path(), Fmt.bytes(llm.path().toFile().length()));
        long t0 = System.nanoTime();
        try (var model = new LlamaModel(llm.path().toString(), a.intOption("gpu-layers", -1), a.intOption("ctx", 4096))) {
            double load = secs(t0);
            out.printf("Load      %s%n", Fmt.seconds(load));
            long[] first = {0};
            long g0 = System.nanoTime();
            var shown = new ThinkFilter(p -> {}, false);
            var r = model.chat(List.of(Message.user(prompt)), o, p -> { if (first[0] == 0) first[0] = System.nanoTime(); shown.accept(p); });
            shown.finish();
            long g1 = System.nanoTime();
            double total = (g1 - g0) / 1e9;
            double ttft = first[0] == 0 ? total : (first[0] - g0) / 1e9;
            double decode = first[0] == 0 ? 0 : (g1 - first[0]) / 1e9;
            out.printf(Locale.ROOT, "Prompt    %d tokens, first token after %s%n", r.promptTokens(), Fmt.seconds(ttft));
            out.printf(Locale.ROOT, "Generate  %d tokens in %s (%.1f tok/s)%n", r.generatedTokens(), Fmt.seconds(total),
                    decode > 0 && r.generatedTokens() > 1 ? (r.generatedTokens() - 1) / decode : r.generatedTokens() / total);
            String text = shown.reply().replaceAll("\\s+", " ");
            if (text.isEmpty() && !r.text().isBlank()) text = "(only reasoning; raise --max-tokens to see a reply)";
            out.printf("Reply     %s%n", text.length() > 160 ? text.substring(0, 160) + "..." : text);
            if (r.generatedTokens() == 0) { out.println("FAILED: no tokens generated"); return 1; }
            out.println("OK");
            return 0;
        }
    }

    private static int set(ModelSet m, Map<String, Path> files, Args a, PrintStream out) throws IOException {
        if (m.kind() != ModelSet.Kind.IMAGE && m.kind() != ModelSet.Kind.SONG)
            throw new Main.Failure(m.id() + " is a " + m.kind().name().toLowerCase(Locale.ROOT) + " model; fornax test runs chat, image and song models");
        long total = files.values().stream().mapToLong(p -> p.toFile().length()).sum();
        out.printf("Model     %s: %s (%s of weights)%n", m.id(), m.title(), Fmt.bytes(total));
        String ext = m.kind() == ModelSet.Kind.IMAGE ? ".png" : ".wav";
        Path file = Path.of(a.option("out", Path.of(System.getProperty("java.io.tmpdir"), "fornax-test-" + m.id() + ext).toString()));
        long t0 = System.nanoTime();
        if (m.kind() == ModelSet.Kind.IMAGE) image(m, files, a.option("prompt", "a small brick kiln at dusk, warm light"), file, t0, out);
        else song(m, files, a.option("prompt", "a kiln that sings while it fires the clay"), a.intOption("seconds", 10), file, t0, out);
        out.println("OK -> " + file);
        return 0;
    }

    private static void image(ModelSet m, Map<String, Path> files, String prompt, Path file, long t0, PrintStream out) throws IOException {
        double guidance = Image.cfg(m);
        int size = 256, steps = 4;
        List<BufferedImage> refs = Image.needsRef(m) ? List.of(gray(size)) : List.of();
        try (var model = new DiffusionModel(Image.config(files))) {
            out.printf("Load      %s (%s)%n", Fmt.seconds(secs(t0)), model.modelVersion());
            long g0 = System.nanoTime();
            var img = model.generate(new DiffusionModel.Request(prompt, size, size, steps, guidance, 7, refs, Map.of())).get(0);
            double s = secs(g0);
            ImageIO.write(img, "png", file.toFile());
            out.printf(Locale.ROOT, "Generate  %dx%d, %d steps in %s (%.2fs/step)%n", size, size, steps, Fmt.seconds(s), s / steps);
        }
    }

    private static void song(ModelSet m, Map<String, Path> files, String about, int seconds, Path file, long t0, PrintStream out) throws IOException {
        String lyrics = "[Verse]\nThe kiln is warm, the clay is new\nIt sings all night the whole way through\n";
        String req = m.engine().equals("acestep.cpp")
                ? "{\"caption\": " + Json.quote("acoustic folk, warm male vocal, " + about) + ", \"lyrics\": "
                        + Json.quote(lyrics) + ", \"bpm\": 100, \"duration\": " + seconds + ", \"keyscale\": \"G major\", "
                        + "\"timesignature\": \"4\", \"vocal_language\": \"en\", \"seed\": 7, \"lm_seed\": 7, "
                        + "\"inference_steps\": 8, \"guidance_scale\": 1.0, \"shift\": 3.0}"
                : "{\"style\": " + Json.quote("acoustic folk, warm male vocal, " + about) + ", \"lyrics\": "
                        + Json.quote(lyrics) + ", \"duration\": " + seconds + ", \"seed\": 7, \"lm_seed\": 7}";
        try (var model = Song.load(m, files)) {
            out.printf("Load      %s%n", Fmt.seconds(secs(t0)));
            long g0 = System.nanoTime();
            var audio = model.sing(req);
            audio.writeWav(file.toFile());
            out.printf("Generate  %s of audio in %s%n", Fmt.seconds(audio.seconds()), Fmt.seconds(secs(g0)));
        }
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
