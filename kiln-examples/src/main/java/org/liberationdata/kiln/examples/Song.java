package org.liberationdata.kiln.examples;

import org.liberationdata.kiln.HfCache;
import org.liberationdata.kiln.acestep.AceStepModel;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A song with ACE-Step 1.5. Usage: Song <out.wav> [request.json] [-Ddit=turbo|xl-turbo|sft] [-Dlm=4B|1.7B|0.6B|none]
 * Weights: hf download Serveurperso/ACE-Step-1.5-GGUF (see the README for the exact files).
 */
public class Song {
    static final String REPO = "Serveurperso/ACE-Step-1.5-GGUF";

    public static void main(String[] args) throws Exception {
        File out = new File(args.length > 0 ? args[0] : "song.wav");
        String json = args.length > 1 ? Files.readString(Path.of(args[1])) : resource("/requests/acestep-rusty.json");
        String lm = System.getProperty("lm", "4B");
        String lmPath = lm.equals("none") ? null : f("acestep-5Hz-lm-" + lm + "-Q8_0.gguf");
        String dit = f("acestep-v15-" + System.getProperty("dit", "turbo") + "-Q8_0.gguf");
        long t0 = System.nanoTime();
        try (var model = new AceStepModel(lmPath, f("Qwen3-Embedding-0.6B-Q8_0.gguf"), dit, f("vae-BF16.gguf"), true)) {
            System.out.printf("[kiln] loaded in %.1fs%n", (System.nanoTime() - t0) / 1e9);
            long g0 = System.nanoTime();
            var r = model.generate(json);
            r.audio().writeWav(out);
            Files.writeString(Path.of(out.getPath().replaceAll("\\.wav$", "") + ".json"), r.resolvedRequestJson());
            System.out.printf("[kiln] %.1fs of audio in %.1fs -> %s%n", r.audio().seconds(), (System.nanoTime() - g0) / 1e9, out);
        }
    }

    private static String f(String name) { return HfCache.require(REPO, name).toString(); }

    static String resource(String name) throws Exception {
        try (var in = Song.class.getResourceAsStream(name)) { return new String(in.readAllBytes(), StandardCharsets.UTF_8); }
    }
}
