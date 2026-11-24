package org.liberationdata.kiln.examples;

import org.liberationdata.kiln.HfCache;
import org.liberationdata.kiln.yue2.Yue2Model;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A song with YuE2 (weights are CC BY-NC 4.0). Usage: Yue2Song <out.wav> [request.json] [-Dquant=Q8_0]
 * Weights: hf download Serveurperso/YuE2-GGUF YuE2-3B-Q8_0.gguf YuE2-Vae-F32.gguf
 * Also writes the ABC score YuE2 composes before singing, next to the .wav.
 */
public class Yue2Song {
    static final String REPO = "Serveurperso/YuE2-GGUF";

    public static void main(String[] args) throws Exception {
        File out = new File(args.length > 0 ? args[0] : "yue2-song.wav");
        String json = args.length > 1 ? Files.readString(Path.of(args[1])) : Song.resource("/requests/yue2-rusty.json");
        String model = HfCache.require(REPO, "YuE2-3B-" + System.getProperty("quant", "Q8_0") + ".gguf").toString();
        long t0 = System.nanoTime();
        try (var yue = new Yue2Model(model, HfCache.require(REPO, "YuE2-Vae-F32.gguf").toString(), true, 0)) {
            System.out.printf("[kiln] loaded in %.1fs%n", (System.nanoTime() - t0) / 1e9);
            long g0 = System.nanoTime();
            var r = yue.generate(json);
            r.audio().writeWav(out);
            String base = out.getPath().replaceAll("\\.wav$", "");
            Files.writeString(Path.of(base + ".json"), r.resolvedRequestJson());
            if (r.abcScore() != null) Files.writeString(Path.of(base + ".abc"), r.abcScore());
            System.out.printf("[kiln] %.1fs of audio in %.1fs -> %s%n", r.audio().seconds(), (System.nanoTime() - g0) / 1e9, out);
        }
    }
}
