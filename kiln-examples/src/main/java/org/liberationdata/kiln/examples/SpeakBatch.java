package org.liberationdata.kiln.examples;

import org.liberationdata.kiln.tts.TtsModel;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Speaks many lines with one loaded model, all at the same seed, so a designed voice stays as alike as the model allows.
 * Usage: SpeakBatch lines.tsv [voice] [-Dmodel=voxcpm2] [-Dseed=1], where each line of lines.tsv is
 * {@code out.wav<TAB>text}. Used by the explainer's build for a voice that runs in Kiln.
 */
public class SpeakBatch {
    public static void main(String[] args) throws Exception {
        var lines = Files.readAllLines(Path.of(args[0]));
        String voice = args.length > 1 && !args[1].isBlank() ? args[1] : null;
        String m = System.getProperty("model", "voxcpm2");
        int seed = Integer.parseInt(System.getProperty("seed", "1"));
        try (var tts = m.endsWith(".gguf") ? new TtsModel(m, System.getProperty("mmproj"))
                                           : new TtsModel(TtsModel.Weights.named(m))) {
            var options = TtsModel.Options.DEFAULT.withSeed(seed);
            for (String line : lines) {
                if (line.isBlank()) continue;
                int tab = line.indexOf('\t');
                File out = new File(line.substring(0, tab));
                long t0 = System.nanoTime();
                var audio = tts.speak(line.substring(tab + 1), voice, options, null);
                if (out.getParentFile() != null) out.getParentFile().mkdirs();
                audio.writeWav(out);
                System.out.printf("[kiln] %.1fs of audio in %.1fs -> %s%n", audio.seconds(), (System.nanoTime() - t0) / 1e9, out);
            }
        }
    }
}
