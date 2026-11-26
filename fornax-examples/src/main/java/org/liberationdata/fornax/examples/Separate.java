package org.liberationdata.fornax.examples;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.demucs.DemucsModel;
import org.liberationdata.fornax.demucs.DemucsModel.Options;
import org.liberationdata.fornax.demucs.DemucsModel.Weights;
import org.liberationdata.fornax.demucs.Stems;

import java.io.File;
import java.nio.file.Path;

/**
 * Splits a song (a WAV file) into vocals and backing with HTDemucs, and with --all into every stem the model makes.
 * <pre>
 * Separate song.wav [--all] [--shifts 2] [--cpu]
 * </pre>
 * writes song-vocals.wav and song-backing.wav beside it (with --all, song-drums.wav, song-bass.wav, ... too).
 * -Dmodel picks the weights: htdemucs (default), htdemucs-f16, htdemucs-6s, htdemucs-ft, or a path to a stems.cpp
 * GGUF. Convert an MP3 first: ffmpeg -i song.mp3 song.wav
 * Default model: hf download thepatch/htdemucs-GGUF htdemucs-42M-v1.0-F32.gguf
 */
public class Separate {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("usage: Separate song.wav [--all] [--shifts n] [--cpu]");
            System.exit(2);
        }
        Path in = Path.of(args[0]);
        Options o = Options.DEFAULT;
        boolean all = false, gpu = true;
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--all" -> all = true;
                case "--shifts" -> o = o.withShifts(Integer.parseInt(args[++i]));
                case "--cpu" -> gpu = false;
                default -> throw new IllegalArgumentException("unknown option: " + args[i]);
            }
        }
        Audio audio;
        try {
            audio = Audio.readWav(in.toFile());
        } catch (java.io.IOException e) {
            throw new IllegalArgumentException(in + ": " + e.getMessage()
                    + " (convert other formats first, e.g. ffmpeg -i song.mp3 song.wav)", e);
        }
        String m = System.getProperty("model", "htdemucs");
        long t0 = System.nanoTime();
        try (var demucs = new DemucsModel(m.endsWith(".gguf") ? m : Weights.named(m).path(), gpu, 0)) {
            System.out.printf("[fornax] loaded %s on %s in %.1fs%n", demucs.name(), demucs.backend(), (System.nanoTime() - t0) / 1e9);
            long s0 = System.nanoTime();
            Stems stems = demucs.separate(audio, o, (done, total) -> System.out.printf("\r[fornax] segment %d/%d", done, total));
            double s = (System.nanoTime() - s0) / 1e9;
            System.out.printf("%n[fornax] %.1fs of audio in %.1fs (%.1fx real time)%n", audio.seconds(), s, audio.seconds() / s);
            String base = in.getFileName().toString().replaceFirst("\\.[^.]+$", "");
            File dir = in.toAbsolutePath().getParent().toFile();
            write(stems.vocals(), new File(dir, base + "-vocals.wav"));
            write(stems.backing(), new File(dir, base + "-backing.wav"));
            if (all) for (String name : stems.names())
                if (!name.equals("vocals")) write(stems.get(name), new File(dir, base + "-" + name + ".wav"));
        }
    }

    private static void write(Audio a, File f) throws java.io.IOException {
        a.writeWav(f);
        System.out.println("[fornax] wrote " + f);
    }
}
