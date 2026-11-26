package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.acestep.AceStepModel;
import org.liberationdata.fornax.cli.Args.UsageException;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.Json;
import org.liberationdata.fornax.yue2.Yue2Model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ThreadLocalRandom;

/** {@code fornax song}: a song from a description (and lyrics), with ACE-Step (acestep.cpp) or YuE2 (yue2.cpp). */
final class Song {
    static final Set<String> OPTIONS = Set.of("lyrics", "seconds", "seed", "out");

    static final String USAGE = """
            usage: fornax song <model> "description" [options]

            Makes a song from a description of its style (genre, instruments, voice, mood) and writes it as a WAV.
            <model> is a song alias: %s. A model that isn't downloaded yet is pulled first.
            YuE2's weights are CC BY-NC 4.0: no commercial use of them or of what they make.

            Without --lyrics, ace-step and ace-step-xl (with the 4B lyric LM) and yue2 write their own words;
            ace-step-turbo and ace-step-xl-turbo make an instrumental.

              --lyrics FILE        the words to sing, with section tags such as [Verse] and [Chorus]; - reads stdin;
                                   [Instrumental] makes an instrumental
              --seconds N          length (default 30)
              --seed N             default random; the seed is printed, to make the same song again
              --out FILE           where to write it (default song.wav here, numbered rather than overwrite one;
                                   a folder puts song.wav in it)""".formatted(Models.ids(Kind.SONG));

    /** A loaded song model: the one call song and test make of ACE-Step and YuE2 alike. */
    interface Singer extends AutoCloseable {
        Audio sing(String requestJson);
        @Override void close();
    }

    static int run(Args a, FornaxHub hub, BufferedReader in, PrintStream out, PrintStream err) throws IOException {
        String spec = a.positional(0);
        if (spec == null) throw new UsageException("which model? " + Models.kinds(Kind.SONG));
        String description = String.join(" ", a.rest(1)).strip();
        if (description.isEmpty()) throw new UsageException("what kind of song? Describe it: genre, instruments, voice, mood");
        String lyricsFile = a.option("lyrics", null);
        String lyrics = lyricsFile == null ? null : lyricsFile.equals("-") ? Chat.readAll(in) : read(Path.of(lyricsFile));
        int seconds = a.intOption("seconds", 30);
        if (seconds < 1) throw new UsageException("--seconds must be at least 1");
        long seed = a.intOption("seed", ThreadLocalRandom.current().nextInt(Integer.MAX_VALUE));
        Path file = OutFile.choose(a.option("out", null), Path.of(""), "song.wav", ".wav");
        var m = Models.fetch(spec, Kind.SONG, "song", false, hub, err, Main.terminal());
        String request = request(m.set(), description, lyrics, seconds, seed);

        err.println("[fornax] loading " + m.set().id() + ": " + m.set().title());
        long t0 = System.nanoTime();
        try (var model = load(m.set(), m.files())) {
            err.printf("[fornax] loaded in %s%n", Fmt.seconds(secs(t0)));
            long g0 = System.nanoTime();
            Audio audio = model.sing(request);
            audio.writeWav(file.toFile());
            err.printf("[fornax] %s of audio in %s, seed %d%n", Fmt.seconds(audio.seconds()), Fmt.seconds(secs(g0)), seed);
        }
        out.println(file);
        return 0;
    }

    /**
     * The engine's request JSON: acestep.cpp's or yue2.cpp's, as the examples' Song and Yue2Song send it. lyrics null:
     * the set writes its own if it has a lyric LM (an ACE-Step set with an {@code lm}, or YuE2), else an instrumental.
     */
    static String request(ModelSet m, String description, String lyrics, int seconds, long seed) {
        if (m.engine().equals("acestep.cpp")) {
            String words = lyrics != null ? lyrics : m.part("lm").isPresent() ? "" : "[Instrumental]";   // "": the LM writes them
            return "{\"caption\": " + Json.quote(description) + ", \"lyrics\": " + Json.quote(words) + ", \"duration\": " + seconds
                    + ", \"seed\": " + seed + ", \"lm_seed\": " + seed + ", \"inference_steps\": 8, \"guidance_scale\": 1.0, \"shift\": 3.0}";
        }
        return "{\"style\": " + Json.quote(description) + ", \"lyrics\": " + Json.quote(lyrics == null ? "" : lyrics)
                + ", \"duration\": " + seconds + ", \"seed\": " + seed + ", \"lm_seed\": " + seed + "}";
    }

    /** The set's engine, loaded and kept loaded. */
    static Singer load(ModelSet m, Map<String, Path> files) {
        if (m.engine().equals("acestep.cpp")) {
            Path lm = files.get("lm");   // the turbo sets without the lyric LM: caption and lyrics drive the DiT directly
            var model = new AceStepModel(lm == null ? null : lm.toString(), files.get("text_encoder").toString(),
                    files.get("dit").toString(), files.get("vae").toString(), true);
            return new Singer() {
                @Override public Audio sing(String json) { return model.generate(json).audio(); }
                @Override public void close() { model.close(); }
            };
        }
        var model = new Yue2Model(files.get("model").toString(), files.get("vae").toString(), true, 0);
        return new Singer() {
            @Override public Audio sing(String json) { return model.generate(json).audio(); }
            @Override public void close() { model.close(); }
        };
    }

    private static String read(Path file) throws IOException {
        if (!Files.isRegularFile(file)) throw new Main.Failure("no such lyrics file: " + file);
        return Files.readString(file).strip();
    }

    private static double secs(long t) { return (System.nanoTime() - t) / 1e9; }
}
