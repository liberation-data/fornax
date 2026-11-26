package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.cli.Args.UsageException;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.tts.TtsModel;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/** {@code fornax speak}: text to a WAV with a text-to-speech model (llama.cpp's TTS). */
final class Speak {
    static final Set<String> OPTIONS = Set.of("model", "voice", "out", "mmproj");
    static final Set<String> FLAGS = Set.of("list-voices");
    static final String MODEL = "kokoro";

    static final String USAGE = """
            usage: fornax speak ["text"] [options]

            Speaks the text and writes it as a WAV. Without text, reads it from stdin. The model is pulled
            first if it isn't downloaded yet.

              --model M            a speech alias (default kokoro), or a GGUF by org/name:file.gguf or path, with --mmproj
              --mmproj M           the model's mmproj (its voices and audio decoder), by org/name:file.gguf or path
              --voice V            default the model's own first voice (Kokoro's af_heart); --list-voices for the others
              --list-voices        print the model's voices and stop
              --out FILE           where to write it (default speech.wav here, numbered rather than overwrite one;
                                   a folder puts speech.wav in it)""";

    static int run(Args a, FornaxHub hub, BufferedReader in, PrintStream out, PrintStream err) throws IOException {
        boolean list = a.has("list-voices");
        String text = String.join(" ", a.positional()).strip();
        if (list && !text.isEmpty()) throw new UsageException("--list-voices takes no text");
        if (!list && text.isEmpty()) {
            if (Main.terminal()) throw new UsageException("what should it say? Give the text, or pipe it on stdin");
            text = Chat.readAll(in);
            if (text.isBlank()) throw new UsageException("no text (give it, or pipe it on stdin)");
        }
        Path file = list ? null : OutFile.choose(a.option("out", null), Path.of(""), "speech.wav", ".wav");
        String spec = a.option("model", MODEL), mmprojSpec = a.option("mmproj", null);
        var m = Models.fetch(spec, Kind.SPEECH, "speak", true, hub, err, Main.terminal());
        if (m.set() == null && mmprojSpec == null)
            throw new UsageException(spec + " needs its mmproj (its voices and audio decoder): --mmproj org/name:mmproj-file.gguf or a path");
        Path mmproj = mmprojSpec != null ? Models.fetch(mmprojSpec, Kind.SPEECH, "speak --mmproj", true, hub, err, Main.terminal()).file("gguf")
                : m.file("mmproj");

        err.println("[fornax] loading " + m.name());
        long t0 = System.nanoTime();
        try (var tts = new TtsModel(m.file("gguf").toString(), mmproj.toString())) {
            err.printf("[fornax] loaded in %s%n", Fmt.seconds(secs(t0)));
            if (list) {
                if (tts.voices().isEmpty()) err.println("[fornax] " + m.name() + " has no predefined voices");
                tts.voices().forEach(out::println);
                return 0;
            }
            String voice = voice(a.option("voice", null), tts.voices(), m.name());
            long g0 = System.nanoTime();
            var audio = tts.speak(text, voice);
            double s = secs(g0);
            audio.writeWav(file.toFile());
            err.printf(Locale.ROOT, "[fornax] %s of audio in %s (%.1fx real time)%s%n", Fmt.seconds(audio.seconds()), Fmt.seconds(s),
                    audio.seconds() / s, voice == null ? "" : ", voice " + voice);
        }
        out.println(file);
        return 0;
    }

    /** The voice asked for, which the model must have, else its default (the first), or null for a model with none. */
    static String voice(String asked, List<String> voices, String name) {
        if (asked == null) return voices.isEmpty() ? null : voices.get(0);
        if (!voices.isEmpty() && !voices.contains(asked))
            throw new Main.Failure(name + " has no voice " + asked + "; its voices: " + String.join(" ", voices));
        return asked;
    }

    private static double secs(long t) { return (System.nanoTime() - t) / 1e9; }
}
