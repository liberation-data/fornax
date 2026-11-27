package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.cli.Args.UsageException;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.whisper.WhisperModel;
import org.liberationdata.fornax.whisper.WhisperModel.Alignment;
import org.liberationdata.fornax.whisper.WhisperModel.Options;
import org.liberationdata.fornax.whisper.WhisperModel.Weights;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Set;

/** {@code fornax transcribe}: a WAV file to text, or to words with their times, with a whisper.cpp model. */
final class Transcribe {
    static final Set<String> OPTIONS = Set.of("model", "language");
    static final Set<String> FLAGS = Set.of("words");
    static final String MODEL = "whisper-base.en";

    static final String USAGE = """
            usage: fornax transcribe <audio.wav> [options]

            Prints what is said in a WAV file (any rate, mono or stereo; convert other formats first, e.g.
            ffmpeg -i in.mp3 in.wav). The model is pulled first if it isn't downloaded yet.

              --model M            whisper-base.en (default: English, small and fast), whisper-large-v3-turbo (99
                                   languages), org/name:file.bin or a path to a whisper.cpp model
              --language xx        the language spoken (default: detected; an .en model hears English only)
              --words              a line per word instead: start and end in seconds, then the word""";

    static int run(Args a, FornaxHub hub, PrintStream out, PrintStream err) throws IOException {
        String wav = a.positional(0);
        if (wav == null) throw new UsageException("which audio file?");
        if (a.positional().size() > 1) throw new UsageException("one audio file at a time, not " + a.positional());
        Path in = Path.of(wav);
        if (!Files.isRegularFile(in)) throw new Main.Failure("no such file: " + wav);
        Audio audio;
        try { audio = Audio.readWav(in.toFile()); }
        catch (IOException e) {
            throw new Main.Failure(wav + ": " + e.getMessage() + " (convert other formats first, e.g. ffmpeg -i in.mp3 in.wav)");
        }
        Options o = a.option("language", null) == null ? Options.DEFAULT : Options.DEFAULT.withLanguage(a.option("language", null));
        var m = Models.fetch(a.option("model", MODEL), Kind.TRANSCRIPTION, "transcribe", true, hub, err, Main.terminal());
        Path model = m.file("model");

        err.println("[fornax] loading " + m.name());
        long t0 = System.nanoTime();
        try (var stt = new WhisperModel(model.toString(), alignment(model), true)) {
            err.printf("[fornax] loaded in %s%n", Fmt.seconds(secs(t0)));
            long g0 = System.nanoTime();
            var t = stt.transcribe(audio, o);
            double s = secs(g0);
            out.print(a.has("words") ? t.wordsTsv() : t.text().strip() + "\n");
            out.flush();
            err.printf(Locale.ROOT, "[fornax] %s: %s of audio in %s (%.1fx real time), %d words%n", t.language(),
                    Fmt.seconds(audio.seconds()), Fmt.seconds(s), audio.seconds() / s, t.words().size());
        }
        return 0;
    }

    /** The DTW alignment heads for one of whisper.cpp's own files (word times within a frame or two), else none. */
    static Alignment alignment(Path model) {
        String name = model.getFileName().toString();
        for (Weights w : Weights.values()) if (w.file.equals(name)) return w.alignment;
        return Alignment.NONE;
    }

    private static double secs(long t) { return (System.nanoTime() - t) / 1e9; }
}
