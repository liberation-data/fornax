package org.liberationdata.kiln.examples;

import org.liberationdata.kiln.Audio;
import org.liberationdata.kiln.whisper.Transcript;
import org.liberationdata.kiln.whisper.WhisperModel;
import org.liberationdata.kiln.whisper.WhisperModel.Alignment;
import org.liberationdata.kiln.whisper.WhisperModel.Options;
import org.liberationdata.kiln.whisper.WhisperModel.Weights;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Transcribes a WAV file: prints the text and writes the word timings and subtitles beside it.
 * <pre>
 * Transcribe audio.wav [--language de] [--translate] [--prompt "Names, spellings"] [--beam 1] [--threads 8] [--no-dtw] [--cpu]
 * </pre>
 * writes audio.words.tsv (start, end, word per line) and audio.srt. -Dmodel picks the weights: large-v3-turbo (default),
 * large-v3-turbo-q8_0, base.en, or a path to a ggml .bin (timed from whisper's timestamp tokens, without DTW).
 * Default model: hf download ggerganov/whisper.cpp ggml-large-v3-turbo-q5_0.bin
 */
public class Transcribe {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("usage: Transcribe audio.wav [--language xx] [--translate] [--prompt text] [--beam n] [--threads n] [--no-dtw] [--cpu]");
            System.exit(2);
        }
        Path in = Path.of(args[0]);
        Options o = Options.DEFAULT;
        boolean dtw = true, gpu = true;
        for (int i = 1; i < args.length; i++) {
            switch (args[i]) {
                case "--language" -> o = o.withLanguage(args[++i]);
                case "--translate" -> o = o.withTranslate(true);
                case "--prompt" -> o = o.withInitialPrompt(args[++i]);
                case "--beam" -> o = o.withBeamSize(Integer.parseInt(args[++i]));
                case "--threads" -> o = o.withThreads(Integer.parseInt(args[++i]));
                case "--no-dtw" -> dtw = false;
                case "--cpu" -> gpu = false;
                default -> throw new IllegalArgumentException("unknown option: " + args[i]);
            }
        }
        Audio audio;
        try {
            audio = Audio.readWav(in.toFile());
        } catch (java.io.IOException e) {
            throw new IllegalArgumentException(in + ": " + e.getMessage()
                    + " (convert other formats first, e.g. ffmpeg -i in.mp3 -ar 16000 -ac 1 in.wav)", e);
        }
        String m = System.getProperty("model", "large-v3-turbo");
        long t0 = System.nanoTime();
        try (var stt = m.endsWith(".bin") ? new WhisperModel(m, Alignment.NONE, gpu)
                : new WhisperModel(Weights.named(m).path(), dtw ? Weights.named(m).alignment : Alignment.NONE, gpu)) {
            System.out.printf("[kiln] loaded %s (%s) in %.1fs%n", m, stt.type(), (System.nanoTime() - t0) / 1e9);
            long g0 = System.nanoTime();
            Transcript t = stt.transcribe(audio, o, new WhisperModel.Listener() {
                @Override public void segment(Transcript.Segment s) {
                    System.out.printf("[%7.2f --> %7.2f] %s%n", s.start(), s.end(), s.text().strip());
                }
            });
            double s = (System.nanoTime() - g0) / 1e9;
            String base = in.getFileName().toString().replaceFirst("\\.[^.]+$", "");
            Path dir = in.toAbsolutePath().getParent();
            Path tsv = dir.resolve(base + ".words.tsv"), srt = dir.resolve(base + ".srt");
            Files.writeString(tsv, t.wordsTsv());
            Files.writeString(srt, t.srt());
            System.out.println();
            System.out.println(t.text());
            System.out.printf("%n[kiln] %s: %.1fs of audio in %.1fs (%.1fx real time), %d words (%.1f words/s) -> %s, %s%n",
                    t.language(), audio.seconds(), s, audio.seconds() / s, t.words().size(), t.words().size() / s,
                    tsv.getFileName(), srt.getFileName());
        }
    }
}
