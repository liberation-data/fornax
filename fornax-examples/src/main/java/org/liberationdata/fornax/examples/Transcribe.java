package org.liberationdata.fornax.examples;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.whisper.Transcript;
import org.liberationdata.fornax.whisper.WhisperModel;
import org.liberationdata.fornax.whisper.WhisperModel.Alignment;
import org.liberationdata.fornax.whisper.WhisperModel.Options;
import org.liberationdata.fornax.whisper.WhisperModel.Vad;
import org.liberationdata.fornax.whisper.WhisperModel.VadWeights;
import org.liberationdata.fornax.whisper.WhisperModel.Weights;

import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Transcribes a WAV file: prints the text and writes the word timings and subtitles beside it.
 * <pre>
 * Transcribe audio.wav [--language de] [--translate] [--prompt "Names, spellings"] [--beam 1] [--threads 8] [--no-dtw] [--cpu]
 *                      [--vad [silero-v6.2.0|model.bin]] [--no-context] [--suppress-nst] [--temperature-inc 0.2]
 *                      [--entropy 2.4] [--logprob -1] [--no-speech 0.6]
 * </pre>
 * writes audio.words.tsv (start, end, word per line) and audio.srt. -Dmodel picks the weights: large-v3-turbo (default),
 * large-v3-turbo-q8_0, base.en, or a path to a ggml .bin (timed from whisper's timestamp tokens, without DTW).
 * --vad transcribes only the speech Silero VAD finds (default weights: hf download ggml-org/whisper-vad ggml-silero-v6.2.0.bin).
 * --no-context stops prompting each 30 s window with the text before it.
 * Default model: hf download ggerganov/whisper.cpp ggml-large-v3-turbo-q5_0.bin
 */
public class Transcribe {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("usage: Transcribe audio.wav [--language xx] [--translate] [--prompt text] [--beam n] [--threads n] [--no-dtw] [--cpu]"
                    + " [--vad [weights|model.bin]] [--no-context] [--suppress-nst] [--temperature-inc t] [--entropy t] [--logprob t] [--no-speech t]");
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
                case "--vad" -> {
                    String v = i + 1 < args.length && !args[i + 1].startsWith("--") ? args[++i] : "silero-v6.2.0";
                    o = o.withVad(v.endsWith(".bin") ? Vad.of(v) : Vad.of(VadWeights.named(v)));
                }
                case "--no-context" -> o = o.withConditionOnPreviousText(false);
                case "--suppress-nst" -> o = o.withSuppressNonSpeech(true);
                case "--temperature-inc" -> o = o.withTemperatureIncrement(Float.parseFloat(args[++i]));
                case "--entropy" -> o = o.withEntropyThreshold(Float.parseFloat(args[++i]));
                case "--logprob" -> o = o.withLogProbThreshold(Float.parseFloat(args[++i]));
                case "--no-speech" -> o = o.withNoSpeechThreshold(Float.parseFloat(args[++i]));
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
            System.out.printf("[fornax] loaded %s (%s) in %.1fs%n", m, stt.type(), (System.nanoTime() - t0) / 1e9);
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
            System.out.printf("%n[fornax] %s: %.1fs of audio in %.1fs (%.1fx real time), %d words (%.1f words/s) -> %s, %s%n",
                    t.language(), audio.seconds(), s, audio.seconds() / s, t.words().size(), t.words().size() / s,
                    tsv.getFileName(), srt.getFileName());
        }
    }
}
