package org.liberationdata.kiln.whisper;

import java.util.List;
import java.util.Locale;

/**
 * What {@link WhisperModel#transcribe} heard: the text, its segments (a phrase or sentence each, with start and end
 * times) and the words in them, each with its own start, end and probability. Times are in seconds from the start of
 * the audio.
 *
 * @param language the language transcribed (detected, unless one was given), e.g. "en"
 */
public record Transcript(String text, String language, List<Segment> segments) {
    public Transcript {
        segments = List.copyOf(segments);
    }

    /**
     * A stretch of speech, usually a sentence or phrase, from its first word's start to its last word's end.
     * {@code text} is whisper's own, with its punctuation.
     */
    public record Segment(double start, double end, String text, List<Word> words) {
        public Segment {
            words = List.copyOf(words);
        }
    }

    /**
     * One word. {@code text} keeps punctuation that whisper attached to it ("Hello,"). {@code probability} is the mean
     * of its tokens' probabilities: low values flag words the model was unsure of.
     */
    public record Word(double start, double end, String text, float probability) {}

    /** Every word, in order. */
    public List<Word> words() {
        return segments.stream().flatMap(s -> s.words().stream()).toList();
    }

    /** One line per word: start, end (seconds, two decimals) and the word, separated by tabs. */
    public String wordsTsv() {
        StringBuilder b = new StringBuilder();
        for (Word w : words()) b.append(String.format(Locale.ROOT, "%.2f\t%.2f\t%s%n", w.start(), w.end(), w.text()));
        return b.toString();
    }

    /** SubRip subtitles, one cue per segment. */
    public String srt() {
        StringBuilder b = new StringBuilder();
        int i = 1;
        for (Segment s : segments) {
            if (s.text().isBlank()) continue;
            b.append(i++).append('\n').append(time(s.start(), ',')).append(" --> ").append(time(s.end(), ','))
             .append('\n').append(s.text().strip()).append("\n\n");
        }
        return b.toString();
    }

    /** WebVTT subtitles, one cue per segment. */
    public String vtt() {
        StringBuilder b = new StringBuilder("WEBVTT\n\n");
        for (Segment s : segments) {
            if (s.text().isBlank()) continue;
            b.append(time(s.start(), '.')).append(" --> ").append(time(s.end(), '.')).append('\n')
             .append(s.text().strip()).append("\n\n");
        }
        return b.toString();
    }

    /** hh:mm:ss,mmm (SRT) or hh:mm:ss.mmm (VTT). */
    static String time(double seconds, char sep) {
        long ms = Math.round(Math.max(0, seconds) * 1000);
        return String.format(Locale.ROOT, "%02d:%02d:%02d%c%03d", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, sep, ms % 1000);
    }
}
