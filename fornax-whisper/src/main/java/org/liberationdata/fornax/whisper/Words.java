package org.liberationdata.fornax.whisper;

import org.liberationdata.fornax.whisper.Transcript.Segment;
import org.liberationdata.fornax.whisper.Transcript.Word;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Turns whisper.cpp's tokens into words with times. Whisper's tokens are pieces of words: a token that starts with a
 * space starts a new word, and the rest (including punctuation) join the word before. A token's text is raw bytes,
 * since one UTF-8 character can be split across two tokens, so a word's bytes are joined before they are decoded.
 *
 * <p>Times: with DTW, a token's time is where the alignment path enters it, so a word starts at its first token's
 * time and ends where the next word starts (OpenAI's word_timestamps does the same). A word followed by a pause would
 * then stretch over the pause, so a word's length is capped at twice the median word length (at least one second).
 * Without DTW, a word runs from its first token's start to its last token's end, from whisper's timestamp tokens.
 * Either way, words never go backwards: each starts no earlier than the one before ends.
 */
final class Words {
    private Words() {}

    /** One token of a segment; times in centiseconds, tDtw -1 when DTW is off. */
    record Token(byte[] text, boolean special, float p, long t0, long t1, long tDtw) {
        static Token of(String text, float p, long t0, long t1, long tDtw) {
            return new Token(text.getBytes(StandardCharsets.UTF_8), false, p, t0, t1, tDtw);
        }
    }

    /** One segment as whisper.cpp reports it; times in centiseconds. */
    record RawSegment(long t0, long t1, String text, List<Token> tokens) {}

    /** A word before its times are settled; times in centiseconds. */
    private record Draft(String text, float p, long start, long tokenEnd, boolean dtw) {}

    static final long MIN_CAP_CS = 100;

    /**
     * With VAD, whisper.cpp transcribes only the speech: each segment it found, plus {@code overlap} seconds of the
     * audio after it (not after the last), with 0.1 s of silence between. It maps segment and token times back to the
     * original audio, but not DTW times, so this does, the same way: a time inside a segment's speech is scaled into
     * it, and one in the overlap or silence after it snaps to the nearer of the two segment edges. The layout follows
     * {@code whisper_vad} and {@code whisper_map_token_time_segment_aware} in whisper.cpp.
     */
    record VadTimeline(long[] origStart, long[] origEnd, long[] vadStart, long[] vadEnd) {
        static final VadTimeline NONE = new VadTimeline(new long[0], new long[0], new long[0], new long[0]);
        static final int SAMPLE_RATE = WhisperModel.SAMPLE_RATE, GAP_SAMPLES = SAMPLE_RATE / 10;

        /**
         * @param t0 the segments' starts, t1 their ends (original audio, centiseconds)
         * @param samples the length of the audio whisper was given
         * @param overlap seconds of audio carried past each segment's end
         */
        static VadTimeline of(long[] t0, long[] t1, int samples, float overlap) {
            int n = t0.length;
            if (n == 0) return NONE;
            long[] vs = new long[n], ve = new long[n];
            int overlapSamples = (int) (overlap * SAMPLE_RATE), offset = 0;
            for (int i = 0; i < n; i++) {
                int start = Math.min(toSamples(t0[i]), samples - 1), end = Math.min(toSamples(t1[i]), samples - 1);
                int speech = end - start;
                if (i < n - 1) end = Math.min(end + overlapSamples, samples - 1);
                vs[i] = toCs(offset);
                ve[i] = toCs(offset + speech);
                offset += end - start;
                if (i < n - 1) offset += GAP_SAMPLES;
            }
            return new VadTimeline(t0.clone(), t1.clone(), vs, ve);
        }

        /** A time on the audio whisper heard, in centiseconds, on the original audio; -1 stays -1. */
        long map(long t) {
            int n = origStart.length;
            if (n == 0 || t < 0) return t;
            if (t <= vadStart[0]) return origStart[0];
            if (t >= vadEnd[n - 1]) return origEnd[n - 1];
            for (int i = 0; i < n; i++) {
                if (t >= vadStart[i] && t <= vadEnd[i]) {
                    long vd = vadEnd[i] - vadStart[i], od = origEnd[i] - origStart[i];
                    return vd <= 0 ? origStart[i] : origStart[i] + (t - vadStart[i]) * od / vd;
                }
                if (i + 1 < n && t > vadEnd[i] && t < vadStart[i + 1])
                    return t <= (vadEnd[i] + vadStart[i + 1]) / 2 ? origEnd[i] : origStart[i + 1];
            }
            return t;
        }

        private static int toSamples(long cs) { return (int) (cs / 100.0 * SAMPLE_RATE + 0.5); }
        private static long toCs(int samples) { return (long) (samples / (double) SAMPLE_RATE * 100.0 + 0.5); }
    }

    /** Segments with their words, timed across segment boundaries. */
    static List<Segment> assemble(List<RawSegment> raw) {
        List<List<Draft>> drafts = raw.stream().map(s -> draft(s.tokens())).toList();
        long cap = cap(drafts, raw);
        List<Segment> out = new ArrayList<>();
        long prevEnd = Long.MIN_VALUE;
        for (int si = 0; si < raw.size(); si++) {
            RawSegment s = raw.get(si);
            List<Draft> ds = drafts.get(si);
            List<Word> words = new ArrayList<>();
            for (int wi = 0; wi < ds.size(); wi++) {
                Draft d = ds.get(wi);
                long start = d.start(), end;
                if (d.dtw()) {
                    Long next = nextStart(drafts, si, wi);
                    end = next == null ? s.t1() : wi + 1 < ds.size() ? next : Math.min(next, s.t1());
                    end = Math.min(end, start + cap);
                } else {
                    end = d.tokenEnd();
                }
                start = Math.max(start, prevEnd);
                end = Math.max(end, start);
                prevEnd = end;
                words.add(new Word(start / 100.0, end / 100.0, d.text(), d.p()));
            }
            out.add(new Segment(s.t0() / 100.0, s.t1() / 100.0, s.text(), words));
        }
        return out;
    }

    /**
     * Moves word edges onto the voice where there is silence to go by. Whisper's times are good to a frame or two in
     * running speech but drift at pauses: DTW starts a word after a pause late, and ends the word before it where the
     * next one starts. So, in 10 ms frames, with silence meaning 35 dB below the audio's loud level (its
     * 95th-percentile frame):
     * <ul>
     *   <li>a word that starts in silence starts at the first voiced frame after it;</li>
     *   <li>a word that starts in voice, with silence shortly before (within 0.5 s, and after the previous word),
     *   starts where that silence ends;</li>
     *   <li>a word ends at its last voiced frame.</li>
     * </ul>
     * Each edge gets 20 ms of margin. Audio without silence (music under a voice) is left as it is.
     */
    static List<Segment> snapToVoice(List<Segment> segments, float[] pcm, int rate) {
        int w = rate / 100, frames = pcm.length / w;
        if (frames == 0) return segments;
        float[] rms = new float[frames];
        for (int f = 0; f < frames; f++) {
            double s = 0;
            for (int i = f * w; i < (f + 1) * w; i++) s += pcm[i] * pcm[i];
            rms[f] = (float) Math.sqrt(s / w);
        }
        float[] sorted = rms.clone();
        Arrays.sort(sorted);
        double threshold = sorted[(int) (0.95 * (frames - 1))] * Math.pow(10, -35 / 20.0);
        if (threshold <= 0) return segments;
        List<Segment> out = new ArrayList<>(segments.size());
        double prevEnd = 0;
        for (Segment s : segments) {
            List<Word> words = new ArrayList<>(s.words().size());
            for (Word word : s.words()) {
                double start = word.start(), end = word.end();
                int f0 = (int) Math.floor(start * 100), f1 = Math.min(frames, (int) Math.ceil(end * 100));
                if (f0 < f1 && rms[f0] <= threshold) {
                    for (int f = f0 + 1; f < f1; f++) if (rms[f] > threshold) { start = Math.max(start, f / 100.0 - 0.02); break; }
                } else if (f0 < frames) {
                    int lo = Math.max((int) Math.ceil(prevEnd * 100), f0 - 50), g = f0;
                    while (g - 1 >= lo && rms[g - 1] > threshold) g--;
                    if (g - 1 >= lo) start = Math.max(prevEnd, Math.min(start, g / 100.0 - 0.02));
                }
                for (int f = f1 - 1; f >= Math.max(0, f0); f--) {
                    if (rms[f] > threshold) { end = Math.min(end, (f + 1) / 100.0 + 0.02); break; }
                }
                end = Math.max(end, start);
                prevEnd = end;
                words.add(new Word(start, end, word.text(), word.probability()));
            }
            out.add(new Segment(s.start(), s.end(), s.text(), words));
        }
        return out;
    }

    /**
     * Each segment from its first word's start to its last word's end. Whisper's own segment times come from its
     * timestamp tokens, which large-v3-turbo rounds to whole seconds or so.
     */
    static List<Segment> fitToWords(List<Segment> segments) {
        return segments.stream().map(s -> s.words().isEmpty() ? s
                : new Segment(s.words().getFirst().start(), s.words().getLast().end(), s.text(), s.words())).toList();
    }

    /** The DTW start of the word after word wi of segment si, in this or a later segment; null if it's the last. */
    private static Long nextStart(List<List<Draft>> drafts, int si, int wi) {
        for (int s = si, w = wi + 1; s < drafts.size(); s++, w = 0) {
            if (w < drafts.get(s).size()) {
                Draft d = drafts.get(s).get(w);
                return d.dtw() ? d.start() : null;
            }
        }
        return null;
    }

    /** The longest a DTW-timed word may last: twice the median gap between word starts, at least a second. */
    private static long cap(List<List<Draft>> drafts, List<RawSegment> raw) {
        List<Long> gaps = new ArrayList<>();
        for (List<Draft> ds : drafts)
            for (int i = 0; i + 1 < ds.size(); i++)
                if (ds.get(i).dtw() && ds.get(i + 1).dtw()) gaps.add(Math.max(0, ds.get(i + 1).start() - ds.get(i).start()));
        if (gaps.isEmpty()) return Long.MAX_VALUE / 4;
        long[] g = gaps.stream().mapToLong(Long::longValue).toArray();
        Arrays.sort(g);
        return Math.max(MIN_CAP_CS, 2 * g[g.length / 2]);
    }

    /** Groups a segment's text tokens into words. */
    private static List<Draft> draft(List<Token> tokens) {
        List<Draft> words = new ArrayList<>();
        List<Token> cur = new ArrayList<>();
        for (Token t : tokens) {
            if (t.special()) continue;
            boolean startsWord = t.text().length > 0 && t.text()[0] == ' ';
            if (startsWord && !cur.isEmpty()) { add(words, cur); cur = new ArrayList<>(); }
            cur.add(t);
        }
        if (!cur.isEmpty()) add(words, cur);
        return words;
    }

    private static void add(List<Draft> words, List<Token> tokens) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        double p = 0;
        for (Token t : tokens) { bytes.writeBytes(t.text()); p += t.p(); }
        String text = bytes.toString(StandardCharsets.UTF_8).strip();
        Token first = tokens.getFirst(), last = tokens.getLast();
        if (text.isEmpty()) return;
        boolean dtw = first.tDtw() >= 0;
        words.add(new Draft(text, (float) (p / tokens.size()), dtw ? first.tDtw() : first.t0(), last.t1(), dtw));
    }
}
