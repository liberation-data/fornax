package org.liberationdata.kiln.whisper;

import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.whisper.Transcript.Segment;
import org.liberationdata.kiln.whisper.Transcript.Word;
import org.liberationdata.kiln.whisper.Words.RawSegment;
import org.liberationdata.kiln.whisper.Words.Token;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class WordsTest {
    static Token special(long t) { return new Token("[_BEG_]".getBytes(), true, 1f, t, t, -1); }
    static Token tok(String text, float p, long t0, long t1) { return Token.of(text, p, t0, t1, -1); }
    static Token dtw(String text, long tDtw) { return Token.of(text, 0.9f, 0, 0, tDtw); }

    @Test
    void joinsTokensIntoWordsAtLeadingSpaces() {
        var seg = new RawSegment(0, 300, " Hello, world.", List.of(special(0),
                tok(" Hel", 0.8f, 0, 20), tok("lo", 0.6f, 20, 40), tok(",", 1f, 40, 45),
                tok(" world", 0.9f, 60, 120), tok(".", 0.9f, 120, 125), special(300)));
        List<Word> w = Words.assemble(List.of(seg)).getFirst().words();
        assertEquals(List.of("Hello,", "world."), w.stream().map(Word::text).toList());
        assertEquals((0.8f + 0.6f + 1f) / 3, w.get(0).probability(), 1e-6);
        // without DTW: from the first token's start to the last token's end
        assertEquals(0.0, w.get(0).start());
        assertEquals(0.45, w.get(0).end(), 1e-9);
        assertEquals(0.60, w.get(1).start(), 1e-9);
        assertEquals(1.25, w.get(1).end(), 1e-9);
    }

    @Test
    void decodesACharacterSplitAcrossTwoTokens() {
        byte[] e = "é".getBytes(StandardCharsets.UTF_8);          // two bytes, one per token
        var seg = new RawSegment(0, 100, " café", List.of(
                new Token(" caf".getBytes(), false, 1f, 0, 10, -1),
                new Token(new byte[] {e[0]}, false, 1f, 10, 20, -1),
                new Token(new byte[] {e[1]}, false, 1f, 20, 30, -1)));
        assertEquals("café", Words.assemble(List.of(seg)).getFirst().words().getFirst().text());
    }

    @Test
    void dtwWordsEndWhereTheNextOneStarts() {
        var a = new RawSegment(0, 120, " one two", List.of(dtw(" one", 10), dtw(" two", 50)));
        var b = new RawSegment(200, 400, " three", List.of(dtw(" three", 230)));
        List<Segment> s = Words.assemble(List.of(a, b));
        List<Word> w = s.stream().flatMap(x -> x.words().stream()).toList();
        assertEquals(0.10, w.get(0).start(), 1e-9);
        assertEquals(0.50, w.get(0).end(), 1e-9);
        assertEquals(0.50, w.get(1).start(), 1e-9);
        // the last word of a segment ends at the next segment's first word, but not after its own segment ends
        assertEquals(1.20, w.get(1).end(), 1e-9);
        assertEquals(2.30, w.get(2).start(), 1e-9);
        assertEquals(3.30, w.get(2).end(), 1e-9);  // capped at the minimum cap, 1 s, short of the segment's end (4 s)
    }

    /** A window whisper.cpp couldn't align has no DTW times: the word before it ends with its own segment. */
    @Test
    void aDtwWordBeforeAnUnalignedSegmentEndsWithItsSegment() {
        var a = new RawSegment(0, 120, " one two", List.of(dtw(" one", 10), dtw(" two", 50)));
        var b = new RawSegment(200, 400, " three", List.of(tok(" three", 1f, 210, 260)));
        List<Word> w = Words.assemble(List.of(a, b)).stream().flatMap(x -> x.words().stream()).toList();
        assertEquals(1.20, w.get(1).end(), 1e-9);
        assertEquals(2.10, w.get(2).start(), 1e-9);
        assertEquals(2.60, w.get(2).end(), 1e-9);
    }

    @Test
    void aWordBeforeAPauseIsCapped() {
        // starts 20 cs apart, then a 5 s pause: "c" would otherwise run until "d"
        var seg = new RawSegment(0, 1000, "", List.of(dtw(" a", 0), dtw(" b", 20), dtw(" c", 40), dtw(" d", 540),
                dtw(" e", 560)));
        List<Word> w = Words.assemble(List.of(seg)).getFirst().words();
        assertEquals(0.40, w.get(2).start(), 1e-9);
        assertEquals(1.40, w.get(2).end(), 1e-9);   // twice the median gap is 0.4 s; the floor is 1 s
        assertEquals(5.40, w.get(3).start(), 1e-9);
    }

    static float[] voiced(double seconds, double[]... spans) {
        float[] pcm = new float[(int) (16000 * seconds)];
        for (double[] sp : spans)
            for (int i = (int) (sp[0] * 16000); i < (int) (sp[1] * 16000); i++) pcm[i] = (float) Math.sin(i * 0.3) * 0.5f;
        return pcm;
    }

    @Test
    void wordEdgesMoveOntoTheVoice() {
        float[] pcm = voiced(3, new double[] {0.2, 0.8}, new double[] {2.0, 2.5});   // silence between
        // "first" starts on its onset and runs over the pause; "second" starts late, inside its voice
        var s = new Segment(0, 3, "", List.of(new Word(0.2, 1.9, "first", 1f), new Word(2.15, 2.9, "second", 1f)));
        List<Word> w = Words.snapToVoice(List.of(s), pcm, 16000).getFirst().words();
        assertEquals(0.18, w.get(0).start(), 1e-9);
        assertEquals(0.82, w.get(0).end(), 1e-9);
        assertEquals(1.98, w.get(1).start(), 1e-9);
        assertEquals(2.52, w.get(1).end(), 1e-9);
    }

    @Test
    void aWordThatStartsInSilenceStartsWithItsVoice() {
        float[] pcm = voiced(3, new double[] {2.0, 2.5});
        var s = new Segment(0, 3, "", List.of(new Word(1.0, 2.9, "late", 1f)));
        Word w = Words.snapToVoice(List.of(s), pcm, 16000).getFirst().words().getFirst();
        assertEquals(1.98, w.start(), 1e-9);
        assertEquals(2.52, w.end(), 1e-9);
    }

    @Test
    void runningSpeechWithoutSilenceIsLeftAlone() {
        float[] pcm = voiced(3, new double[] {0, 3});
        var s = new Segment(0, 3, "", List.of(new Word(0.5, 1.0, "a", 1f), new Word(1.0, 1.9, "b", 1f)));
        assertEquals(s.words(), Words.snapToVoice(List.of(s), pcm, 16000).getFirst().words());
    }

    @Test
    void silentAudioLeavesTheTimesAlone() {
        var s = new Segment(0, 3, "", List.of(new Word(0.5, 1.0, "a", 1f)));
        assertEquals(List.of(s), Words.snapToVoice(List.of(s), new float[16000 * 3], 16000));
        assertEquals(List.of(s), Words.snapToVoice(List.of(s), new float[100], 16000));   // under one frame
    }

    @Test
    void subtitlesSkipBlankSegmentsAndNumberOnwards() {
        var t = new Transcript("a b", "en", List.of(new Segment(0, 1, " a", List.of()), new Segment(1, 2, "  ", List.of()),
                new Segment(2, 3, " b", List.of())));
        assertEquals("1\n00:00:00,000 --> 00:00:01,000\na\n\n2\n00:00:02,000 --> 00:00:03,000\nb\n\n", t.srt());
        assertEquals("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\na\n\n00:00:02.000 --> 00:00:03.000\nb\n\n", t.vtt());
    }

    @Test
    void segmentsSpanTheirWords() {
        var s = new Segment(0, 6, " a b", List.of(new Word(0.26, 0.5, "a", 1f), new Word(0.5, 5.12, "b", 1f)));
        var empty = new Segment(6, 7, "", List.of());
        List<Segment> fit = Words.fitToWords(List.of(s, empty));
        assertEquals(0.26, fit.get(0).start());
        assertEquals(5.12, fit.get(0).end());
        assertEquals(empty, fit.get(1));
    }

    @Test
    void timesNeverGoBackwards() {
        var seg = new RawSegment(0, 500, "", List.of(tok(" a", 1, 100, 200), tok(" b", 1, 150, 180), tok(" c", 1, 300, 290)));
        List<Word> w = Words.assemble(List.of(seg)).getFirst().words();
        for (int i = 0; i < w.size(); i++) {
            assertTrue(w.get(i).end() >= w.get(i).start(), w.toString());
            if (i > 0) assertTrue(w.get(i).start() >= w.get(i - 1).end(), w.toString());
        }
    }

    @Test
    void skipsSpecialAndEmptyTokens() {
        var seg = new RawSegment(0, 100, "", List.of(special(0), tok(" ", 1, 0, 0), special(100)));
        assertTrue(Words.assemble(List.of(seg)).getFirst().words().isEmpty());
    }

    @Test
    void writesTsvAndSubtitles() {
        var t = new Transcript("Hello world.", "en", List.of(
                new Segment(0.0, 1.5, " Hello world.", List.of(new Word(0.0, 0.5, "Hello", 0.9f), new Word(0.5, 1.25, "world.", 0.8f))),
                new Segment(3661.25, 3662.0, " Again.", List.of(new Word(3661.25, 3662.0, "Again.", 0.7f)))));
        assertEquals("0.00\t0.50\tHello\n0.50\t1.25\tworld.\n3661.25\t3662.00\tAgain.\n", t.wordsTsv().replace("\r\n", "\n"));
        assertEquals("1\n00:00:00,000 --> 00:00:01,500\nHello world.\n\n2\n01:01:01,250 --> 01:01:02,000\nAgain.\n\n", t.srt());
        assertTrue(t.vtt().startsWith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello world.\n"), t.vtt());
        assertEquals(3, t.words().size());
    }

    @Test
    void escapesWhatWouldBreakTheFormats() {
        var t = new Transcript("", "en", List.of(new Segment(0.0, 1.0, " Tom & <Jerry> -->\n\n go\n", List.of(
                new Word(0.0, 0.5, "Tom\t&", 0.9f), new Word(0.5, 1.0, "line\nbreak", 0.9f)))));
        assertEquals("0.00\t0.50\tTom &\n0.50\t1.00\tline break\n", t.wordsTsv().replace("\r\n", "\n"));
        assertEquals("1\n00:00:00,000 --> 00:00:01,000\nTom & <Jerry> -->\ngo\n\n", t.srt());
        assertEquals("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTom &amp; &lt;Jerry&gt; --&gt;\ngo\n\n", t.vtt());
    }

    @Test
    void weightsByName() {
        assertEquals(WhisperModel.Weights.BASE_EN, WhisperModel.Weights.named("base.en"));
        assertEquals(WhisperModel.Weights.LARGE_V3_TURBO, WhisperModel.Weights.named("large-v3-turbo"));
        assertEquals(WhisperModel.Alignment.LARGE_V3_TURBO, WhisperModel.Weights.LARGE_V3_TURBO.alignment);
        assertTrue(Arrays.stream(WhisperModel.Weights.values()).allMatch(w -> w.file.startsWith("ggml-") && w.file.endsWith(".bin")));
    }

    /** Speech at 1-3 s and 5-8 s of 10 s, as whisper.cpp lays it out: 2 s + 0.1 s overlap + 0.1 s gap, then 3 s. */
    static Words.VadTimeline twoSegments() {
        return Words.VadTimeline.of(new long[] {100, 500}, new long[] {300, 800}, 10 * 16000, 0.1f);
    }

    @Test
    void vadTimelineLaysOutSpeechAsWhisperCppDoes() {
        var t = twoSegments();
        assertArrayEquals(new long[] {0, 220}, t.vadStart());
        assertArrayEquals(new long[] {200, 520}, t.vadEnd());
    }

    @Test
    void vadTimelineMapsDtwTimesBackToTheAudio() {
        var t = twoSegments();
        assertEquals(100, t.map(0));       // the first segment's start
        assertEquals(200, t.map(100));     // inside it, scaled
        assertEquals(300, t.map(200));     // its end
        assertEquals(300, t.map(205));     // in the overlap and gap: the nearer edge
        assertEquals(500, t.map(215));
        assertEquals(500, t.map(220));     // the second segment's start
        assertEquals(650, t.map(370));
        assertEquals(800, t.map(520));
        assertEquals(800, t.map(900));     // past the end
        assertEquals(-1, t.map(-1));       // no DTW time
    }

    @Test
    void vadTimelineWithoutSegmentsLeavesTimesAlone() {
        var t = Words.VadTimeline.of(new long[0], new long[0], 16000, 0.1f);
        assertEquals(1234, t.map(1234));
        assertEquals(-1, t.map(-1));
    }

    @Test
    void vadTimelineClampsSegmentsToTheAudio() {
        // the last segment runs to the end, past the last sample: whisper.cpp stops it there
        var t = Words.VadTimeline.of(new long[] {0}, new long[] {200}, 16000, 0.1f);
        assertArrayEquals(new long[] {0}, t.vadStart());
        assertArrayEquals(new long[] {100}, t.vadEnd());
    }
}
