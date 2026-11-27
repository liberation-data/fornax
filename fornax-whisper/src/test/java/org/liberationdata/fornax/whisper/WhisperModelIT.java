package org.liberationdata.fornax.whisper;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.tts.TtsModel;
import org.liberationdata.fornax.whisper.Transcript.Segment;
import org.liberationdata.fornax.whisper.Transcript.Word;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CancellationException;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Speaks two sentences with Kokoro, a known gap apart, and transcribes them with whisper (base.en if it's downloaded,
 * else large-v3-turbo). Runs when both models are in the Hugging Face cache and the native libraries are built;
 * skipped otherwise. Two engines, one JVM.
 */
class WhisperModelIT {
    static final String FIRST = "Fornax runs open generative models inside your Java application.";
    static final String SECOND = "Speech becomes text, with a time for every word.";
    static final double GAP = 1.0;

    static WhisperModel stt;
    static Audio audio;
    static double secondOnset;

    @BeforeAll
    static void setUp() {
        assumeTrue(TtsModel.Weights.KOKORO.downloaded(), "Kokoro not downloaded: hf download " + TtsModel.Weights.KOKORO.repo);
        var w = WhisperModel.Weights.BASE_EN.downloaded() ? WhisperModel.Weights.BASE_EN : WhisperModel.Weights.LARGE_V3_TURBO;
        assumeTrue(w.downloaded(), "whisper not downloaded: hf download " + w.repo + " " + w.file);
        float[] a, b;
        int rate;
        try (var tts = new TtsModel(TtsModel.Weights.KOKORO)) {
            rate = tts.sampleRate();
            a = tts.speak(FIRST, "bm_george").left();
            b = tts.speak(SECOND, "af_heart").left();
        }
        float[] all = new float[a.length + (int) (GAP * rate) + b.length];
        System.arraycopy(a, 0, all, 0, a.length);
        int off = a.length + (int) (GAP * rate);
        System.arraycopy(b, 0, all, off, b.length);
        audio = Audio.ofMono(all, rate);                       // 24 kHz: transcribe resamples it
        secondOnset = (off + onset(b)) / (double) rate;
        stt = new WhisperModel(w);
    }

    /** An alias loads the same file, with the alignment its {@link WhisperModel.Weights} entry has. */
    @Test
    void loadsByAlias() {
        assumeTrue(WhisperModel.Weights.BASE_EN.downloaded(), "whisper base.en not downloaded");
        try (var byAlias = new WhisperModel("whisper-base.en")) {
            Transcript t = byAlias.transcribe(audio);
            assertTrue(t.text().toLowerCase(java.util.Locale.ROOT).contains("speech becomes text"), t.text());
            assertFalse(t.words().isEmpty());
        }
        var e = assertThrows(IllegalArgumentException.class, () -> new WhisperModel("kokoro"));
        assertEquals("kokoro is a speech model (llama.cpp); WhisperModel loads speech-to-text models", e.getMessage());
    }

    /** The first sample of a 10 ms window louder than -40 dBFS. */
    static int onset(float[] x) {
        for (int i = 0; i + 240 <= x.length; i += 240) {
            double s = 0;
            for (int j = i; j < i + 240; j++) s += x[j] * x[j];
            if (Math.sqrt(s / 240) > 0.01) return i;
        }
        return 0;
    }

    @AfterAll
    static void tearDown() { if (stt != null) stt.close(); }

    static String norm(String s) { return s.toLowerCase(Locale.ROOT).replaceAll("[^a-z ]", ""); }

    @Test
    void transcribesWithPlausibleWordTimes() {
        Transcript t = stt.transcribe(audio, WhisperModel.Options.DEFAULT.withLanguage("en"));
        System.out.println(t.text());
        System.out.print(t.wordsTsv());
        assertEquals("en", t.language());
        String text = norm(t.text());
        assertTrue(text.contains("generative models"), t.text());
        assertTrue(text.contains("java application"), t.text());
        assertTrue(text.contains("every word"), t.text());

        List<Word> words = t.words();
        assertTrue(words.size() >= 15 && words.size() <= 22, "words: " + words.size());
        for (int i = 0; i < words.size(); i++) {
            Word w = words.get(i);
            assertTrue(w.end() >= w.start(), w.toString());
            assertTrue(w.end() - w.start() < 2.0, w.toString());
            assertTrue(w.start() >= 0 && w.end() <= audio.seconds() + 0.05, w.toString());
            assertTrue(w.probability() > 0 && w.probability() <= 1, w.toString());
            if (i > 0) assertTrue(w.start() >= words.get(i - 1).end(), words.get(i - 1) + " then " + w);
        }
        // 18 words in about 9 s of audio, a narrator's pace
        double rate = words.size() / (words.getLast().end() - words.getFirst().start() - GAP);
        assertTrue(rate > 1.5 && rate < 4.5, "words/s " + rate);

        // the second sentence starts on its onset, after the gap
        Word speechWord = words.stream().filter(w -> norm(w.text()).equals("speech")).findFirst().orElseThrow();
        assertEquals(secondOnset, speechWord.start(), 0.15, "onset of \"Speech\"");
        Word before = words.get(words.indexOf(speechWord) - 1);
        assertTrue(before.end() < secondOnset - GAP * 0.7, "the word before the gap runs into it: " + before);
    }

    @Test
    void streamsSegmentsAndProgress() {
        List<Segment> streamed = new ArrayList<>();
        List<Integer> progress = new ArrayList<>();
        Transcript t = stt.transcribe(audio, WhisperModel.Options.DEFAULT.withBeamSize(1), new WhisperModel.Listener() {
            @Override public void progress(int percent) { progress.add(percent); }
            @Override public void segment(Segment s) { streamed.add(s); }
        });
        assertEquals(t.segments().size(), streamed.size());
        for (int i = 0; i < streamed.size(); i++) assertEquals(t.segments().get(i).text(), streamed.get(i).text());
        assertEquals(100, progress.getLast());
    }

    @Test
    void cancelsFromTheListener() {
        assertThrows(CancellationException.class, () -> stt.transcribe(audio, WhisperModel.Options.DEFAULT,
                new WhisperModel.Listener() {
                    @Override public void progress(int percent) { stt.cancel(); }
                }));
        // and the model still works afterwards
        assertFalse(stt.transcribe(audio, WhisperModel.Options.DEFAULT.withBeamSize(1)).text().isBlank());
    }

    /** whisper.cpp keeps the last spectrogram and skips input under 100 ms, which would hear the last call again. */
    @Test
    void emptyAndVeryShortAudioDoNotRepeatTheLastCall() {
        var greedy = WhisperModel.Options.DEFAULT.withBeamSize(1);
        assertTrue(norm(stt.transcribe(audio, greedy).text()).contains("generative"));
        for (float[] pcm : new float[][] {new float[0], new float[800]}) {
            Transcript t = stt.transcribe(pcm, greedy, null);
            assertFalse(norm(t.text()).contains("generative"), t.text());
            assertEquals("en", t.language());
        }
    }

    @Test
    void anUnknownLanguageIsRefused() {
        var e = assertThrows(IllegalArgumentException.class,
                () -> stt.transcribe(new float[0], WhisperModel.Options.DEFAULT.withLanguage("klingon"), null));
        assertTrue(e.getMessage().contains("klingon"), e.getMessage());
    }

    @Test
    void anExceptionInTheListenerStopsTheCall() {
        var e = assertThrows(RuntimeException.class, () -> stt.transcribe(audio, WhisperModel.Options.DEFAULT,
                new WhisperModel.Listener() {
                    @Override public void segment(Segment s) { throw new IllegalStateException("boom"); }
                }));
        assertEquals("boom", e.getCause().getMessage());
    }
}
