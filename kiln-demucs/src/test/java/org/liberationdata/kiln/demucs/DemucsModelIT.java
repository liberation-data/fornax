package org.liberationdata.kiln.demucs;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.Audio;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletionException;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Separates a synthetic mix with HTDemucs. Runs when {@code htdemucs-42M-v1.0-F32.gguf} is in the Hugging Face cache
 * and the native libraries are built; skipped otherwise.
 *
 * <p>{@link #matchesTheReferenceVocals} also needs a song and the vocals torchaudio's Hybrid Demucs made from it, which
 * aren't in the repository: {@code -Pkiln.demucs.song=song.wav -Pkiln.demucs.vocals=song-vocals.wav} (or
 * {@code KILN_DEMUCS_SONG} and {@code KILN_DEMUCS_VOCALS}). With {@code -Pkiln.demucs.out=<dir>} (or
 * {@code KILN_DEMUCS_OUT}) it also separates the whole song, times it, and writes the stems there.
 */
class DemucsModelIT {
    static DemucsModel demucs;
    static Audio mix;

    @BeforeAll
    static void setUp() {
        var w = DemucsModel.Weights.HTDEMUCS;
        assumeTrue(w.downloaded(), "htdemucs not downloaded: hf download " + w.repo + " " + w.file);
        demucs = new DemucsModel(w);
        mix = syntheticMix(48000, 12);
    }

    @AfterAll
    static void tearDown() { if (demucs != null) demucs.close(); }

    /** A plucked bass line, a kick on every beat and noise hats between, at 120 bpm: no vocals. */
    static Audio syntheticMix(int rate, int seconds) {
        int n = rate * seconds, beat = rate / 2;
        float[] l = new float[n], r = new float[n];
        Random rnd = new Random(7);
        double[] notes = {55, 55, 73.42, 65.41};
        for (int i = 0; i < n; i++) {
            int b = i / beat, t = i % beat;
            double s = t / (double) rate, f = notes[b % notes.length];
            double bass = 0.25 * Math.exp(-3 * s) * (Math.sin(2 * Math.PI * f * s) + 0.5 * Math.sin(4 * Math.PI * f * s));
            double kick = 0.5 * Math.exp(-25 * s) * Math.sin(2 * Math.PI * (50 + 100 * Math.exp(-30 * s)) * s);
            int h = (i + beat / 2) % beat;
            double hat = 0.08 * Math.exp(-200 * h / (double) rate) * (rnd.nextDouble() * 2 - 1);
            l[i] = (float) (bass + kick + hat);
            r[i] = (float) (bass + kick + 0.7 * hat);
        }
        return new Audio(l, r, rate);
    }

    // ---- measures ----

    /** Signal to distortion ratio of an estimate against a reference, both channels, in dB. */
    static double sdr(Audio ref, Audio est) {
        double s = 0, e = 0;
        for (float[][] p : new float[][][] {{ref.left(), est.left()}, {ref.right(), est.right()}}) {
            for (int i = 0; i < p[0].length; i++) {
                double d = p[0][i] - p[1][i];
                s += p[0][i] * (double) p[0][i];
                e += d * d;
            }
        }
        return 10 * Math.log10(s / e);
    }

    /** Scale-invariant SDR: the estimate's error after the best scaling of the reference, in dB. */
    static double siSdr(Audio ref, Audio est) {
        double dot = 0, rr = 0;
        for (float[][] p : new float[][][] {{ref.left(), est.left()}, {ref.right(), est.right()}})
            for (int i = 0; i < p[0].length; i++) { dot += p[0][i] * (double) p[1][i]; rr += p[0][i] * (double) p[0][i]; }
        double a = dot / rr, s = 0, e = 0;
        for (float[][] p : new float[][][] {{ref.left(), est.left()}, {ref.right(), est.right()}})
            for (int i = 0; i < p[0].length; i++) {
                double t = a * p[0][i], d = t - p[1][i];
                s += t * t;
                e += d * d;
            }
        return 10 * Math.log10(s / e);
    }

    /** The energy of mix minus the sum of the stems, relative to the mix's, in dB. */
    static double residualDb(Audio mix, Stems stems) { return -sdr(mix, stems.mix()); }

    static double rms(Audio a) {
        double s = 0;
        for (int i = 0; i < a.left().length; i++) s += a.left()[i] * (double) a.left()[i] + a.right()[i] * (double) a.right()[i];
        return Math.sqrt(s / (2.0 * a.left().length));
    }

    static Audio slice(Audio a, double fromSeconds, double seconds) {
        int from = (int) (fromSeconds * a.sampleRate()), to = Math.min(a.left().length, from + (int) (seconds * a.sampleRate()));
        return new Audio(Arrays.copyOfRange(a.left(), from, to), Arrays.copyOfRange(a.right(), from, to), a.sampleRate());
    }

    // ---- the synthetic mix ----

    @Test
    void separatesIntoFourStemsAtTheInputsRateAndLength() {
        assertEquals("htdemucs", demucs.name());
        assertEquals(List.of("drums", "bass", "other", "vocals"), demucs.sources());
        assertEquals(44100, demucs.sampleRate());
        System.out.println("[demucs] backend " + demucs.backend());

        Stems s = demucs.separate(mix);
        assertEquals(demucs.sources(), s.names());
        assertEquals(48000, s.sampleRate());
        for (String name : s.names()) assertEquals(mix.left().length, s.get(name).left().length, name);

        double residual = residualDb(mix, s);
        System.out.printf("[demucs] synthetic mix: residual %.1f dB; rms drums %.4f bass %.4f other %.4f vocals %.4f (mix %.4f)%n",
                residual, rms(s.get("drums")), rms(s.get("bass")), rms(s.get("other")), rms(s.vocals()), rms(mix));
        assertTrue(residual < -20, "stems don't add up to the mix: residual " + residual + " dB");
        // there is no voice in it, and the kick and the bass go where they belong
        assertTrue(rms(s.vocals()) < 0.1 * rms(mix), "vocals in an instrumental: " + rms(s.vocals()));
        assertTrue(rms(s.get("drums")) > 0.3 * rms(mix), "drums " + rms(s.get("drums")));
        assertTrue(rms(s.get("bass")) > 0.2 * rms(mix), "bass " + rms(s.get("bass")));
    }

    @Test
    void reportsProgressSegmentBySegment() {
        List<int[]> seen = new ArrayList<>();
        demucs.separate(mix, DemucsModel.Options.DEFAULT, (done, total) -> seen.add(new int[] {done, total}));
        assertFalse(seen.isEmpty());
        int total = seen.getLast()[1];
        assertTrue(total >= 2, "12 s is more than one 7.8 s segment: " + total);
        assertEquals(total, seen.getLast()[0]);
        for (int i = 1; i < seen.size(); i++) assertTrue(seen.get(i)[0] > seen.get(i - 1)[0]);
    }

    static int segmentPasses(Audio audio, DemucsModel.Options o) {
        int[] total = {0};
        demucs.separate(audio, o, (done, t) -> total[0] = t);
        return total[0];
    }

    /** Each option crosses FFM as its own argument; each one changes what the engine does. */
    @Test
    void optionsReachTheEngine() {
        var o = DemucsModel.Options.DEFAULT;
        int plain = segmentPasses(mix, o);
        assertEquals(2 * plain, segmentPasses(mix, o.withShifts(2)), "a pass per shift");
        assertTrue(segmentPasses(mix, o.withOverlap(0.5f)) > plain, "more overlap, more segments");
        Audio clip = slice(mix, 0, 3);
        float[] a = demucs.separate(clip, o.withShifts(1).withSeed(1)).get("drums").left();
        float[] b = demucs.separate(clip, o.withShifts(1).withSeed(2)).get("drums").left();
        assertFalse(Arrays.equals(a, b), "the seed picks the shift");
    }

    @Test
    void runsOnTheCpu() {
        try (var cpu = new DemucsModel(DemucsModel.Weights.HTDEMUCS.path(), false, 0)) {
            assertEquals("CPU", cpu.backend());
            assertEquals(48000, cpu.separate(slice(mix, 0, 1)).vocals().left().length);
        }
    }

    @Test
    void cancelsFromTheProgressCallbackAndKeepsWorking() {
        assertThrows(CancellationException.class,
                () -> demucs.separate(mix, DemucsModel.Options.DEFAULT, (done, total) -> demucs.cancel()));
        assertEquals(4, demucs.separate(slice(mix, 0, 2)).names().size());
    }

    @Test
    void anExceptionInTheProgressCallbackStopsTheCall() {
        var e = assertThrows(IllegalStateException.class, () -> demucs.separate(mix, DemucsModel.Options.DEFAULT,
                (done, total) -> { throw new IllegalStateException("boom"); }));
        assertEquals("boom", e.getMessage());
    }

    /** At the last segment there is no cancel check left, so the engine finishes; the exception still reaches us. */
    @Test
    void anExceptionAtTheLastSegmentStillStopsTheCall() {
        var e = assertThrows(IllegalStateException.class, () -> demucs.separate(mix, DemucsModel.Options.DEFAULT,
                (done, total) -> { if (done == total) throw new IllegalStateException("last"); }));
        assertEquals("last", e.getMessage());
        assertEquals(4, demucs.separate(slice(mix, 0, 1)).names().size());
    }

    /** A second model used from the first one's callback, on the same thread, must not take its progress over. */
    @Test
    void anotherModelCalledFromTheProgressCallbackLeavesItsProgressAlone() {
        try (var other = new DemucsModel(DemucsModel.Weights.HTDEMUCS)) {
            List<Integer> seen = new ArrayList<>();
            int[] total = {0};
            demucs.separate(mix, DemucsModel.Options.DEFAULT, (done, t) -> {
                seen.add(done);
                total[0] = t;
                if (done == 1) other.separate(slice(mix, 0, 1), DemucsModel.Options.DEFAULT, (d, tt) -> {});
            });
            assertEquals(total[0], seen.getLast(), "progress after the nested call was lost: " + seen);
            var e = assertThrows(IllegalStateException.class, () -> demucs.separate(mix, DemucsModel.Options.DEFAULT, (done, t) -> {
                if (done == 1) other.separate(slice(mix, 0, 1), DemucsModel.Options.DEFAULT, (d, tt) -> {});
                if (done == 2) throw new IllegalStateException("after");
            }));
            assertEquals("after", e.getMessage());
        }
    }

    /**
     * close() or separate() on the same model from its own callback would free the weights under the running native
     * call, or wait for itself: both are refused, the refusal comes back from the outer call, and the model still works.
     */
    @Test
    void closeOrSeparateFromItsOwnCallbackIsRefused() {
        var e = assertThrows(IllegalStateException.class,
                () -> demucs.separate(mix, DemucsModel.Options.DEFAULT, (done, total) -> demucs.close()));
        assertTrue(e.getMessage().contains("close()"), e.getMessage());
        e = assertThrows(IllegalStateException.class, () -> demucs.separate(mix, DemucsModel.Options.DEFAULT,
                (done, total) -> demucs.separate(slice(mix, 0, 1))));
        assertTrue(e.getMessage().contains("own callbacks"), e.getMessage());
        assertEquals(4, demucs.separate(slice(mix, 0, 1)).names().size());
    }

    @Test
    void separatesAsyncAndCancelsThroughTheFuture() {
        Stems s = demucs.separateAsync(slice(mix, 0, 3)).join();
        assertEquals(3 * 48000, s.vocals().left().length);

        var started = new java.util.concurrent.CountDownLatch(1);
        var f = demucs.separateAsync(mix, DemucsModel.Options.DEFAULT, (done, total) -> started.countDown());
        try { started.await(); } catch (InterruptedException ex) { throw new RuntimeException(ex); }
        f.cancel(true);
        assertThrows(CancellationException.class, f::join);
        assertEquals(4, demucs.separateAsync(slice(mix, 0, 1)).join().names().size());
    }

    @Test
    void shortAndOddInputWorks() {
        Audio tiny = slice(mix.resample(22050), 0, 0.3);
        Stems s = demucs.separate(tiny);
        assertEquals(tiny.left().length, s.vocals().left().length);
        assertEquals(22050, s.sampleRate());
        assertThrows(IllegalArgumentException.class, () -> demucs.separate(new Audio(new float[0], new float[0], 44100)));
        var e = assertThrows(IllegalArgumentException.class, () -> demucs.separate(new Audio(new float[10], new float[10], 0)));
        assertTrue(e.getMessage().contains("sample rate"), e.getMessage());
    }

    @Test
    void aMissingModelFileIsAnError() {
        var e = assertThrows(IllegalStateException.class, () -> new DemucsModel("/nonexistent/htdemucs.gguf"));
        assertTrue(e.getMessage().contains("/nonexistent/htdemucs.gguf"), e.getMessage());
    }

    @Test
    void closeIsIdempotentAndLaterCallsAreRefused() {
        var m = new DemucsModel(DemucsModel.Weights.HTDEMUCS);
        m.close();
        m.close();
        assertThrows(IllegalStateException.class, () -> m.separate(slice(mix, 0, 1)));
    }

    /** close() cancels the running separation through the native flag, fails the queued one, and only then frees. */
    @Test
    void closeDuringAnAsyncCallCancelsItAndQueuedCalls() throws Exception {
        var m = new DemucsModel(DemucsModel.Weights.HTDEMUCS);
        var started = new java.util.concurrent.CountDownLatch(1);
        var running = m.separateAsync(syntheticMix(44100, 60), DemucsModel.Options.DEFAULT, (done, total) -> started.countDown());
        var queued = m.separateAsync(slice(mix, 0, 1));
        assertTrue(started.await(60, java.util.concurrent.TimeUnit.SECONDS));
        long t0 = System.nanoTime();
        m.close();
        long closeMs = (System.nanoTime() - t0) / 1_000_000;
        assertThrows(CancellationException.class, running::join);
        assertThrows(CancellationException.class, queued::join);
        assertTrue(closeMs < 3000, "close waited " + closeMs + " ms: the separation wasn't cancelled");
        assertThrows(IllegalStateException.class, () -> m.separate(slice(mix, 0, 1)));
    }

    // ---- the episode's song against torchaudio's Hybrid Demucs ----

    static String setting(String property, String env) {
        String v = System.getProperty(property);
        return v != null && !v.isBlank() ? v : System.getenv(env);
    }

    @Test
    void matchesTheReferenceVocals() throws Exception {
        String song = setting("kiln.demucs.song", "KILN_DEMUCS_SONG"), vocals = setting("kiln.demucs.vocals", "KILN_DEMUCS_VOCALS");
        assumeTrue(song != null && new File(song).isFile(), "no song: set -Pkiln.demucs.song=song.wav (KILN_DEMUCS_SONG)");
        assumeTrue(vocals != null && new File(vocals).isFile(), "no reference vocals: set -Pkiln.demucs.vocals (KILN_DEMUCS_VOCALS)");
        Audio full = Audio.readWav(new File(song)), refFull = Audio.readWav(new File(vocals));
        assertEquals(full.sampleRate(), refFull.sampleRate());

        // 20 s with singing in it, from 40 s in
        Audio excerpt = slice(full, 40, 20), ref = slice(refFull, 40, 20);
        long t0 = System.nanoTime();
        Stems s = demucs.separate(excerpt);
        double secs = (System.nanoTime() - t0) / 1e9;
        double sdr = sdr(ref, s.vocals()), si = siSdr(ref, s.vocals()), residual = residualDb(excerpt, s);
        System.out.printf("[demucs] 20 s excerpt in %.2f s: vocals SDR %.2f dB, SI-SDR %.2f dB against torchaudio; residual %.1f dB%n",
                secs, sdr, si, residual);
        assertTrue(sdr > 10, "vocals SDR " + sdr);
        assertTrue(residual < -20, "residual " + residual);

        String out = setting("kiln.demucs.out", "KILN_DEMUCS_OUT");
        if (out == null) return;
        File dir = new File(out);
        assertTrue(dir.isDirectory() || dir.mkdirs(), "cannot create " + dir);
        t0 = System.nanoTime();
        Stems all = demucs.separate(full);
        secs = (System.nanoTime() - t0) / 1e9;
        String base = new File(song).getName().replaceFirst("\\.[^.]+$", "");
        all.vocals().writeWav(new File(dir, base + "-vocals.wav"));
        all.backing().writeWav(new File(dir, base + "-backing.wav"));
        System.out.printf("[demucs] whole song, %.1f s of audio in %.1f s (%.1f s of audio per second): vocals SDR %.2f dB, "
                        + "SI-SDR %.2f dB against torchaudio; residual %.1f dB -> %s%n", full.seconds(), secs, full.seconds() / secs,
                sdr(refFull, all.vocals()), siSdr(refFull, all.vocals()), residualDb(full, all), dir);
    }
}
