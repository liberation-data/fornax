package org.liberationdata.kiln.tts;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.Audio;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Flow;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Runs against Kokoro-82M when it's in the Hugging Face cache and the native library is built; skipped otherwise. */
class TtsModelIT {
    static final TtsModel.Weights W = TtsModel.Weights.KOKORO;
    static final String TEXT = "Kiln runs open generative models inside your Java application.";
    static TtsModel model;

    @BeforeAll
    static void load() {
        assumeTrue(W.downloaded(), "model not downloaded: hf download " + W.repo);
        model = new TtsModel(W);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    static double rms(float[] x) {
        double s = 0;
        for (float v : x) s += v * v;
        return Math.sqrt(s / Math.max(1, x.length));
    }

    @Test
    void listsItsVoices() {
        assertTrue(model.voices().contains("am_michael"), model.voices().toString());
        assertEquals("af_heart", model.voices().getFirst());
        assertEquals(24000, model.sampleRate());
    }

    @Test
    void speaksASentence() {
        Audio a = model.speak(TEXT, "bm_george");
        // 9 words at a narrator's 2-3.5 words a second
        assertTrue(a.seconds() > 2 && a.seconds() < 8, "seconds: " + a.seconds());
        assertTrue(rms(a.left()) > 0.01, "rms: " + rms(a.left()));
    }

    @Test
    void streamsTheSameAudio() {
        List<Audio> chunks = new ArrayList<>();
        Audio a = model.speak(TEXT, "af_heart", TtsModel.Options.DEFAULT.withSeed(1), chunks::add);
        assertFalse(chunks.isEmpty());
        assertEquals(a.left().length, chunks.stream().mapToInt(c -> c.left().length).sum());
        assertTrue(rms(a.left()) > 0.01);
    }

    @Test
    void rejectsAnUnknownVoice() {
        var e = assertThrows(IllegalStateException.class, () -> model.speak(TEXT, "nobody"));
        assertTrue(e.getMessage().contains("am_michael"), e.getMessage());
    }

    @Test
    void aThrowingChunkConsumerSurfacesInsteadOfCrashing() {
        var boom = new IllegalStateException("the consumer broke");
        assertSame(boom, assertThrows(IllegalStateException.class, () -> model.speak(TEXT, "af_heart", chunk -> { throw boom; })));
        assertTrue(model.speak("Still here.", "af_heart").seconds() > 0.3, "the model still works");
    }

    /** Records chunks; cancels its subscription after cancelAfter of them (or never, for -1). */
    static class Chunks implements Flow.Subscriber<Audio> {
        final List<Audio> items = Collections.synchronizedList(new ArrayList<>());
        final CountDownLatch done = new CountDownLatch(1);
        final int cancelAfter;
        Flow.Subscription subscription;
        volatile Throwable error;

        Chunks(int cancelAfter) { this.cancelAfter = cancelAfter; }

        @Override public void onSubscribe(Flow.Subscription s) { subscription = s; s.request(Long.MAX_VALUE); }
        @Override public void onNext(Audio a) { items.add(a); if (items.size() == cancelAfter) subscription.cancel(); }
        @Override public void onError(Throwable t) { error = t; done.countDown(); }
        @Override public void onComplete() { done.countDown(); }
    }

    @Test
    void speakAsyncStreamsTheSameAudioToASubscriber() throws Exception {
        var chunks = new Chunks(-1);
        Audio a = model.speakAsync(TEXT, "af_heart", TtsModel.Options.DEFAULT.withSeed(1), chunks).get(60, TimeUnit.SECONDS);
        assertTrue(chunks.done.await(30, TimeUnit.SECONDS));
        assertNull(chunks.error);
        assertFalse(chunks.items.isEmpty());
        assertEquals(a.left().length, chunks.items.stream().mapToInt(c -> c.left().length).sum());
        assertTrue(model.speakAsync(TEXT, "bm_george", TtsModel.Options.DEFAULT).get(60, TimeUnit.SECONDS).seconds() > 2);
    }
}
