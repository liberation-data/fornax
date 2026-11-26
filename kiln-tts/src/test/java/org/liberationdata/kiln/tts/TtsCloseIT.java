package org.liberationdata.kiln.tts;

import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.Audio;

import java.util.Collections;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** close() against Kokoro-82M: each test loads its own instance and closes it. */
class TtsCloseIT {
    static final TtsModel.Weights W = TtsModel.Weights.KOKORO;
    static final String TEXT = String.join(" ", Collections.nCopies(12, "Kiln runs open generative models inside your Java application."));

    @Test
    void closeTwiceIsSafe() {
        assumeTrue(W.downloaded(), "model not downloaded: hf download " + W.repo);
        var m = new TtsModel(W);
        m.close();
        m.close();
        assertThrows(IllegalStateException.class, () -> m.speak("Hello.", null));
    }

    /**
     * Kokoro isn't autoregressive: it decodes the whole text in one step and streams it as a single chunk, so there is
     * nothing to stop part way. close() then waits for the call and frees the model after it.
     */
    @Test
    void closeDuringACallThatCannotStopWaitsForIt() throws Exception {
        assumeTrue(W.downloaded(), "model not downloaded: hf download " + W.repo);
        var m = new TtsModel(W);
        var f = m.speakAsync(TEXT, "af_heart", TtsModel.Options.DEFAULT);
        var queued = m.speakAsync("Queued.", "af_heart", TtsModel.Options.DEFAULT);
        Thread.sleep(200);
        m.close();
        Audio all = f.get(10, TimeUnit.SECONDS);
        assertTrue(all.seconds() > 30, "spoke only " + all.seconds() + " s");
        assertThrows(CancellationException.class, queued::join);
    }
}
