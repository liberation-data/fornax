package org.liberationdata.kiln.tts;

import org.liberationdata.kiln.Audio;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * A chunk consumer that calls another model, on the same thread, inside the outer call's upcall (#33). Each call's
 * sink is bound into its own upcall stub, so the inner call can't replace or clear the outer one's. Kokoro, tiny-voca
 * and Soprano all hand back an utterance in one chunk, so this can't show a lost later chunk; it checks that both
 * streams are delivered whole and that the outer consumer's exception still surfaces after the inner call. Runs
 * against Kokoro when it's in the Hugging Face cache; skipped otherwise.
 */
class TtsNestedCallIT {
    static final TtsModel.Weights W = TtsModel.Weights.KOKORO;
    static final String TEXT = "Kiln runs open generative models inside your Java application. It loads the weights once. "
            + "Then every call is a plain method call, with no server between.";

    @Test
    void anotherModelSpeakingFromAChunkConsumerLeavesTheOuterStreamAlone() {
        assumeTrue(W.downloaded(), "model not downloaded: hf download " + W.repo);
        try (var outerModel = new TtsModel(W); var innerModel = new TtsModel(W)) {
            String voice = outerModel.voices().isEmpty() ? null : outerModel.voices().getFirst();
            List<Audio> outer = new ArrayList<>(), inner = new ArrayList<>(), innerResult = new ArrayList<>();
            Audio a = outerModel.speak(TEXT, voice, TtsModel.Options.DEFAULT.withSeed(1), chunk -> {
                if (outer.isEmpty()) innerResult.add(innerModel.speak("Hello.", voice, TtsModel.Options.DEFAULT, inner::add));
                outer.add(chunk);
            });
            assertFalse(inner.isEmpty(), "the inner call streamed");
            assertEquals(innerResult.getFirst().left().length, inner.stream().mapToInt(c -> c.left().length).sum(),
                    "the inner consumer got every chunk");
            assertEquals(a.left().length, outer.stream().mapToInt(c -> c.left().length).sum(),
                    "the outer consumer got every chunk");

            var boom = new IllegalStateException("the outer consumer broke");
            assertSame(boom, assertThrows(IllegalStateException.class, () -> outerModel.speak(TEXT, voice,
                    TtsModel.Options.DEFAULT, chunk -> {
                        innerModel.speak("Hello.", voice, TtsModel.Options.DEFAULT, c -> {});
                        throw boom;
                    })));
        }
    }
}
