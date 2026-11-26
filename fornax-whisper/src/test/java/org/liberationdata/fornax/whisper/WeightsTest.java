package org.liberationdata.fornax.whisper;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.whisper.WhisperModel.Alignment;
import org.liberationdata.fornax.whisper.WhisperModel.Weights;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/** Which alignment a model loaded by reference gets: no native library needed. */
class WeightsTest {
    @TempDir Path dir;

    @Test void anAliasOrReferenceToAWeightsFileGetsItsAlignment() {
        assertEquals(Alignment.LARGE_V3_TURBO, Weights.alignmentOf("whisper-large-v3-turbo"));
        assertEquals(Alignment.BASE_EN, Weights.alignmentOf("whisper-base.en"));
        assertEquals(Alignment.BASE_EN, Weights.alignmentOf("ggerganov/whisper.cpp:ggml-base.en.bin"));
        assertEquals(Alignment.LARGE_V3_TURBO, Weights.alignmentOf("ggerganov/whisper.cpp@main:ggml-large-v3-turbo-q8_0.bin"));
        assertEquals(Alignment.BASE_EN, Weights.alignmentOf(" ggerganov/whisper.cpp:ggml-base.en.bin "));   // loads, stripped, as the hub reads it
    }

    @Test void anythingElseGetsNone() throws Exception {
        assertEquals(Alignment.NONE, Weights.alignmentOf("ggerganov/whisper.cpp:ggml-tiny.bin"));   // not an entry
        assertEquals(Alignment.NONE, Weights.alignmentOf("someone/whisper-copy:ggml-base.en.bin"));   // another repo
        Path file = Files.writeString(dir.resolve("ggml-base.en.bin"), "x");
        assertEquals(Alignment.NONE, Weights.alignmentOf(file.toString()));   // a path keeps "without DTW"
    }
}
