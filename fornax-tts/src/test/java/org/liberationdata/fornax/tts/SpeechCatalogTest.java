package org.liberationdata.fornax.tts;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.hub.ModelCatalog;
import org.liberationdata.fornax.hub.ModelSet;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/** The catalogue's speech sets are the files TtsModel.Weights names, so fornax pull fetches what TtsModel loads. */
class SpeechCatalogTest {
    @Test void eachSpeechSetHasItsWeightsFiles() {
        Map<String, TtsModel.Weights> sets = Map.of("kokoro", TtsModel.Weights.KOKORO, "zonos3-mini", TtsModel.Weights.ZONOS3,
                "voxcpm2", TtsModel.Weights.VOXCPM2);
        sets.forEach((id, w) -> {
            ModelSet m = ModelCatalog.require(id);
            assertEquals(ModelSet.Kind.SPEECH, m.kind(), id);
            var gguf = m.part("gguf").orElseThrow().primary();
            var mmproj = m.part("mmproj").orElseThrow().primary();
            assertEquals(w.repo, gguf.repo(), id);
            assertEquals(w.modelFile, gguf.file(), id);
            assertEquals(w.repo, mmproj.repo(), id);
            assertEquals(w.mmprojFile, mmproj.file(), id);
        });
    }
}
