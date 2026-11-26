package org.liberationdata.fornax.snippets;

// --8<-- [start:speak]
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.tts.TtsModel;

import java.io.File;
// --8<-- [end:speak]

/** The snippets of website/src/content/docs/docs/guides/speech.md: compiled, never run. */
class Speech {
    void speak() throws Exception {
        // --8<-- [start:speak]
        try (var tts = new TtsModel(TtsModel.Weights.KOKORO)) {
            System.out.println(tts.voices());                      // the built-in voices, the default first
            Audio audio = tts.speak("Hello from the JVM.", "bm_george");
            audio.writeWav(new File("hello.wav"));
            System.out.printf("%.1f s of audio at %d Hz%n", audio.seconds(), audio.sampleRate());
        }
        // --8<-- [end:speak]
    }

    @SuppressWarnings("try")
    void files() {
        // --8<-- [start:files]
        try (var tts = new TtsModel("/models/model.gguf", "/models/mmproj.gguf")) {   // or (model, mmproj, gpuLayers, contextSize)
            // ...
        }
        new TtsModel("kokoro");
        new TtsModel("ggml-org/Kokoro-82M-GGUF:F32", "ggml-org/Kokoro-82M-GGUF:mmproj-Kokoro-82M-F16.gguf");
        // --8<-- [end:files]
    }

    void voxcpm() throws Exception {
        // --8<-- [start:voxcpm]
        try (var tts = new TtsModel(TtsModel.Weights.VOXCPM2)) {
            var options = TtsModel.Options.DEFAULT.withSeed(1);
            tts.speak("(A calm male narrator)Fornax runs open models inside your JVM.", null, options, null)
               .writeWav(new File("narrator.wav"));
        }
        // --8<-- [end:voxcpm]
    }

    void cloneVoice() throws Exception {
        // --8<-- [start:clone]
        try (var tts = new TtsModel(TtsModel.Weights.QWEN3_TTS)) {
            tts.speak("This is my voice, reading something new.", "@/clips/me.wav").writeWav(new File("me-new.wav"));
        }
        // --8<-- [end:clone]
    }

    TtsModel.Options options() {
        // --8<-- [start:options]
        var options = TtsModel.Options.DEFAULT
                .withSeed(7)            // -1 (the default) is random
                .withTemperature(0.6f)  // below 0 is the model's default
                .withSpeed(1.1f)        // 0 or less is the model's own pace; most models take no other value
                .withMaxFrames(0);      // 0 or less is the model's limit
        // --8<-- [end:options]
        return options;
    }

    Audio stream(TtsModel tts) {
        // --8<-- [start:stream]
        Audio all = tts.speak("A longer passage to read aloud.", null,
                chunk -> System.out.printf("%.2f s more%n", chunk.seconds()));   // or hand each chunk to a player
        // --8<-- [end:stream]
        return all;
    }
}
