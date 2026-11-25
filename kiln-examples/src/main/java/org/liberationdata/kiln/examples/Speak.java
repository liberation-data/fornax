package org.liberationdata.kiln.examples;

import org.liberationdata.kiln.tts.TtsModel;

import java.io.File;

/**
 * Speaks text to a WAV. Usage: Speak out.wav "text" [voice] [-Dmodel=kokoro|zonos3|lfm3-audio|tiny-voca|chatterbox-turbo]
 * (or -Dmodel=/path/model.gguf -Dmmproj=/path/mmproj.gguf). With no voice, lists the model's voices and uses its default.
 * Default model: hf download ggml-org/Kokoro-82M-GGUF
 */
public class Speak {
    public static void main(String[] args) throws Exception {
        String out = args.length > 0 ? args[0] : "out/speak.wav";
        String text = args.length > 1 ? args[1]
                : "Kiln runs open generative models inside your Java application, and they run on your GPU.";
        String voice = args.length > 2 ? args[2] : null;
        String m = System.getProperty("model", "kokoro");
        long t0 = System.nanoTime();
        try (var tts = m.endsWith(".gguf") ? new TtsModel(m, System.getProperty("mmproj"))
                                           : new TtsModel(TtsModel.Weights.named(m))) {
            System.out.printf("[kiln] loaded %s in %.1fs%n", m, (System.nanoTime() - t0) / 1e9);
            if (voice == null && !tts.voices().isEmpty()) System.out.println("[kiln] voices: " + String.join(" ", tts.voices()));
            long g0 = System.nanoTime();
            var audio = tts.speak(text, voice);
            double s = (System.nanoTime() - g0) / 1e9;
            File f = new File(out);
            if (f.getParentFile() != null) f.getParentFile().mkdirs();
            audio.writeWav(f);
            System.out.printf("[kiln] %.1fs of audio in %.1fs (%.1fx real time) -> %s%n", audio.seconds(), s, audio.seconds() / s, out);
        }
    }
}
