---
title: Speech (text to speech)
description: Speak text from Java with Kokoro, ZONOS3, LFM3-Audio, VoxCPM2, Qwen3-TTS and the other models llama.cpp's audio generation runs, with built-in voices or a reference clip.
---

`fornax-tts` turns text into speech through llama.cpp's audio generation (its `mtmd` library). One engine runs every
text-to-speech model llama.cpp supports. Each model is two GGUF files: the backbone, and an **mmproj** that holds the
audio decoder and the model's built-in voices. The class is `TtsModel`.

## Speak a line

```sh
hf download ggml-org/Kokoro-82M-GGUF Kokoro-82M-F32.gguf mmproj-Kokoro-82M-F16.gguf
```

<!-- snippet: Speech#speak -->
```java
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.tts.TtsModel;

import java.io.File;

try (var tts = new TtsModel(TtsModel.Weights.KOKORO)) {
    System.out.println(tts.voices());                      // the built-in voices, the default first
    Audio audio = tts.speak("Hello from the JVM.", "bm_george");
    audio.writeWav(new File("hello.wav"));
    System.out.printf("%.1f s of audio at %d Hz%n", audio.seconds(), audio.sampleRate());
}
```

A `null` voice uses the model's default. For a model with built-in voices, an unknown voice fails with a message that
lists them.

`Audio` is stereo float PCM in [-1, 1] (`left`, `right`, `sampleRate`); a mono model plays on both sides.
`writeWav` writes a 16-bit stereo WAV.

## The models Fornax knows by name

`TtsModel.Weights` names the models Fornax has been tried with, with their Hugging Face repository and files.
`new TtsModel(Weights)` finds both files in the Hugging Face cache; if they aren't there, it fails with the
`hf download` command that fetches them. `Weights.named("zonos3")` finds one by name, and `downloaded()` says whether
its files are there.

| `Weights` | Voices | Licence | Notes |
|---|---|---|---|
| `KOKORO` | 54: American and British English, plus Spanish, French, Hindi, Italian, Japanese, Brazilian Portuguese and Chinese | Apache-2.0 | small (82M) and fast: about 17× real time on the 8060S. The author grades the voices: `af_heart` A, `af_bella` A- |
| `ZONOS3` | 27: American, British and Irish English, plus other languages | Apache-2.0 | 44.1 kHz, about 8× real time; British, Irish and announcer male voices |
| `LFM3_AUDIO` | `female` (default), `male`; US English | LFM Open License v1.0 | free below US$10M annual revenue; a licence is needed above that |
| `CHATTERBOX_TURBO` | `default` | MIT | |
| `SOPRANO` | one fixed voice | Apache-2.0 | 80M, very fast, 32 kHz |
| `TINY_VOCA` | `M2` (default), `F2` | Apache-2.0 | 4.5M parameters; a toy, and sounds robotic |
| `VOXCPM2` | none: describe one (see below) | Apache-2.0 | 2B, 48 kHz; slow, about 0.8× real time on the 8060S |
| `QWEN3_TTS` | none: give it a reference clip (see below) | Apache-2.0 | Qwen3-TTS 1.7B Base |

Any other TTS GGUF that llama.cpp runs works too: pass both paths. Either can be a
[model reference](/docs/guides/models/#by-reference) instead, downloaded on first use, and the catalogue alias
`kokoro` brings both files:

<!-- snippet: Speech#files -->
```java
try (var tts = new TtsModel("/models/model.gguf", "/models/mmproj.gguf")) {   // or (model, mmproj, gpuLayers, contextSize)
    // ...
}
new TtsModel("kokoro");
new TtsModel("ggml-org/Kokoro-82M-GGUF:F32", "ggml-org/Kokoro-82M-GGUF:mmproj-Kokoro-82M-F16.gguf");
```

Call `voices()` for a model's exact voice names. Check each model's licence before you use what it makes.

## Designing a voice: VoxCPM2

VoxCPM2 has no built-in voices. Instead, a description in parentheses at the start of the text designs one:

<!-- snippet: Speech#voxcpm -->
```java
try (var tts = new TtsModel(TtsModel.Weights.VOXCPM2)) {
    var options = TtsModel.Options.DEFAULT.withSeed(1);
    tts.speak("(A calm male narrator)Fornax runs open models inside your JVM.", null, options, null)
       .writeWav(new File("narrator.wav"));
}
```

A designed voice changes from run to run. Keep the seed fixed, and the description the same, for lines that should
sound alike.

## Speaking like a reference clip

For a model that clones a voice (VoxCPM2 and Qwen3-TTS), pass `@` and the path of a WAV or MP3 clip as the voice:

<!-- snippet: Speech#clone -->
```java
try (var tts = new TtsModel(TtsModel.Weights.QWEN3_TTS)) {
    tts.speak("This is my voice, reading something new.", "@/clips/me.wav").writeWav(new File("me-new.wav"));
}
```

Put the clip's transcript beside it, in a file named after the clip with `.txt` added (`/clips/me.wav.txt`). With
the transcript, the model continues that speaker rather than only imitating the sound, which holds an accent much
more steadily. Without the file, the clip is used on its own. Qwen3-TTS needs a reference encoder in its mmproj for
the transcript, and the published one (`Weights.QWEN3_TTS`) has none, so it ignores the transcript:
`scripts/convert-qwen3-tts.sh` converts one that has it, to load with `new TtsModel(model, mmproj)`.

Only clone a voice you have the right to use.

## Options

`TtsModel.Options` is a record: `lang`, `temperature`, `seed`, `speed` and `maxFrames`. `Options.DEFAULT` leaves all
of them to the model. `withLang` picks a language for a multilingual model, in the codes that model uses; `null` is
its default.

<!-- snippet: Speech#options -->
```java
var options = TtsModel.Options.DEFAULT
        .withSeed(7)            // -1 (the default) is random
        .withTemperature(0.6f)  // below 0 is the model's default
        .withSpeed(1.1f)        // 0 or less is the model's own pace; most models take no other value
        .withMaxFrames(0);      // 0 or less is the model's limit
```

## Streaming

Pass a `Consumer<Audio>` and the audio arrives in chunks as it is decoded, so playback can start before the whole
line is ready. The call still returns all of it.

<!-- snippet: Speech#stream -->
```java
Audio all = tts.speak("A longer passage to read aloud.", null,
        chunk -> System.out.printf("%.2f s more%n", chunk.seconds()));   // or hand each chunk to a player
```

`cancel()`, from another thread, stops a streaming call and returns the audio so far (Kokoro, which decodes the
whole text in one step, streams a single chunk, so it has nothing to stop). If the callback throws, speech stops and
`speak` throws that exception. `speakAsync` runs on the model's own thread and returns a `CompletableFuture`, with or
without a `Flow.Subscriber` for the chunks: see [Threading](/docs/guides/threading/). Without a callback, `speak`
runs exactly as llama.cpp's `llama-tts` does. At the same seed, Fornax's output matches `llama-tts` to within 16-bit
rounding for Kokoro, ZONOS3 and LFM3-Audio.

## Try it

```sh
./gradlew speak --args='out.wav "Hello from the JVM" bm_george'
./gradlew speak --args='out.wav "Hello from the JVM"' -Pmodel=zonos3      # no voice: lists them, uses the default
./gradlew speak --args='out.wav "Hello"' -Pmodel=/path/model.gguf -Pmmproj=/path/mmproj.gguf
./gradlew speakBatch --args='lines.tsv' -Pmodel=voxcpm2                   # one out.wav<TAB>text per line, one seed
```
