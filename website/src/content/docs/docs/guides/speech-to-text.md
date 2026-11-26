---
title: Speech to text
description: Transcribe audio from Java with Whisper through whisper.cpp, with a start, end and probability for every word, and write word timings, SRT or WebVTT subtitles.
---

`kiln-whisper` turns speech into text through [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Every word comes
with its own start and end time and a probability, which is what captions, lyric titles and editing to the word need.
The class is `WhisperModel`.

## Transcribe a file

```sh
hf download ggerganov/whisper.cpp ggml-large-v3-turbo-q5_0.bin
```

```java
import org.liberationdata.kiln.Audio;
import org.liberationdata.kiln.whisper.Transcript;
import org.liberationdata.kiln.whisper.WhisperModel;

import java.io.File;

try (var stt = new WhisperModel(WhisperModel.Weights.LARGE_V3_TURBO)) {
    Transcript t = stt.transcribe(Audio.readWav(new File("talk.wav")));
    System.out.println(t.text());
    for (Transcript.Word w : t.words())
        System.out.printf("%6.2f %6.2f %4.2f %s%n", w.start(), w.end(), w.probability(), w.text());
}
```

Any `Audio` works: Kiln mixes it to mono and resamples it to Whisper's 16 kHz. `Audio.readWav` reads 8, 16, 24 and
32-bit PCM and float WAV files. For MP3 and other formats, convert first (`ffmpeg -i in.mp3 in.wav`). If you already
have 16 kHz mono samples, pass the `float[]` straight to `transcribe(pcm, options, listener)`.

A `Transcript` has:

- `text`: everything Whisper heard, with its punctuation;
- `language`: the language it transcribed, detected unless you gave one;
- `segments`: phrases or sentences, each with `start`, `end`, `text` and its `words`;
- `words()`: every `Word` in order, with `start`, `end`, `text` and `probability`. Punctuation stays attached to the
  word before it ("Hello,"). A low probability flags a word the model was unsure of.

Times are in seconds from the start of the audio. Word times never go backwards: each word starts at or after the end
of the one before.

## Subtitles and word timings

```java
Files.writeString(Path.of("talk.srt"), t.srt());          // one cue per segment
Files.writeString(Path.of("talk.vtt"), t.vtt());          // WebVTT
Files.writeString(Path.of("talk.words.tsv"), t.wordsTsv()); // start <TAB> end <TAB> word, one word per line
```

## How words are timed

Whisper itself predicts timestamps only every few seconds. Kiln times each word in two steps:

1. **DTW alignment.** A few of the decoder's attention heads line text up with audio. whisper.cpp runs dynamic time
   warping over them, which puts each token on a 20 ms audio frame. A word starts at its first token and ends where the
   next word starts. Every `Weights` entry knows its model's alignment heads.
2. **Onto the voice.** Alignment drifts at pauses: a word after a pause starts a little late, and the word before a
   pause would run on into it. Where the audio has silence (35 dB below its loud level), Kiln moves a word's start to
   where the voice starts and its end to where the voice stops. Speech over music, with no silence, is left as DTW
   placed it.

On 55 s of synthesized narration (eleven sentences, a known gap between each), sentence starts landed within 0.03 s of
the true onset on average (at most 0.12 s), and sentence ends within 0.08 s (at most 0.2 s).

For a model file that isn't one of the `Weights`, give its alignment heads, or `Alignment.NONE` to time words from
Whisper's timestamp tokens instead. Those are rougher: in the same test, starts were up to 0.5 s off and ends ran into
the next sentence.

```java
new WhisperModel("/models/ggml-small.en.bin", WhisperModel.Alignment.SMALL_EN);
new WhisperModel("/models/ggml-custom.bin");                 // no DTW
new WhisperModel("/models/ggml-small.bin", WhisperModel.Alignment.SMALL, false);   // on the CPU
```

On a machine with no GPU, running on the CPU also needs `-Dkiln.allowCpu=true` (or `KILN_ALLOW_CPU=true`): otherwise
Kiln refuses to load its GPU build there ([Kiln says it found no GPU](/docs/gpu-drivers/#kiln-says-it-found-no-gpu)).

DTW needs whisper.cpp's flash attention off, so Kiln turns it off for a model with alignment heads. That costs some
speed: about 50% longer than without DTW in the test above.

## The models Kiln knows by name

`WhisperModel.Weights` names the files from whisper.cpp's own repository on Hugging Face (`ggerganov/whisper.cpp`).
`new WhisperModel(Weights)` finds the file in the Hugging Face cache; if it isn't there, it fails with the
`hf download` command that fetches it. `Weights.named("base.en")` finds one by name.

| `Weights` | File | Size | Notes |
|---|---|---|---|
| `LARGE_V3_TURBO` | `ggml-large-v3-turbo-q5_0.bin` | 574 MB | the default: 99 languages, close to large-v3's accuracy, much faster |
| `LARGE_V3_TURBO_Q8_0` | `ggml-large-v3-turbo-q8_0.bin` | 874 MB | the same model at 8 bits |
| `BASE_EN` | `ggml-base.en.bin` | 148 MB | English only, small and fast |

Any other ggml Whisper model works too: pass its path (and its `Alignment` for DTW). The Whisper weights are MIT
licensed.

## Options

`WhisperModel.Options` is a record. `Options.DEFAULT` detects the language and decodes with a beam of 5, the way
whisper.cpp's CLI does.

```java
var options = WhisperModel.Options.DEFAULT
        .withLanguage("de")             // null or "auto" detects it; an English-only model always transcribes English
        .withTranslate(true)            // German speech, English text
        .withInitialPrompt("Kiln, ggml, ComfyUI")   // names and spellings to expect
        .withBeamSize(1)                // greedy: faster; 5 (the default) is a little more accurate
        .withThreads(8)                 // 0 or less is whisper.cpp's default
        .withTemperature(-1);           // below 0 is the default: 0, raised when a window fails its checks
```

large-v3-turbo was trained to transcribe, not to translate, so it translates poorly; for translation, use a full
large or medium model.

## Progress and streaming

A `Listener` hears about the transcription as it runs, on the calling thread: `progress(percent)`, and each
`segment(segment)` once Whisper has finished it, with its words. The call still returns the whole `Transcript`.

```java
Transcript t = stt.transcribe(audio, WhisperModel.Options.DEFAULT, new WhisperModel.Listener() {
    @Override public void progress(int percent) { System.out.printf("%d%%%n", percent); }
    @Override public void segment(Transcript.Segment s) { System.out.println(s.text()); }
});
```

The last word of a streamed segment may end a little later in the returned transcript, which can time it against the
next segment. `cancel()`, from another thread (or from the listener), stops a call, which then throws
`CancellationException`. An exception thrown by the listener stops the call too, and comes back as the cause.

## Speed

On the Radeon 8060S with Vulkan, large-v3-turbo transcribes 55 s of narration (107 words) in about 1.8 s with a beam
of 5 (31× real time, 60 words a second), and in about 1.1 s greedy (52×, 100 words a second). The model loads in
under half a second once it is in the page cache.

## Try it

```sh
./gradlew transcribe --args='talk.wav'                                 # writes talk.words.tsv and talk.srt
./gradlew transcribe --args='talk.wav --language de --translate'
./gradlew transcribe --args='talk.wav --beam 1 --prompt "Kiln, ggml"' -Pmodel=base.en
./gradlew transcribe --args='talk.wav' -Pmodel=/path/ggml-model.bin       # any ggml Whisper file, timed without DTW
```
