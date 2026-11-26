---
title: Stem separation
description: Split a song into vocals and backing, or drums, bass, other and vocals, from Java with Meta's HTDemucs through stems.cpp, on the GPU.
---

`fornax-demucs` splits music into its stems through [stems.cpp](https://github.com/betweentwomidnights/stems.cpp), a
ggml port of Meta's [HTDemucs](https://github.com/facebookresearch/demucs) v4. A vocal stem drives lip-sync and word
timing; the backing is a karaoke track or a bed for a new vocal. The class is `DemucsModel`.

## Split a song

```sh
hf download thepatch/htdemucs-GGUF htdemucs-42M-v1.0-F32.gguf
```

```java
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.demucs.DemucsModel;
import org.liberationdata.fornax.demucs.Stems;

import java.io.File;

try (var demucs = new DemucsModel(DemucsModel.Weights.HTDEMUCS)) {
    Stems s = demucs.separate(Audio.readWav(new File("song.wav")));
    s.vocals().writeWav(new File("song-vocals.wav"));
    s.backing().writeWav(new File("song-backing.wav"));
    Audio drums = s.get("drums");
}
```

Any `Audio` works. The model runs at 44.1 kHz stereo; Fornax resamples to it and back, so every stem has the input's
sample rate and length. `Audio.readWav` reads 8, 16, 24 and 32-bit PCM and float WAV files. For MP3 and other formats,
convert first (`ffmpeg -i song.mp3 song.wav`).

`Stems` has:

- `names()`: the stems in the model's order, `drums`, `bass`, `other`, `vocals` (then `guitar`, `piano` for the
  6-stem model);
- `get(name)`, `vocals()`, and `asMap()` for all of them;
- `backing()`: every stem but the vocals, added together (what demucs calls `no_vocals`);
- `mix()`: every stem added together, which comes back to the input to within the model's error (about -35 dB).

stems.cpp runs demucs' own separate path, not just the network: it normalises the mix by its mono mean and
deviation, separates overlapping 7.8 s segments, cross-fades them back together and scales the stems back. Its
published weights match PyTorch demucs sample for sample.

## The models Fornax knows by name

`DemucsModel.Weights` names stems.cpp's conversions of Meta's checkpoints, in `thepatch/htdemucs-GGUF` on Hugging
Face. `new DemucsModel(Weights)` finds the file in the Hugging Face cache; if it isn't there, it fails with the
`hf download` command that fetches it.

| `Weights` | File | Size | Stems |
|---|---|---|---|
| `HTDEMUCS` | `htdemucs-42M-v1.0-F32.gguf` | 160 MB | drums, bass, other, vocals: demucs' default model |
| `HTDEMUCS_F16` | `htdemucs-42M-v1.0-F16.gguf` | 100 MB | the same, with F16 weights |
| `HTDEMUCS_6S` | `htdemucs_6s-27M-v1.0-F32.gguf` | 104 MB | drums, bass, other, vocals, guitar, piano |
| `HTDEMUCS_FT` | `htdemucs_ft-4x42M-v1.0-F32.gguf` | 640 MB | four fine-tuned models, one per stem: the best of these, at four times the time |

The demucs code and weights are MIT licensed. Any other stems.cpp GGUF loads by path: `new DemucsModel(path)`.

## Options

```java
var options = DemucsModel.Options.DEFAULT
        .withShifts(2)       // average 2 randomly time-shifted passes (demucs --shifts): a little cleaner, 2x the time
        .withOverlap(0.25f)  // between segments; below 0 (the default) is the model's own, 0.25
        .withSeed(7);        // picks the shifts
```

`new DemucsModel(path, false, 0)` runs on the CPU (the last argument is its threads, 0 for one per physical core).
On a machine with no GPU, that also needs `-Dfornax.allowCpu=true` (or `FORNAX_ALLOW_CPU=true`): otherwise Fornax refuses
to load its GPU build there ([Fornax says it found no GPU](/docs/gpu-drivers/#fornax-says-it-found-no-gpu)).

## Progress, cancel and async

A progress callback hears `done` of `total` segment passes, on the calling thread:

```java
Stems s = demucs.separate(audio, DemucsModel.Options.DEFAULT,
        (done, total) -> System.out.printf("%d/%d%n", done, total));
```

`cancel()`, from another thread or from the callback, stops a separation before its next segment, and the call throws
`CancellationException`. An exception thrown by the callback stops the call too, and is rethrown. `separateAsync`
runs on the model's own thread and returns a `CompletableFuture`; cancelling the future stops the separation. See
[Threading](/docs/guides/threading/).

## Speed and quality

On the Radeon 8060S with Vulkan, `HTDEMUCS` splits a 3 min 9 s song into its four stems in 26 s: 7.3 s of audio per
second. On the same machine's CPU it managed about 2.9 s of audio per second.

On Endless Country's episode 2 song, Fornax's vocals measured 13.5 dB SDR against torchaudio's Hybrid Demucs
(`HDEMUCS_HIGH_MUSDB_PLUS`, the v3 model the episode was made with), and its backing 14.8 dB. The two are different
models, so they don't agree exactly.

## Try it

```sh
./gradlew separate --args='song.wav'                    # writes song-vocals.wav and song-backing.wav
./gradlew separate --args='song.wav --all'              # and song-drums.wav, song-bass.wav, song-other.wav
./gradlew separate --args='song.wav --shifts 2' -Pmodel=htdemucs-ft
./gradlew separate --args='song.wav --cpu'
```
