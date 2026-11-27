---
title: Songs
description: Generate songs with vocals and lyrics from Java with ACE-Step 1.5 (acestep.cpp) and YuE2 (yue2.cpp).
---

Fornax has two song models, each through its own engine: **ACE-Step 1.5** (`fornax-acestep`, through acestep.cpp) and
**YuE2** (`fornax-yue2`, through yue2.cpp). Both take the engine's own JSON request and return 48 kHz stereo `Audio`.

The request formats are the engines' own, so their READMEs are the full reference:
[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp) and [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp).

## ACE-Step 1.5

ACE-Step is four models working together: a 5 Hz language model that plans the song (metadata, lyrics and audio
codes), a text encoder, a DiT that generates the audio, and a VAE that decodes it.

<!-- snippet: Songs#acestep -->
```java
import org.liberationdata.fornax.acestep.AceStepModel;

import java.io.File;

String request = """
        {"caption": "modern country rock, male baritone vocal with a southern twang, telecaster, driving drums",
         "lyrics": "[Verse 1]\\nRusty rolled in on a Tuesday night\\n\\n[Chorus]\\nHe's a machine with a honky-tonk heart\\n",
         "bpm": 96, "duration": 60, "keyscale": "G major", "timesignature": "4", "vocal_language": "en",
         "seed": 11, "lm_seed": 11, "inference_steps": 8, "guidance_scale": 1.0, "shift": 3.0}""";

try (var songs = new AceStepModel("ace-step")) {     // the four files, downloaded on first use
    AceStepModel.Result r = songs.generate(request);
    r.audio().writeWav(new File("song.wav"));
    System.out.println(r.resolvedRequestJson());      // the request after the LM filled in what it decided
}
```

`"ace-step"` is a catalogue set: turbo with the 4B LM. `"ace-step-turbo"` is the DiT without the LM, and
`"ace-step-xl"` and `"ace-step-xl-turbo"` the same with the XL DiT. `new AceStepModel(name, false)` sets
`keepLoaded` (below). To fetch a set ahead of time, use `fornax pull ace-step`, or `hf download`:

```sh
hf download Serveurperso/ACE-Step-1.5-GGUF acestep-5Hz-lm-4B-Q8_0.gguf Qwen3-Embedding-0.6B-Q8_0.gguf \
    acestep-v15-turbo-Q8_0.gguf vae-BF16.gguf
```

To pick each file yourself, pass the four, as paths or [model references](/docs/guides/models/#by-reference):

<!-- snippet: Songs#acestep-files -->
```java
String repo = "Serveurperso/ACE-Step-1.5-GGUF";
try (var songs = new AceStepModel(repo + ":acestep-5Hz-lm-1.7B-Q8_0.gguf", repo + ":Qwen3-Embedding-0.6B-Q8_0.gguf",
        repo + ":acestep-v15-turbo-Q8_0.gguf", repo + ":vae-BF16.gguf", true)) {   // lm, text encoder, DiT, VAE
    // ...
}
```

`resolvedRequestJson` is the request as the engine ran it, after the language model filled in metadata, lyrics and
seeds. Save it beside the song to make it again.

### Choices

- **Without the language model.** Use `"ace-step-turbo"`, or pass `null` as the first file. The caption and lyrics
  then drive the DiT directly, which is faster and uses less memory; `./gradlew single` works this way.
- **A smaller or larger planner.** The LM comes in 4B, 1.7B and 0.6B (`acestep-5Hz-lm-1.7B-Q8_0.gguf`, ...).
- **Another DiT.** `acestep-v15-turbo` takes 8 steps. The XL turbo DiT (`acestep-v15-xl-turbo-Q8_0.gguf`) is
  larger; `sft` takes about 50 steps.
- **`keepLoaded`.** `true` keeps every part resident between calls. `false` keeps at most one part on the GPU at a
  time, loading each as it is needed: slower, but it leaves memory for other models in the same JVM.

On the Radeon 8060S with Vulkan, the turbo DiT made a 60 s song in about 15 s, and XL turbo a 3 min 20 s song in
57 s.

If decoding hangs or loses the GPU on a power-limited machine (a laptop on battery, say), see
[`FORNAX_ACESTEP_VAE_CHUNK`](/docs/troubleshooting/#ace-step-on-a-power-limited-gpu).

## YuE2

YuE2 writes an ABC-notation score first, a plan of the song in plain text, then the audio from it. The score comes
back with the song, and yue2.cpp can take an edited one back in its request (as `abc`).

:::caution[Non-commercial weights]
YuE2's weights are CC BY-NC 4.0: no commercial use of what it makes.
:::

<!-- snippet: Songs#yue2 -->
```java
import org.liberationdata.fornax.yue2.Yue2Model;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;

String request = """
        {"style": "modern country rock, male baritone lead vocal, overdriven electric guitars, mid-tempo stomp",
         "lyrics": "[Verse 1]\\nRusty rolled in on a Tuesday night\\n\\n[Chorus]\\nHe's a machine with a honky-tonk heart\\n",
         "duration": 60, "seed": 1, "lm_seed": 1}""";

try (var yue = new Yue2Model("yue2", true, 0)) {     // the model and its VAE, downloaded on first use
    Yue2Model.Result r = yue.generate(request);
    r.audio().writeWav(new File("yue2-song.wav"));
    if (r.abcScore() != null) Files.writeString(Path.of("yue2-song.abc"), r.abcScore());
}
```

`new Yue2Model(modelPath, vaePath, keepLoaded, maxSeq)` takes the two files instead, as paths or
[model references](/docs/guides/models/#by-reference). To fetch them ahead of time:

```sh
hf download Serveurperso/YuE2-GGUF YuE2-3B-Q8_0.gguf YuE2-Vae-F32.gguf
```

The last two constructor arguments are `keepLoaded`, as for ACE-Step, and `maxSeq`, the model's context in tokens
(0 is its full 24,576). The result has the audio, the resolved request and the ABC score. On the 8060S, YuE2 3B
(Q8_0) made a 60 s song in about 60 s.

## Try it

```sh
./gradlew song --args='song.wav'                      # ACE-Step, with the 4B LM and the turbo DiT
./gradlew song --args='song.wav my-request.json' -Pdit=xl-turbo -Plm=1.7B
./gradlew song --args='song.wav' -Plm=none            # without the LM
./gradlew yue2 --args='yue2-song.wav'
./gradlew single                                      # an LLM writes the lyrics, ACE-Step sings them
```

The examples' requests are in `fornax-examples/src/main/resources/requests/`. Each writes the resolved request next to
the `.wav`, and the YuE2 example writes the score too.
