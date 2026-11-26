# Text to speech: what runs, and why kiln-tts looks the way it does

Research for [issue #12](https://github.com/liberation-data/kiln/issues/12), against llama.cpp at the commit pinned in
`native/engines.properties` (22b99122, 2026-11-24).

## The short version

The issue assumed llama.cpp's TTS meant OuteTTS plus the WavTokenizer decoder, and that Kokoro had no ggml runtime.
Both are out of date. llama.cpp's TTS went model-agnostic (PR #26254): `tools/tts` is now a thin loop over libmtmd's
audio generation (`mtmd_helper_gen_audio_*`), and the model's audio half lives in an **mmproj GGUF** next to the
backbone GGUF, like vision models. The same API runs Kokoro, ZONOS3, LFM3-Audio, Chatterbox, Soprano, tiny-voca,
VoxCPM2, Qwen3-TTS, Fish S2 Pro, Drama 3 and (for songs) YuE2 and MiniMax Music 3. Predefined voices ship inside the
mmproj and are listed with `mtmd_gen_audio_voice_name()`.

So `kiln-tts` is one shim (`kiln-tts/native/kiln_tts.cpp`) over that API, and every model above comes for free. It
follows `tools/tts/tts.cpp` step for step. At the same seed its output matches `llama-tts` to within 16-bit rounding
(RMS difference about 9e-6) for Kokoro, ZONOS3 and LFM3-Audio.

## Candidates

| Model | Runs in llama.cpp | Built-in voices | Licence (weights / voices) | Notes |
|---|---|---|---|---|
| **Kokoro-82M** (hexgrad) | yes, `ggml-org/Kokoro-82M-GGUF` | 54: English 20 US (11 F, 9 M) and 8 UK (4 F, 4 M), plus es, fr, hi, it, ja, pt-BR, zh. Graded A to F by the author (`VOICES.md`): af_heart A, af_bella A-, af_nicole and bf_emma B-, the best male voices C+ (am_michael, am_fenrir, am_puck) | Apache-2.0 / Apache-2.0 (the voices are in the repo) | The best-known small open TTS. Not autoregressive: fast, deterministic apart from the vocoder noise. English G2P is built into the GGUF's `tts` template; there is no espeak |
| **ZONOS3-Mini** (Zyphra) | yes, `ggml-org/ZONOS3-Mini-GGUF` | 27. English: Savannah (US F), Wyatt (US M), Imogen (UK F), Rupert (UK M), Cillian (Irish M), Thaddeus (announcer M), Jett (energetic M), Willow (ASMR F), Morgana (witch F), Sakura (anime F); 17 more in other languages | Apache-2.0 / Apache-2.0 (the default voices ship in the release) | Released Nov 2026: 800M MoE (100M active), 44.1 kHz. Zyphra leads its own speaker-similarity and WER benchmark. The only British and Irish *male* voices with good training data |
| **LFM3-Audio-2B** (Liquid AI) | yes, `LiquidAI/LFM3-Audio-2B-GGUF` | 2: `female` (default), `male`; US English | LFM Open License v1.0: free below US$10M annual revenue, commercial use above that needs a licence | An audio chat model that also does TTS and ASR. 24 kHz |
| **Chatterbox Turbo** (Resemble AI) | yes, `ggml-org/chatterbox-turbo-GGUF` | 1: `default` (US female, from `conds.pt`) | MIT / MIT | The reference Python adds a Perth watermark; llama.cpp does not |
| **Soprano-1.1-80M** | yes, `ggml-org/Soprano-1.1-80M-GGUF` | 1, fixed (US female) | Apache-2.0 | Very fast, 32 kHz |
| **tiny-voca** (4.5M) | yes, `ggml-org/tiny-voca-GGUF` | 2: `M2`, `F2` | Apache-2.0 | A toy for its size; robotic |
| **VoxCPM2** (OpenBMB, 2B) | yes, `ggml-org/VoxCPM2-GGUF` | none; *voice design*: a description in parentheses before the text, e.g. `(A calm male narrator)` | Apache-2.0 | 48 kHz. The only way to ask for an **Australian** accent without cloning a real voice. Slow here: about 0.8x real time on the 8060S |
| Qwen3-TTS (Base) | yes, `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF` | none in llama.cpp: it needs a speaker reference clip (cloning), and the CustomVoice checkpoints with named speakers aren't supported | Apache-2.0 | Not for built-in voices, but Kiln knows it (`QWEN3_TTS`) for cloning a reference clip with its transcript (see Voices below) |
| Fish S2 Pro, Drama 3 | yes | speaker *tags* (`<\|speaker:0\|>`...) give arbitrary voices that aren't fixed between runs; cloning otherwise | S2 Pro: Fish Audio Research License (non-commercial); Drama 3: Fish Audio Community License | Excluded: no stable built-in voice, non-commercial, and Drama 3 is 35B |
| Pocket TTS (Kyutai) | the mtmd pipeline exists, but no llama.cpp-format GGUF with voices is published (Serveurperso's GGUFs are for pockettts.cpp) | 27 (alba, marius, javert, ...; some named after real readers) | CC-BY-4.0 | Would need converting with llama.cpp's Python converter. Skipped |
| OuteTTS 0.2/0.3 | **no longer**. libllama still has the WavTokenizer decoder architecture, but no pipeline drives OuteTTS: the old prompt building, speaker JSON and codes-to-audio loop were removed from `tools/tts` | speaker JSON profiles (en_male_1, en_female_1, ...) | 0.2-500M CC-BY-NC-4.0, 0.3-500M CC-BY-SA-4.0, 0.3-1B and 1.0-1B CC-BY-NC-SA-4.0 | Bringing it back would mean porting the removed loop, for a model that is a generation behind Kokoro and ZONOS3 |
| Orpheus 3B | no: it needs a SNAC decoder, and llama.cpp has none | 8 (tara, leah, jess, leo, dan, mia, zac, zoe), US English | Apache-2.0 on Hugging Face (fine-tuned from Llama 3.2) | Expressive (laughs, sighs), but needs a new decoder engine, as YuE2 has its codec |

**Australian voices:** there are no Australian built-in voices in any of these. Kokoro is US and UK only, ZONOS3
English is US, UK and Irish, LFM3 and the rest are US. The only route without cloning a real person is VoxCPM2's
voice design, by asking for an Australian accent in the description. Whether the result really sounds Australian
has to be checked by ear.

## Choice

The engine is **llama.cpp's mtmd TTS**, through one shim, so the choice of model stays open. It reuses the llama.cpp
checkout that `kiln-llama` already pins, adds no new engine to fetch, and supports streaming (`inp.stream`, as
llama-server does).

For narration, **Kokoro** and **ZONOS3** are the strongest candidates with built-in voices: Kokoro for its graded
voices and speed (about 17x real time here), ZONOS3 for its British, Irish and announcer male voices at 44.1 kHz
(about 8x real time). VoxCPM2 is there for the Australian accent. Which voice to use is the user's call, from the
auditions in `videos/explainer/voices/`.

## Implementation notes

- **Thread counts.** `common_init_from_params()` expects `common_params_parse()` to have run. The parser fills in
  `cpuparams.n_threads` from the hardware afterwards (`postprocess_cpu_params`). Without that, the threadpool is
  sized from -1 and ggml tries to allocate 2^64 bytes. The shim calls `postprocess_cpu_params` itself.
- **llama-tts's defaults.** `LLAMA_EXAMPLE_TTS` sets `penalty_repeat = 1.05` and `penalty_last_n = -1` for the
  backbone sampler. The shim does the same, so its output matches `llama-tts`'s.
- **Voices.** If a model has predefined voices, `mtmd_gen_audio_get_voice(name)` returns a voice bitmap that goes in
  as `speaker_ref`. The voice name is also passed to the GGUF's `tts` chat template (`common_chat_get_tts_prompt`),
  which LFM3-Audio uses.
- **Reference clips.** A voice of `@<file>` loads that WAV or MP3 with `mtmd_helper_bitmap_init_from_file()` and passes
  it as `speaker_ref` instead (VoxCPM2 and Qwen3-TTS clone it). If `<file>.txt` exists, its text, trailing whitespace
  trimmed, goes in as `speaker_ref_text`, as `llama-tts --tts-speaker-text` does: with the transcript the model
  continues that speaker rather than only imitating the sound, which keeps an accent much steadier.
- **Streaming.** With `inp.stream`, `get_output()` returns only the audio since the last call, after each
  `step_gen()`, and `flush()` is called once at the end. Without a callback, the shim runs non-streaming, exactly as
  `llama-tts` does.
- **The build** shares `engines-src/llama.cpp/build-<backend>` with `kiln-llama` and adds the `mtmd` target
  (`-DLLAMA_BUILD_MTMD=ON`). `libkiln_tts.so` is self-contained like the others: ggml, llama and mtmd are static,
  and only `kiln_*` is exported.
- **Checking intelligibility without listening.** `llama-mtmd-cli` with `ggml-org/Qwen3-ASR-1.7B-GGUF` transcribes
  a WAV; this is how the auditions were checked for word errors.
