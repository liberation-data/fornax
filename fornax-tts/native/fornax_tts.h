// Small, stable C ABI over llama.cpp's text-to-speech pipeline (libmtmd's audio generation) for Fornax (Java FFM).
// One shim runs every TTS model llama.cpp supports (Kokoro, ZONOS3, LFM3-Audio, tiny-voca, Chatterbox, ...):
// a backbone GGUF plus its mmproj GGUF, which holds the audio decoder and the model's predefined voices.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct fornax_tts fornax_tts;

// Called with each new piece of audio while streaming: n_frames frames of n_channels interleaved float PCM.
// Return non-zero to stop generation (the call then returns the audio produced so far).
typedef int (*fornax_tts_audio_cb)(const float * pcm, int64_t n_frames, int32_t n_channels, void * user);

const char * fornax_tts_last_error(void);

// n_gpu_layers: -1 = all. n_ctx: 0 = llama.cpp's default for the model.
fornax_tts * fornax_tts_create(const char * model_path, const char * mmproj_path, int n_gpu_layers, int n_ctx);

// Output format and the number of predefined voices (0 when the model has none, e.g. LFM3-Audio names its voices
// in the prompt: "male" / "female").
void fornax_tts_info(fornax_tts * h, int32_t * sample_rate, int32_t * n_channels, int32_t * n_voices);

// Name of the idx-th predefined voice, NULL when out of range. Voice 0 is the model's default.
const char * fornax_tts_voice_name(fornax_tts * h, int32_t idx);

// Speaks text. voice: a predefined voice name, a model-specific voice word, "@<file>" for a WAV or MP3 reference clip
// to speak like (with its transcript in "<file>.txt" if that exists), or NULL/"" for the model's default.
// lang: NULL/"" for the model's default. temperature < 0 = the model's default. seed 0xFFFFFFFF = random.
// speed <= 0 = the model's own pace (only some models take another value). max_frames <= 0 = the model's limit.
// cb (optional) streams the audio as it is decoded. *out_pcm (interleaved float, malloc'd, free it with
// fornax_tts_free) receives all of it either way.
int fornax_tts_speak(fornax_tts * h, const char * text, const char * voice, const char * lang, float temperature,
                   uint32_t seed, float speed, int32_t max_frames, fornax_tts_audio_cb cb, void * user,
                   float ** out_pcm, int64_t * out_n_frames, int32_t * out_n_generated);

void fornax_tts_free(void * p);
void fornax_tts_destroy(fornax_tts * h);

#ifdef __cplusplus
}
#endif
