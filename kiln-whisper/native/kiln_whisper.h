// Small, stable C ABI over whisper.cpp for Kiln (Java FFM). whisper.h passes its two parameter structs by value
// (whisper_full_params is ~70 fields and changes between releases), so this layer takes plain arguments instead and
// leaves whisper's own symbols private to the library. Results are read back segment by segment, token by token.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct kiln_whisper kiln_whisper;

// Progress of a transcription, 0..100.
typedef void (*kiln_whisper_progress_cb)(int32_t percent, void * user);
// Segments [first, first + n) are final and can be read with the accessors below (during the callback, too).
typedef void (*kiln_whisper_segment_cb)(int32_t first, int32_t n, void * user);

const char * kiln_whisper_last_error(void);

// use_gpu: 0 runs on the CPU. flash_attn: 1 for flash attention (ignored when DTW is on: whisper.cpp can't do both).
// dtw_preset: a whisper_alignment_heads_preset (WHISPER_AHEADS_*), 0 for no DTW token timestamps.
kiln_whisper * kiln_whisper_create(const char * model_path, int32_t use_gpu, int32_t gpu_device, int32_t flash_attn,
                                   int32_t dtw_preset);

int32_t kiln_whisper_is_multilingual(kiln_whisper * h);
// "tiny", "base", "small", "medium", "large".
const char * kiln_whisper_model_type(kiln_whisper * h);

// Transcribes 16 kHz mono float PCM. language: a code ("en", "de", ...) or NULL/""/"auto" to detect it (an English-only
// model always transcribes English). translate: 1 translates to English. initial_prompt: NULL or text that conditions
// the first window (spelling, names, style). n_threads <= 0: whisper.cpp's default. beam_size <= 1: greedy decoding.
// temperature < 0: whisper.cpp's default (0, raised on fallback). Returns 0 on success.
int32_t kiln_whisper_transcribe(kiln_whisper * h, const float * pcm, int32_t n_samples, const char * language,
                                int32_t translate, const char * initial_prompt, int32_t n_threads, int32_t beam_size,
                                float temperature, kiln_whisper_progress_cb progress, kiln_whisper_segment_cb segment,
                                void * user);

// Stops the transcription in progress (from any thread); it then returns -100.
void kiln_whisper_cancel(kiln_whisper * h);

// The language of the last transcription (detected or given), e.g. "en".
const char * kiln_whisper_language(kiln_whisper * h);

int32_t kiln_whisper_n_segments(kiln_whisper * h);
// Segment times are in centiseconds (10 ms).
int64_t kiln_whisper_segment_t0(kiln_whisper * h, int32_t segment);
int64_t kiln_whisper_segment_t1(kiln_whisper * h, int32_t segment);
const char * kiln_whisper_segment_text(kiln_whisper * h, int32_t segment);
int32_t kiln_whisper_n_tokens(kiln_whisper * h, int32_t segment);

// One token: its text (raw bytes; a multi-byte character can be split across tokens), and through the out pointers
// whether it is a special token (timestamps, end of text, ...), its probability, its start and end from whisper's
// timestamp tokens, and its DTW time (where the alignment enters the token), all in centiseconds. t_dtw is -1
// without DTW.
const char * kiln_whisper_token(kiln_whisper * h, int32_t segment, int32_t token, int32_t * special, float * p,
                                int64_t * t0, int64_t * t1, int64_t * t_dtw);

void kiln_whisper_destroy(kiln_whisper * h);

#ifdef __cplusplus
}
#endif
