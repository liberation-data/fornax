// Small, stable C ABI over whisper.cpp for Fornax (Java FFM). whisper.h passes its two parameter structs by value
// (whisper_full_params is ~70 fields and changes between releases), so this layer takes plain arguments and a small
// struct of its own instead, and leaves whisper's own symbols private to the library. Results are read back segment by segment, token by token.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct fornax_whisper fornax_whisper;

// Progress of a transcription, 0..100.
typedef void (*fornax_whisper_progress_cb)(int32_t percent, void * user);
// Segments [first, first + n) are final and can be read with the accessors below (during the callback, too).
typedef void (*fornax_whisper_segment_cb)(int32_t first, int32_t n, void * user);

const char * fornax_whisper_last_error(void);

// use_gpu: 0 runs on the CPU. flash_attn: 1 for flash attention (ignored when DTW is on: whisper.cpp can't do both).
// dtw_preset: a whisper_alignment_heads_preset (WHISPER_AHEADS_*), 0 for no DTW token timestamps.
fornax_whisper * fornax_whisper_create(const char * model_path, int32_t use_gpu, int32_t gpu_device, int32_t flash_attn,
                                   int32_t dtw_preset);

int32_t fornax_whisper_is_multilingual(fornax_whisper * h);
// "tiny", "base", "small", "medium", "large".
const char * fornax_whisper_model_type(fornax_whisper * h);

// How to transcribe: Fornax's own struct, passed by pointer and copied field by field into whisper_full_params. Its
// layout is checked from Java through fornax_whisper_params_layout.
typedef struct fornax_whisper_params {
    const char * language;        // a code ("en", "de", ...) or NULL/""/"auto" to detect it (an English-only model
                                  // always transcribes English)
    const char * initial_prompt;  // NULL or text that conditions the first window (spelling, names, style)
    const char * vad_model_path;  // a Silero VAD ggml model: transcribe only the speech it finds. NULL: no VAD
    int32_t translate;            // 1 translates to English
    int32_t n_threads;            // <= 0: whisper.cpp's default
    int32_t beam_size;            // <= 1: greedy decoding
    float   temperature;          // < 0: whisper.cpp's default (0)
    float   temperature_inc;      // fallback: retry a window that fails a check this much hotter (0: never retry)
    float   entropy_thold;        // a window whose token entropy is below this (repetitive text) fails
    float   logprob_thold;        // a window whose average token log-probability is below this fails
    float   no_speech_thold;      // a window more likely than this to be silence, that also fails the log-prob
                                  // check, is skipped
    int32_t max_text_ctx;         // tokens of the text before each window to prompt it with; < 0: whisper.cpp's
                                  // default (as many as fit), 0: none
    int32_t suppress_nst;         // 1: suppress non-speech tokens ([music], notes, ...)
    // whisper_vad_params, used with vad_model_path
    float   vad_threshold;        // speech probability above which a frame is speech
    int32_t vad_min_speech_ms;    // shorter speech is dropped
    int32_t vad_min_silence_ms;   // silence this long ends a speech segment
    float   vad_max_speech_s;     // longer speech is split
    int32_t vad_speech_pad_ms;    // padding either side of each speech segment
    float   vad_samples_overlap;  // seconds of audio carried past the end of each segment
} fornax_whisper_params;

// The size of fornax_whisper_params, and through offsets (up to n entries, in declaration order) each field's offset.
// Returns the number of fields.
int32_t fornax_whisper_params_layout(int64_t * size, int64_t * offsets, int32_t n);

// Transcribes 16 kHz mono float PCM. Returns 0 on success, -101 for an unknown language.
int32_t fornax_whisper_transcribe(fornax_whisper * h, const float * pcm, int32_t n_samples, const fornax_whisper_params * params,
                                fornax_whisper_progress_cb progress, fornax_whisper_segment_cb segment, void * user);

// Stops the transcription in progress (from any thread); it then returns -100.
void fornax_whisper_cancel(fornax_whisper * h);

// The language of the last transcription (detected or given), e.g. "en".
const char * fornax_whisper_language(fornax_whisper * h);

int32_t fornax_whisper_n_segments(fornax_whisper * h);
// Segment times are in centiseconds (10 ms).
int64_t fornax_whisper_segment_t0(fornax_whisper * h, int32_t segment);
int64_t fornax_whisper_segment_t1(fornax_whisper * h, int32_t segment);
const char * fornax_whisper_segment_text(fornax_whisper * h, int32_t segment);
int32_t fornax_whisper_n_tokens(fornax_whisper * h, int32_t segment);

// The speech segments VAD found in the last transcription, in centiseconds on the original audio (none without VAD).
// Segment and token times (t0, t1) are already on that timeline; DTW times are not (see fornax_whisper_token).
int32_t fornax_whisper_n_vad_segments(fornax_whisper * h);
int64_t fornax_whisper_vad_segment_t0(fornax_whisper * h, int32_t segment);
int64_t fornax_whisper_vad_segment_t1(fornax_whisper * h, int32_t segment);

// One token: its text (raw bytes; a multi-byte character can be split across tokens), and through the out pointers
// whether it is a special token (timestamps, end of text, ...), its probability, its start and end from whisper's
// timestamp tokens, and its DTW time (where the alignment enters the token), all in centiseconds. t_dtw is -1
// without DTW. With VAD, t_dtw is on the audio whisper heard (the speech segments joined by 0.1 s of silence), which
// whisper.cpp doesn't map back: the caller maps it from the VAD segments.
const char * fornax_whisper_token(fornax_whisper * h, int32_t segment, int32_t token, int32_t * special, float * p,
                                int64_t * t0, int64_t * t1, int64_t * t_dtw);

void fornax_whisper_destroy(fornax_whisper * h);

#ifdef __cplusplus
}
#endif
