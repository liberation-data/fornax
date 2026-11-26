// Small, stable C ABI over stems.cpp for Kiln (Java FFM). stems.cpp's own C ABI (libstems_v1.h) is a function table
// of size-tagged structs passed by pointer; this layer takes plain arguments instead, keeps a cancel flag the Java side
// can set from any thread, and leaves stems.cpp's and ggml's symbols private to the library.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct kiln_demucs kiln_demucs;

// Called between segments (about 7.8 s of audio each for HTDemucs): done of total segment passes.
typedef void (*kiln_demucs_progress_cb)(int32_t done, int32_t total, void * user);

// Status codes: libstems' (STEMS_STATUS_*_V1).
#define KILN_DEMUCS_OK 0
#define KILN_DEMUCS_INVALID_ARGUMENT 1
#define KILN_DEMUCS_CANCELLED 3

const char * kiln_demucs_last_error(void);

// Loads a stems.cpp GGUF (htdemucs, htdemucs_6s, htdemucs_ft, ...). device: NULL or "" for the best GPU (falling back
// to the CPU), "cpu" for the CPU. cpu_threads <= 0: one per physical core.
kiln_demucs * kiln_demucs_create(const char * model_path, const char * device, int32_t cpu_threads);

const char * kiln_demucs_name(kiln_demucs * h);          // "htdemucs", "htdemucs_6s", ...
const char * kiln_demucs_backend(kiln_demucs * h);       // the ggml backend, e.g. "Vulkan0" or "CPU"
int32_t kiln_demucs_n_sources(kiln_demucs * h);
const char * kiln_demucs_source(kiln_demucs * h, int32_t i);   // "drums", "bass", "other", "vocals", ...
int32_t kiln_demucs_sample_rate(kiln_demucs * h);        // the model's own rate (44100)
int32_t kiln_demucs_channels(kiln_demucs * h);           // channels of every stem (2)

// Separates planar float PCM (pcm[channel * n_samples + i]) at any rate and channel count. Stems come back at the
// input's rate and length, in *out: planar, out[(source * channels + channel) * n_samples + i], freed with
// kiln_demucs_free. shifts: random time shifts averaged (0 for one deterministic pass). overlap < 0: the model's own
// (0.25 for HTDemucs). progress may be NULL. Returns KILN_DEMUCS_OK, KILN_DEMUCS_CANCELLED, or another libstems status.
int32_t kiln_demucs_separate(kiln_demucs * h, const float * pcm, int64_t n_samples, int32_t n_channels,
                             int32_t sample_rate, int32_t shifts, float overlap, uint32_t seed,
                             kiln_demucs_progress_cb progress, void * user, float ** out);

// on = 1 asks the separation in progress to stop before its next segment (from any thread); on = 0 clears it.
void kiln_demucs_set_cancel(kiln_demucs * h, int32_t on);

void kiln_demucs_free(float * p);
void kiln_demucs_destroy(kiln_demucs * h);

#ifdef __cplusplus
}
#endif
