#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef struct fornax_yue2 fornax_yue2;
const char * fornax_yue2_last_error(void);
/* keep_loaded != 0 keeps both backbone halves + VAE resident (EVICT_NEVER). max_seq 0 = full 24576 context. */
fornax_yue2 * fornax_yue2_create(const char * model_path, const char * vae_path, int keep_loaded, int max_seq);
/* request_json: yue2.cpp request schema. Returns 0 and a malloc'd planar stereo f32 buffer at 48 kHz,
   plus (optional) the replay request JSON (score, semantic tokens, seeds) and ABC score. Free with fornax_yue2_free.
   Returns FORNAX_YUE2_CANCELLED if stopped. */
int fornax_yue2_generate(fornax_yue2 * h, const char * request_json, float ** out_planar, int * out_n_samples,
                    char ** out_replay_json, char ** out_score);
#define FORNAX_YUE2_CANCELLED 3
/* cancel != 0 asks the generate call in progress to stop (thread-safe): yue2.cpp polls it between LM tokens, NAR
   steps and VAE tiles. It stays set until cleared with cancel = 0, which the caller does before each call. */
void fornax_yue2_set_cancel(fornax_yue2 * h, int cancel);
void fornax_yue2_free(void * p);
void fornax_yue2_destroy(fornax_yue2 * h);
#ifdef __cplusplus
}
#endif
