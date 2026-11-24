#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef struct kiln_yue2 kiln_yue2;
const char * kiln_yue2_last_error(void);
/* keep_loaded != 0 keeps both backbone halves + VAE resident (EVICT_NEVER). max_seq 0 = full 24576 context. */
kiln_yue2 * kiln_yue2_create(const char * model_path, const char * vae_path, int keep_loaded, int max_seq);
/* request_json: yue2.cpp request schema. Returns 0 and a malloc'd planar stereo f32 buffer at 48 kHz,
   plus (optional) the replay request JSON (score, semantic tokens, seeds) and ABC score. Free with kiln_yue2_free. */
int kiln_yue2_generate(kiln_yue2 * h, const char * request_json, float ** out_planar, int * out_n_samples,
                    char ** out_replay_json, char ** out_score);
void kiln_yue2_free(void * p);
void kiln_yue2_destroy(kiln_yue2 * h);
#ifdef __cplusplus
}
#endif
