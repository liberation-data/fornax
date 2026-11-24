#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct kiln_acestep kiln_acestep;

/* Thread-local error text for the last failing call. */
const char * kiln_acestep_last_error(void);

/* lm_path may be NULL/"" to skip the LM stage (then requests need lyrics/caption only, no codes).
   keep_loaded != 0 keeps all modules resident between calls (EVICT_NEVER). Returns NULL on failure. */
kiln_acestep * kiln_acestep_create(const char * lm_path, const char * text_enc_path, const char * dit_path,
                     const char * vae_path, int keep_loaded);

/* request_json: acestep.cpp request schema. On success returns 0 and sets *out_planar to
   malloc'd planar stereo float32 [L0..LN-1, R0..RN-1] at 48 kHz (free with kiln_acestep_free) and
   *out_request_json (optional) to the LM-enriched request. */
int kiln_acestep_generate(kiln_acestep * h, const char * request_json, float ** out_planar, int * out_n_samples,
                   char ** out_request_json);

void kiln_acestep_free(void * p);
void kiln_acestep_destroy(kiln_acestep * h);

#ifdef __cplusplus
}
#endif
