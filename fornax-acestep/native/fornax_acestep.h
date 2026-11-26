#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct fornax_acestep fornax_acestep;

/* Thread-local error text for the last failing call. */
const char * fornax_acestep_last_error(void);

/* lm_path may be NULL/"" to skip the LM stage (then requests need lyrics/caption only, no codes).
   keep_loaded != 0 keeps all modules resident between calls (EVICT_NEVER). Returns NULL on failure. */
fornax_acestep * fornax_acestep_create(const char * lm_path, const char * text_enc_path, const char * dit_path,
                     const char * vae_path, int keep_loaded);

/* request_json: acestep.cpp request schema. On success returns 0 and sets *out_planar to
   malloc'd planar stereo float32 [L0..LN-1, R0..RN-1] at 48 kHz (free with fornax_acestep_free) and
   *out_request_json (optional) to the LM-enriched request. Returns FORNAX_ACESTEP_CANCELLED if stopped. */
int fornax_acestep_generate(fornax_acestep * h, const char * request_json, float ** out_planar, int * out_n_samples,
                   char ** out_request_json);

#define FORNAX_ACESTEP_CANCELLED 5

/* cancel != 0 asks the generate call in progress to stop (thread-safe): acestep.cpp polls it between LM tokens,
   DiT steps and VAE tiles. It stays set until cleared with cancel = 0, which the caller does before each call. */
void fornax_acestep_set_cancel(fornax_acestep * h, int cancel);

void fornax_acestep_free(void * p);
void fornax_acestep_destroy(fornax_acestep * h);

#ifdef __cplusplus
}
#endif
