// Small, stable C ABI over llama.cpp for Kiln (Java FFM). llama.cpp's own API passes param structs by
// value and is mid-migration (llama_batch -> llama_batch_ext), so Java binds to this instead.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct kiln_llama kiln_llama;

// Called for each generated piece of text. Return non-zero to stop generation.
typedef int (*kiln_llama_token_cb)(const char * piece, void * user);

const char * kiln_llama_last_error(void);

// n_gpu_layers: -1 = all. n_ctx: 0 = model default.
kiln_llama * kiln_llama_create(const char * model_path, int n_gpu_layers, int n_ctx);

// Formats messages with the model's chat template, then generates. Each call starts from a fresh context.
// roles[i] is "system", "user" or "assistant". *out_text is malloc'd; free it with kiln_llama_free.
// seed: 0xFFFFFFFF = random. temperature <= 0 = greedy.
int kiln_llama_chat(kiln_llama * h, const char ** roles, const char ** contents, int n_messages,
                    int max_tokens, float temperature, float top_p, float min_p, uint32_t seed,
                    kiln_llama_token_cb cb, void * user, char ** out_text, int * out_n_prompt, int * out_n_generated);

// Raw completion of an already-formatted prompt.
int kiln_llama_complete(kiln_llama * h, const char * prompt, int max_tokens, float temperature, float top_p,
                        float min_p, uint32_t seed, kiln_llama_token_cb cb, void * user, char ** out_text,
                        int * out_n_prompt, int * out_n_generated);

void kiln_llama_free(void * p);
void kiln_llama_destroy(kiln_llama * h);

#ifdef __cplusplus
}
#endif
