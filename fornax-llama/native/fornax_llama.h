// Small, stable C ABI over llama.cpp for Fornax (Java FFM). llama.cpp's own API passes param structs by
// value and is mid-migration (llama_batch -> llama_batch_ext), so Java binds to this instead.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct fornax_llama fornax_llama;

// Called for each generated piece of text. Return non-zero to stop generation.
typedef int (*fornax_llama_token_cb)(const char * piece, void * user);

const char * fornax_llama_last_error(void);

// n_gpu_layers: -1 = all. n_ctx: 0 = model default.
fornax_llama * fornax_llama_create(const char * model_path, int n_gpu_layers, int n_ctx);

// Formats messages with the model's chat template, then generates. Each call starts from a fresh context.
// roles[i] is "system", "user" or "assistant". *out_text is malloc'd; free it with fornax_llama_free.
// seed: 0xFFFFFFFF = random. temperature <= 0 = greedy.
int fornax_llama_chat(fornax_llama * h, const char ** roles, const char ** contents, int n_messages,
                    int max_tokens, float temperature, float top_p, float min_p, uint32_t seed,
                    fornax_llama_token_cb cb, void * user, char ** out_text, int * out_n_prompt, int * out_n_generated);

// Raw completion of an already-formatted prompt.
int fornax_llama_complete(fornax_llama * h, const char * prompt, int max_tokens, float temperature, float top_p,
                        float min_p, uint32_t seed, fornax_llama_token_cb cb, void * user, char ** out_text,
                        int * out_n_prompt, int * out_n_generated);

// OpenAI-style chat with tool calling, through llama.cpp's chat templates and parsers (as llama-server does).
// request_json: {"messages": [...], "tools": [...], "tool_choice": "auto|required|none", "parallel_tool_calls": bool,
//                "max_tokens": n, "temperature": t, "top_p": p, "min_p": p, "top_k": k, "seed": s, "enable_thinking": bool,
//                "json_schema": {...}}
// messages and tools use OpenAI's chat completions format.
// *out_response_json: {"message": {"role", "content", "reasoning_content", "tool_calls": [...]},
//                      "finish_reason": "stop|length|tool_calls", "usage": {"prompt_tokens", "completion_tokens"}}
// cb (optional) receives each new piece of the message's content as it's parsed, and an empty piece for each token that
// adds no content (reasoning, a tool call), so that returning non-zero can stop those too. Free the response with
// fornax_llama_free.
int fornax_llama_chat_json(fornax_llama * h, const char * request_json, fornax_llama_token_cb cb, void * user,
                         char ** out_response_json);

// --- Images (vision language models, through llama.cpp's mtmd) ---
//
// Images go in as packed RGB, 3 bytes a pixel, row by row: rgb[i] holds widths[i] * heights[i] * 3 bytes.

// A vision language model: the language model's GGUF plus its mmproj GGUF (the vision encoder and projector).
// A handle from fornax_llama_create_mm also does everything a fornax_llama_create handle does.
fornax_llama * fornax_llama_create_mm(const char * model_path, const char * mmproj_path, int n_gpu_layers, int n_ctx);

// 1 if the handle takes images (it was created with an mmproj that has a vision encoder), else 0.
int fornax_llama_supports_vision(fornax_llama * h);

// The marker fornax_llama_complete_images replaces with an image (llama.cpp's mtmd_default_marker()).
const char * fornax_llama_media_marker(void);

// fornax_llama_chat with images: message i carries the next image_counts[i] images (in order), placed before its text.
// The chat is formatted with the model's own (Jinja) chat template, as fornax_llama_chat_json does.
int fornax_llama_chat_images(fornax_llama * h, const char ** roles, const char ** contents, const int * image_counts,
                           int n_messages, const unsigned char ** rgb, const uint32_t * widths, const uint32_t * heights,
                           int n_images, int max_tokens, float temperature, float top_p, float min_p, uint32_t seed,
                           fornax_llama_token_cb cb, void * user, char ** out_text, int * out_n_prompt, int * out_n_generated);

// fornax_llama_complete with images: the prompt holds fornax_llama_media_marker() once for each image, in order.
// out_n_prompt counts the image tokens too.
int fornax_llama_complete_images(fornax_llama * h, const char * prompt, const unsigned char ** rgb, const uint32_t * widths,
                               const uint32_t * heights, int n_images, int max_tokens, float temperature, float top_p,
                               float min_p, uint32_t seed, fornax_llama_token_cb cb, void * user, char ** out_text,
                               int * out_n_prompt, int * out_n_generated);

// fornax_llama_chat_json with images. In a message's content array, {"type": "image", "index": i} stands for the i-th
// RGB buffer given here, and an OpenAI {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}} part
// is decoded too (PNG, JPEG, BMP, GIF). "json_schema" constrains the reply to a JSON schema, as in fornax_llama_chat_json.
int fornax_llama_chat_json_images(fornax_llama * h, const char * request_json, const unsigned char ** rgb,
                                const uint32_t * widths, const uint32_t * heights, int n_images,
                                fornax_llama_token_cb cb, void * user, char ** out_response_json);

// --- Embeddings and reranking (pooled output, as llama-embedding and llama-server's /embedding and /rerank) ---
//
// A pooled handle runs the model with llama.cpp's embeddings output and the pooling type its GGUF sets (mean, cls,
// last; rank for a reranker), and takes only the calls below: fornax_llama_chat and the rest fail on it. Several inputs
// go in one batch, each as its own sequence; one input may be up to n_ctx tokens. cb (optional) is called with an
// empty piece after each batch; returning non-zero stops the call, which then fails with "cancelled" (rc 13).

// rerank 0: an embedding model (its GGUF sets a pooling type other than rank). rerank 1: a reranker (rank pooling,
// with a classifier head). n_gpu_layers: -1 = all. n_ctx: 0 = model default.
fornax_llama * fornax_llama_create_pooled(const char * model_path, int n_gpu_layers, int n_ctx, int rerank);

// The length of one embedding (floats).
int fornax_llama_n_embd(fornax_llama * h);

// Embeds n texts into out (n * fornax_llama_n_embd floats, row by row). normalize non-zero: each row L2-normalised.
// Texts are tokenized with the vocab's own BOS/EOS added, and not parsed for special tokens.
int fornax_llama_embed(fornax_llama * h, const char ** texts, int n, int normalize, float * out,
                     fornax_llama_token_cb cb, void * user);

// Scores n documents against a query into out_scores (n floats, in the documents' order), as llama-server's /rerank:
// the GGUF's "rerank" prompt template with {query} and {document} filled in. For Qwen3-Reranker a score is the
// probability (0 to 1) that the document answers the query.
int fornax_llama_rerank(fornax_llama * h, const char * query, const char ** docs, int n, float * out_scores,
                      fornax_llama_token_cb cb, void * user);

void fornax_llama_free(void * p);
void fornax_llama_destroy(fornax_llama * h);

#ifdef __cplusplus
}
#endif
