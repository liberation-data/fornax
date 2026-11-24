#include "kiln_llama.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
thread_local std::string g_err;
int fail(const std::string & e, int rc) { g_err = e; return rc; }
char * dup(const std::string & s) {
    char * p = (char *) malloc(s.size() + 1);
    memcpy(p, s.c_str(), s.size() + 1);
    return p;
}
}  // namespace

struct kiln_llama {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_batch_ext * batch = nullptr;
};

extern "C" {

const char * kiln_llama_last_error(void) { return g_err.c_str(); }

kiln_llama * kiln_llama_create(const char * model_path, int n_gpu_layers, int n_ctx) {
    static bool inited = false;
    if (!inited) {
        // Quiet by default: only errors, unless KILN_VERBOSE is set.
        if (!getenv("KILN_VERBOSE")) {
            llama_log_set([](ggml_log_level level, const char * text, void *) {
                if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
            }, nullptr);
        }
        ggml_backend_load_all();
        llama_backend_init();
        inited = true;
    }
    auto * h = new kiln_llama();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers < 0 ? 999 : n_gpu_layers;
    if (!getenv("KILN_VERBOSE")) mp.progress_callback = [](float, void *) { return true; };  // no loading dots
    h->model = llama_model_load_from_file(model_path, mp);
    if (!h->model) { g_err = std::string("unable to load model: ") + model_path; delete h; return nullptr; }
    h->vocab = llama_model_get_vocab(h->model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = n_ctx > 0 ? n_ctx : 0;
    h->ctx = llama_init_from_model(h->model, cp);
    if (!h->ctx) { g_err = "failed to create llama_context"; kiln_llama_destroy(h); return nullptr; }
    h->batch = llama_batch_ext_init(h->ctx);
    return h;
}

int kiln_llama_complete(kiln_llama * h, const char * prompt, int max_tokens, float temperature, float top_p,
                        float min_p, uint32_t seed, kiln_llama_token_cb cb, void * user, char ** out_text,
                        int * out_n_prompt, int * out_n_generated) {
    *out_text = nullptr;
    llama_memory_clear(llama_get_memory(h->ctx), true);

    const int len = (int) strlen(prompt);
    int n = -llama_tokenize(h->vocab, prompt, len, nullptr, 0, true, true);
    std::vector<llama_token> toks(n);
    if (llama_tokenize(h->vocab, prompt, len, toks.data(), n, true, true) < 0) return fail("tokenize failed", 1);
    const int n_ctx = (int) llama_n_ctx(h->ctx);
    if (n >= n_ctx) return fail("prompt (" + std::to_string(n) + " tokens) exceeds context " + std::to_string(n_ctx), 2);
    if (out_n_prompt) *out_n_prompt = n;

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (top_p > 0 && top_p < 1) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        if (min_p > 0) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }

    std::string out;
    int pos = 0, generated = 0, rc = 0;
    const llama_token * cur = toks.data();
    int n_cur = n;
    llama_token tok;
    while (max_tokens <= 0 || generated < max_tokens) {
        if (pos + n_cur > n_ctx) break;  // context full: return what we have
        llama_batch_ext_clear(h->batch);
        for (int i = 0; i < n_cur; i++) {
            int idx = llama_batch_ext_add_token(h->batch, 0, cur[i]);
            llama_pos p = pos + i;
            llama_batch_ext_set_pos(h->batch, idx, &p);
        }
        llama_batch_ext_set_output_logits(h->batch, n_cur - 1, true);
        if (int r = llama_process(h->ctx, LLAMA_PROCESS_TYPE_DECODE, h->batch); r != 0) {
            rc = fail("llama_process failed: " + std::to_string(r), 3);
            break;
        }
        pos += n_cur;
        tok = llama_sampler_sample(smpl, h->ctx, -1);
        if (llama_vocab_is_eog(h->vocab, tok)) break;
        char buf[256];
        int m = llama_token_to_piece(h->vocab, tok, buf, sizeof buf, 0, true);
        if (m < 0) { rc = fail("token_to_piece failed", 4); break; }
        std::string piece(buf, m);
        out += piece;
        generated++;
        if (cb && cb(piece.c_str(), user) != 0) break;
        cur = &tok;
        n_cur = 1;
    }
    llama_sampler_free(smpl);
    if (out_n_generated) *out_n_generated = generated;
    *out_text = dup(out);
    return rc;
}

int kiln_llama_chat(kiln_llama * h, const char ** roles, const char ** contents, int n_messages,
                    int max_tokens, float temperature, float top_p, float min_p, uint32_t seed,
                    kiln_llama_token_cb cb, void * user, char ** out_text, int * out_n_prompt, int * out_n_generated) {
    std::vector<llama_chat_message> msgs(n_messages);
    for (int i = 0; i < n_messages; i++) msgs[i] = {roles[i], contents[i]};
    const char * tmpl = llama_model_chat_template(h->model, nullptr);
    int need = llama_chat_apply_template(tmpl, msgs.data(), n_messages, true, nullptr, 0);
    if (need < 0) return fail("failed to apply the chat template", 5);
    std::vector<char> buf(need + 1);
    llama_chat_apply_template(tmpl, msgs.data(), n_messages, true, buf.data(), (int) buf.size());
    std::string prompt(buf.data(), need);
    return kiln_llama_complete(h, prompt.c_str(), max_tokens, temperature, top_p, min_p, seed, cb, user, out_text,
                               out_n_prompt, out_n_generated);
}

void kiln_llama_free(void * p) { free(p); }

void kiln_llama_destroy(kiln_llama * h) {
    if (!h) return;
    if (h->batch) llama_batch_ext_free(h->batch);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

}  // extern "C"
