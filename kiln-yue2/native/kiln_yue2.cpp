// Minimal C ABI over yue2.cpp's header-only pipeline, for Java FFM.
#include "kiln_yue2.h"

#include "pipeline.h"
#include "request.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
thread_local std::string g_err;
char * dup(const std::string & s) {
    char * p = (char *) malloc(s.size() + 1);
    memcpy(p, s.c_str(), s.size() + 1);
    return p;
}
}  // namespace

struct kiln_yue2 {
    ModelStore * store = nullptr;
    Yue2Pipeline pipeline;
    std::string  model_path, vae_path;  // owned: the pipeline and store key off these
};

extern "C" {

const char * kiln_yue2_last_error(void) { return g_err.c_str(); }

kiln_yue2 * kiln_yue2_create(const char * model_path, const char * vae_path, int keep_loaded, int max_seq) {
    auto * h       = new kiln_yue2();
    h->model_path  = model_path;
    h->vae_path    = vae_path;
    h->store       = store_create(keep_loaded ? EVICT_NEVER : EVICT_STRICT);
    h->pipeline.store = h->store;
    Yue2PipelineParams params;
    params.max_seq = max_seq;
    if (!pipeline_configure(&h->pipeline, h->model_path.c_str(), h->vae_path.c_str(), params)) {
        g_err = "pipeline_configure failed (see stderr)";
        store_free(h->store);
        delete h;
        return nullptr;
    }
    return h;
}

int kiln_yue2_generate(kiln_yue2 * h, const char * request_json, float ** out_planar, int * out_n_samples,
                    char ** out_replay_json, char ** out_score) {
    *out_planar    = nullptr;
    *out_n_samples = 0;
    if (out_replay_json) *out_replay_json = nullptr;
    if (out_score) *out_score = nullptr;

    Yue2Request r;
    request_init(&r);
    if (!request_parse_json(&r, request_json)) { g_err = "malformed request JSON"; return 1; }
    request_resolve_seed(&r);
    r.lm_batch_size    = 1;
    r.synth_batch_size = 1;

    std::vector<Yue2Song> songs;
    if (!pipeline_generate(&h->pipeline, r, &songs) || songs.empty()) { g_err = "pipeline_generate failed (see stderr)"; return 2; }
    Yue2Song & s = songs[0];
    size_t n = (size_t) s.T_audio * 2;
    *out_planar = (float *) malloc(n * sizeof(float));
    memcpy(*out_planar, s.audio.data(), n * sizeof(float));
    *out_n_samples = s.T_audio;
    if (out_replay_json) {
        Yue2Request replay = request_replay(r, s.score.empty() ? r.abc : s.score, pipeline_format_tokens(s.tokens), 0, 0);
        *out_replay_json = dup(request_to_json(&replay));
    }
    if (out_score) *out_score = dup(s.score);
    return 0;
}

void kiln_yue2_free(void * p) { free(p); }

void kiln_yue2_destroy(kiln_yue2 * h) {
    if (!h) return;
    pipeline_free(&h->pipeline);
    store_free(h->store);
    delete h;
}

}  // extern "C"
