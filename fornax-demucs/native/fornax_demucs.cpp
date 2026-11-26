// Follows stems.cpp's tools/stems-libtest.c (its reference host for the C ABI), which matches stems-split --float32.
#include "fornax_demucs.h"
#include "libstems_v1.h"
#include "ggml.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
thread_local std::string g_err;

bool verbose() { return getenv("FORNAX_VERBOSE") != nullptr; }

const stems_api_v1 * api() {
    static const stems_api_v1 * a = stems_get_api(STEMS_ABI_VERSION_1);
    return a;
}
}  // namespace

struct fornax_demucs {
    stems_context * ctx = nullptr;
    stems_model_info_v1 info{};
    std::atomic<bool> cancel{false};
    fornax_demucs_progress_cb progress = nullptr;
    void * user = nullptr;
};

extern "C" {

const char * fornax_demucs_last_error(void) { return g_err.c_str(); }

fornax_demucs * fornax_demucs_create(const char * model_path, const char * device, int32_t cpu_threads) {
    const stems_api_v1 * a = api();
    if (!a) { g_err = "libstems ABI v1 is not available"; return nullptr; }
    static bool inited = false;
    if (!inited) {
        // Quiet by default: only ggml's errors, unless FORNAX_VERBOSE is set. (stems.cpp itself prints a line or two
        // naming the backend it chose.)
        if (!verbose()) {
            ggml_log_set([](ggml_log_level level, const char * text, void *) {
                if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
            }, nullptr);
        }
        inited = true;
    }
    stems_context_config_v1 cfg{};
    cfg.size = sizeof cfg;
    a->context_config_init(&cfg);
    cfg.model_path = model_path;
    cfg.device = device;
    cfg.cpu_threads = cpu_threads > 0 ? cpu_threads : 0;
    stems_error_v1 err{};
    err.size = sizeof err;
    a->error_init(&err);
    auto * h = new fornax_demucs();
    if (a->context_create(&cfg, &h->ctx, &err) != STEMS_STATUS_OK_V1) {
        g_err = std::string("unable to load ") + (model_path ? model_path : "(null)") + ": " + err.message;
        delete h;
        return nullptr;
    }
    h->info.size = sizeof h->info;
    a->model_info_init(&h->info);
    if (a->model_info(h->ctx, &h->info, &err) != STEMS_STATUS_OK_V1) {
        g_err = err.message;
        fornax_demucs_destroy(h);
        return nullptr;
    }
    return h;
}

const char * fornax_demucs_name(fornax_demucs * h) { return h->info.name; }
const char * fornax_demucs_backend(fornax_demucs * h) { return h->info.backend; }
int32_t fornax_demucs_n_sources(fornax_demucs * h) { return (int32_t) h->info.n_sources; }
const char * fornax_demucs_source(fornax_demucs * h, int32_t i) {
    return i >= 0 && (uint32_t) i < h->info.n_sources ? h->info.source_names[i] : nullptr;
}
int32_t fornax_demucs_sample_rate(fornax_demucs * h) { return (int32_t) h->info.sample_rate; }
int32_t fornax_demucs_channels(fornax_demucs * h) { return (int32_t) h->info.audio_channels; }

int32_t fornax_demucs_separate(fornax_demucs * h, const float * pcm, int64_t n_samples, int32_t n_channels,
                             int32_t sample_rate, int32_t shifts, float overlap, uint32_t seed,
                             fornax_demucs_progress_cb progress, void * user, float ** out) {
    const stems_api_v1 * a = api();
    *out = nullptr;
    if (n_samples <= 0 || n_channels <= 0 || sample_rate <= 0) {
        g_err = "audio needs samples, channels and a sample rate";
        return FORNAX_DEMUCS_INVALID_ARGUMENT;
    }
    stems_request_v1 req{};
    req.size = sizeof req;
    a->request_init(&req);
    req.input.samples = pcm;
    req.input.n_samples = (uint64_t) n_samples;
    req.input.n_channels = (uint32_t) n_channels;
    req.input.sample_rate = (uint32_t) sample_rate;
    req.input.layout = STEMS_AUDIO_PLANAR_V1;
    req.shifts = shifts;
    req.overlap = overlap;
    req.seed = seed;
    h->progress = progress;
    h->user = user;
    req.callback_user = h;
    req.should_cancel = [](void * u) -> int32_t { return static_cast<fornax_demucs *>(u)->cancel.load() ? 1 : 0; };
    if (progress) {
        req.on_progress = [](void * u, const stems_progress_v1 * p) {
            auto * hh = static_cast<fornax_demucs *>(u);
            hh->progress(p->pass, p->total, hh->user);
        };
    }
    stems_result_v1 res{};
    res.size = sizeof res;
    a->result_init(&res);
    stems_error_v1 err{};
    err.size = sizeof err;
    a->error_init(&err);
    int32_t rc = a->separate(h->ctx, &req, &res, &err);
    h->progress = nullptr;
    h->user = nullptr;
    if (rc != STEMS_STATUS_OK_V1) {
        g_err = rc == STEMS_STATUS_CANCELLED_V1 ? "cancelled" : err.message;
        a->result_free(&res);
        return rc;
    }
    // libstems allocates its result with malloc and frees it with free; the caller takes it over.
    *out = res.samples;
    return FORNAX_DEMUCS_OK;
}

void fornax_demucs_set_cancel(fornax_demucs * h, int32_t on) { h->cancel = on != 0; }

void fornax_demucs_free(float * p) { std::free(p); }

void fornax_demucs_destroy(fornax_demucs * h) {
    if (!h) return;
    if (h->ctx) api()->context_destroy(h->ctx);
    delete h;
}

}  // extern "C"
