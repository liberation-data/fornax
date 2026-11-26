// Follows whisper.cpp's examples/cli/cli.cpp (the reference): whisper_full on the whole clip, token timestamps on.
#include "kiln_whisper.h"
#include "whisper.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
thread_local std::string g_err;

bool verbose() { return getenv("KILN_VERBOSE") != nullptr; }
}  // namespace

struct kiln_whisper {
    whisper_context * ctx = nullptr;
    bool dtw = false;
    std::atomic<bool> cancelled{false};
    // streaming: segments already handed to the segment callback
    kiln_whisper_progress_cb progress = nullptr;
    kiln_whisper_segment_cb segment = nullptr;
    void * user = nullptr;
    int32_t reported = 0;
};

namespace {
// Hands every segment not yet reported to the callback. whisper.cpp's own n_new is unreliable with DTW on (its DTW
// branch passes a segment index), so this counts for itself, and transcribe() flushes whatever is left at the end.
void report(kiln_whisper * h) {
    if (!h->segment) return;
    int32_t n = whisper_full_n_segments(h->ctx);
    if (n > h->reported) {
        int32_t first = h->reported;
        h->reported = n;
        h->segment(first, n - first, h->user);
    }
}
}  // namespace

extern "C" {

const char * kiln_whisper_last_error(void) { return g_err.c_str(); }

kiln_whisper * kiln_whisper_create(const char * model_path, int32_t use_gpu, int32_t gpu_device, int32_t flash_attn,
                                   int32_t dtw_preset) {
    static bool inited = false;
    if (!inited) {
        // Quiet by default: only errors, unless KILN_VERBOSE is set.
        if (!verbose()) {
            whisper_log_set([](ggml_log_level level, const char * text, void *) {
                if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
            }, nullptr);
        }
        inited = true;
    }
    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = use_gpu != 0;
    cp.gpu_device = gpu_device;
    cp.flash_attn = flash_attn != 0 && dtw_preset == 0;
    if (dtw_preset != 0) {
        cp.dtw_token_timestamps = true;
        cp.dtw_aheads_preset = (whisper_alignment_heads_preset) dtw_preset;
    }
    whisper_context * ctx = whisper_init_from_file_with_params(model_path, cp);
    if (!ctx) { g_err = std::string("unable to load whisper model: ") + model_path; return nullptr; }
    auto * h = new kiln_whisper();
    h->ctx = ctx;
    h->dtw = cp.dtw_token_timestamps;
    return h;
}

int32_t kiln_whisper_is_multilingual(kiln_whisper * h) { return whisper_is_multilingual(h->ctx); }

const char * kiln_whisper_model_type(kiln_whisper * h) { return whisper_model_type_readable(h->ctx); }

int32_t kiln_whisper_transcribe(kiln_whisper * h, const float * pcm, int32_t n_samples, const char * language,
                                int32_t translate, const char * initial_prompt, int32_t n_threads, int32_t beam_size,
                                float temperature, kiln_whisper_progress_cb progress, kiln_whisper_segment_cb segment,
                                void * user) {
    const bool beam = beam_size > 1;
    whisper_full_params p = whisper_full_default_params(beam ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
    if (n_threads > 0) p.n_threads = n_threads;
    if (beam) p.beam_search.beam_size = beam_size;
    if (temperature >= 0) p.temperature = temperature;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_timestamps = false;
    p.print_special = false;
    p.token_timestamps = true;
    const bool multilingual = whisper_is_multilingual(h->ctx);
    const bool detect = language == nullptr || *language == '\0' || strcmp(language, "auto") == 0;
    // whisper.cpp takes an unknown language as id -1 and decodes from a garbage prompt, so refuse it here.
    if (!detect && whisper_lang_id(language) < 0) {
        g_err = std::string("unknown language: ") + language;
        return -101;
    }
    if (!multilingual) {
        p.language = "en";
        p.translate = false;
    } else {
        p.language = detect ? "auto" : language;
        p.translate = translate != 0;
    }
    if (initial_prompt && *initial_prompt) p.initial_prompt = initial_prompt;

    h->cancelled = false;
    h->progress = progress;
    h->segment = segment;
    h->user = user;
    h->reported = 0;
    p.abort_callback = [](void * u) { return static_cast<kiln_whisper *>(u)->cancelled.load(); };
    p.abort_callback_user_data = h;
    p.encoder_begin_callback = [](whisper_context *, whisper_state *, void * u) {
        return !static_cast<kiln_whisper *>(u)->cancelled.load();
    };
    p.encoder_begin_callback_user_data = h;
    if (progress) {
        p.progress_callback = [](whisper_context *, whisper_state *, int pct, void * u) {
            auto * hh = static_cast<kiln_whisper *>(u);
            hh->progress(pct, hh->user);
        };
        p.progress_callback_user_data = h;
    }
    if (segment) {
        p.new_segment_callback = [](whisper_context *, whisper_state *, int, void * u) {
            report(static_cast<kiln_whisper *>(u));
        };
        p.new_segment_callback_user_data = h;
    }

    int rc = whisper_full(h->ctx, p, pcm, n_samples);
    if (h->cancelled) rc = -100;
    if (rc != 0) {
        g_err = h->cancelled ? "cancelled" : "whisper_full failed: " + std::to_string(rc);
    } else {
        report(h);
        if (progress) progress(100, user);
    }
    h->progress = nullptr;
    h->segment = nullptr;
    h->user = nullptr;
    return rc;
}

void kiln_whisper_cancel(kiln_whisper * h) { h->cancelled = true; }

const char * kiln_whisper_language(kiln_whisper * h) { return whisper_lang_str(whisper_full_lang_id(h->ctx)); }

int32_t kiln_whisper_n_segments(kiln_whisper * h) { return whisper_full_n_segments(h->ctx); }
int64_t kiln_whisper_segment_t0(kiln_whisper * h, int32_t s) { return whisper_full_get_segment_t0(h->ctx, s); }
int64_t kiln_whisper_segment_t1(kiln_whisper * h, int32_t s) { return whisper_full_get_segment_t1(h->ctx, s); }
const char * kiln_whisper_segment_text(kiln_whisper * h, int32_t s) { return whisper_full_get_segment_text(h->ctx, s); }
int32_t kiln_whisper_n_tokens(kiln_whisper * h, int32_t s) { return whisper_full_n_tokens(h->ctx, s); }

const char * kiln_whisper_token(kiln_whisper * h, int32_t s, int32_t t, int32_t * special, float * p, int64_t * t0,
                                int64_t * t1, int64_t * t_dtw) {
    whisper_token_data d = whisper_full_get_token_data(h->ctx, s, t);
    *special = d.id >= whisper_token_eot(h->ctx);
    *p = d.p;
    *t0 = whisper_full_get_token_t0(h->ctx, s, t);
    *t1 = whisper_full_get_token_t1(h->ctx, s, t);
    *t_dtw = h->dtw ? d.t_dtw : -1;
    return whisper_full_get_token_text(h->ctx, s, t);
}

void kiln_whisper_destroy(kiln_whisper * h) {
    if (!h) return;
    if (h->ctx) whisper_free(h->ctx);
    delete h;
}

}  // extern "C"
