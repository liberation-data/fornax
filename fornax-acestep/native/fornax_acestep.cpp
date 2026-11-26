// Minimal C ABI over acestep.cpp's C++ pipeline, for Java FFM (or any FFI).
// One handle = one ModelStore + optional LM + synth (text-enc, DiT, VAE).
// Request in = acestep.cpp JSON (same schema as examples/*.json). Audio out = planar stereo f32 @ 48 kHz.

#include "fornax_acestep.h"

#include "model-store.h"
#include "pipeline-lm.h"
#include "pipeline-synth.h"
#include "request.h"
#include "task-types.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
thread_local std::string g_err;
void set_err(const std::string & e) { g_err = e; }
char * dup(const std::string & s) {
    char * p = (char *) malloc(s.size() + 1);
    memcpy(p, s.c_str(), s.size() + 1);
    return p;
}
}  // namespace

struct fornax_acestep {
    ModelStore * store = nullptr;
    AceLm *      lm    = nullptr;
    AceSynth *   synth = nullptr;
    // acestep keeps these const char* in its params and reads them lazily, so the handle owns copies
    std::string  lm_path, text_enc_path, dit_path, vae_path;
    std::atomic<bool> cancel{false};
};

namespace {
bool cancelled(void * h) { return static_cast<fornax_acestep *>(h)->cancel.load(std::memory_order_relaxed); }
int  stopped_or(fornax_acestep * h, int rc, const char * what) {
    if (h->cancel.load()) { set_err("cancelled"); return FORNAX_ACESTEP_CANCELLED; }
    set_err(what);
    return rc;
}
}  // namespace

extern "C" {

const char * fornax_acestep_last_error(void) { return g_err.c_str(); }

fornax_acestep * fornax_acestep_create(const char * lm_path, const char * text_enc_path, const char * dit_path, const char * vae_path,
                     int keep_loaded) {
    auto * h  = new fornax_acestep();
    h->lm_path       = lm_path ? lm_path : "";
    h->text_enc_path = text_enc_path;
    h->dit_path      = dit_path;
    h->vae_path      = vae_path;
    h->store  = store_create(keep_loaded ? EVICT_NEVER : EVICT_STRICT);
    if (!h->lm_path.empty()) {
        AceLmParams p;
        ace_lm_default_params(&p);
        p.model_path = h->lm_path.c_str();
        p.max_batch  = 1;
        h->lm        = ace_lm_load(h->store, &p);
        if (!h->lm) { set_err("ace_lm_load failed: " + h->lm_path); fornax_acestep_destroy(h); return nullptr; }
    }
    AceSynthParams sp;
    ace_synth_default_params(&sp);
    sp.text_encoder_path = h->text_enc_path.c_str();
    sp.dit_path          = h->dit_path.c_str();
    sp.vae_path          = h->vae_path.c_str();
    // acestep.cpp decodes 1024 latent frames per tile. On a power-limited GPU (a laptop on battery) one tile
    // that long can exceed the driver's compute timeout and lose the device; 512 avoids that and costs little.
    sp.vae_chunk = 512;
    if (const char * c = getenv("FORNAX_ACESTEP_VAE_CHUNK")) sp.vae_chunk = atoi(c);
    h->synth             = ace_synth_load(h->store, &sp);
    if (!h->synth) { set_err("ace_synth_load failed"); fornax_acestep_destroy(h); return nullptr; }
    return h;
}

int fornax_acestep_generate(fornax_acestep * h, const char * request_json, float ** out_planar, int * out_n_samples,
                   char ** out_request_json) {
    *out_planar    = nullptr;
    *out_n_samples = 0;
    if (out_request_json) *out_request_json = nullptr;

    AceRequest req;
    request_init(&req);
    if (!request_parse_json(&req, request_json)) { set_err("malformed request JSON"); return 1; }
    request_resolve_lm_seed(&req);
    request_resolve_seed(&req);

    // Stage 1 (optional): the 5 Hz LM writes metadata, lyrics and audio codes.
    if (h->lm && req.audio_codes.empty()) {
        int mode = req.lm_mode == LM_MODE_NAME_INSPIRE ? LM_MODE_INSPIRE
                 : req.lm_mode == LM_MODE_NAME_FORMAT  ? LM_MODE_FORMAT
                                                       : LM_MODE_GENERATE;
        AceRequest out;
        request_init(&out);
        if (ace_lm_generate(h->lm, &req, 1, &out, nullptr, nullptr, cancelled, h, mode) != 0)
            return stopped_or(h, 2, "ace_lm_generate failed");
        req = out;
    }

    // Stage 2: DiT (flow matching over latents), then VAE decode to 48 kHz stereo.
    AceSynthJob * job = ace_synth_job_run_dit(h->synth, &req, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, 1, cancelled, h);
    if (!job) return stopped_or(h, 3, "ace_synth_job_run_dit failed");
    AceAudio audio{};
    int      rc = ace_synth_job_run_vae(h->synth, job, &audio, cancelled, h);
    ace_synth_job_free(job);
    if (rc != 0 || !audio.samples) {
        if (audio.samples) ace_audio_free(&audio);
        return stopped_or(h, 4, "ace_synth_job_run_vae failed");
    }

    size_t n    = (size_t) audio.n_samples * 2;
    *out_planar = (float *) malloc(n * sizeof(float));
    memcpy(*out_planar, audio.samples, n * sizeof(float));
    *out_n_samples = audio.n_samples;
    ace_audio_free(&audio);
    if (out_request_json) *out_request_json = dup(request_to_json(&req, true));
    return 0;
}

void fornax_acestep_set_cancel(fornax_acestep * h, int cancel) { if (h) h->cancel.store(cancel != 0); }

void fornax_acestep_free(void * p) { free(p); }

void fornax_acestep_destroy(fornax_acestep * h) {
    if (!h) return;
    if (h->lm) ace_lm_free(h->lm);
    if (h->synth) ace_synth_free(h->synth);
    if (h->store) store_free(h->store);
    delete h;
}

}  // extern "C"
