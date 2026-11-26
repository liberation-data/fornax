// Follows llama.cpp's tools/tts/tts.cpp (the reference loop) and the server's process_tts_slots() (streaming).
#include <cctype>
#include <fstream>
#include <iterator>
#include "kiln_tts.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
thread_local std::string g_err;
int fail(const std::string & e, int rc) { g_err = e; return rc; }
}  // namespace

struct kiln_tts {
    common_params params;
    common_init_result_ptr init;
    llama_model *   model = nullptr;
    llama_context * lctx  = nullptr;
    mtmd_context *  mctx  = nullptr;
    mtmd_helper_gen_audio * gen = nullptr;
    common_chat_templates_ptr tmpls;
    mtmd_gen_audio_info info{};
};

extern "C" {

const char * kiln_tts_last_error(void) { return g_err.c_str(); }

kiln_tts * kiln_tts_create(const char * model_path, const char * mmproj_path, int n_gpu_layers, int n_ctx) {
    static bool inited = false;
    if (!inited) {
        // Quiet by default: only errors, unless KILN_VERBOSE is set.
        if (!getenv("KILN_VERBOSE")) {
            auto quiet = [](ggml_log_level level, const char * text, void *) {
                if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
            };
            llama_log_set(quiet, nullptr);
            mtmd_helper_log_set(quiet, nullptr);
            common_log_set_verbosity_thold(-1);
        } else {
            mtmd_helper_log_set(common_log_default_callback, nullptr);
        }
        ggml_backend_load_all();
        llama_backend_init();
        inited = true;
    }
    auto * h = new kiln_tts();
    common_params & p = h->params;
    p.model.path  = model_path;
    p.mmproj.path = mmproj_path;
    p.n_gpu_layers = n_gpu_layers < 0 ? -2 : n_gpu_layers;  // <= -2 is all
    if (n_ctx > 0) p.n_ctx = n_ctx;
    p.embedding = true;  // the audio pipeline reads the backbone's hidden states
    p.warmup = false;
    if (!getenv("KILN_VERBOSE")) p.load_progress_callback = [](float, void *) { return true; };  // no loading dots
    // llama-tts's defaults (common/arg.cpp, LLAMA_EXAMPLE_TTS)
    p.sampling.penalty_repeat = 1.05f;
    p.sampling.penalty_last_n = -1;
    // what common_params_parse() finishes off after the arguments: thread counts from the hardware
    postprocess_cpu_params(p.cpuparams, nullptr);
    postprocess_cpu_params(p.cpuparams_batch, &p.cpuparams);

    try {
        h->init = common_init_from_params(p);
    } catch (const std::exception & e) {
        g_err = std::string("load: ") + e.what();
        delete h;
        return nullptr;
    }
    h->model = h->init ? h->init->model() : nullptr;
    h->lctx  = h->init ? h->init->context() : nullptr;
    if (!h->model || !h->lctx) { g_err = std::string("unable to load model: ") + model_path; kiln_tts_destroy(h); return nullptr; }

    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu   = p.mmproj_use_gpu;
    mp.n_threads = p.cpuparams.n_threads;
    mp.warmup    = false;
    h->mctx = mtmd_init_from_file(mmproj_path, h->model, mp);
    if (!h->mctx) { g_err = std::string("unable to load mmproj: ") + mmproj_path; kiln_tts_destroy(h); return nullptr; }
    h->info = mtmd_gen_audio_get_info(h->mctx);
    if (h->info.type == MTMD_GEN_AUDIO_TYPE_NONE) {
        g_err = std::string("the mmproj does not generate audio: ") + mmproj_path;
        kiln_tts_destroy(h);
        return nullptr;
    }
    h->gen = mtmd_helper_gen_audio_init(h->lctx, h->mctx);
    if (!h->gen) { g_err = "the context cannot run this model's audio pipeline (context too small?)"; kiln_tts_destroy(h); return nullptr; }
    try {
        h->tmpls = common_chat_templates_init(h->model, "");
    } catch (const std::exception & e) {
        g_err = std::string("chat template: ") + e.what();
        kiln_tts_destroy(h);
        return nullptr;
    }
    return h;
}

void kiln_tts_info(kiln_tts * h, int32_t * sample_rate, int32_t * n_channels, int32_t * n_voices) {
    if (sample_rate) *sample_rate = h->info.sample_rate;
    if (n_channels) *n_channels = h->info.n_channels;
    if (n_voices) *n_voices = h->info.n_voices;
}

const char * kiln_tts_voice_name(kiln_tts * h, int32_t idx) { return mtmd_gen_audio_voice_name(h->mctx, idx); }

int kiln_tts_speak(kiln_tts * h, const char * text, const char * voice_c, const char * lang_c, float temperature,
                   uint32_t seed, float speed, int32_t max_frames, kiln_tts_audio_cb cb, void * user,
                   float ** out_pcm, int64_t * out_n_frames, int32_t * out_n_generated) {
    *out_pcm = nullptr;
    *out_n_frames = 0;
    if (out_n_generated) *out_n_generated = 0;
    if (!text || !*text) return fail("no text to speak", 1);
    std::string voice = voice_c ? voice_c : "", lang = lang_c ? lang_c : "";

    // A voice of "@<file>" is a reference clip to speak like (wav or mp3; VoxCPM2 and Qwen3-TTS clone it). A
    // "<file>.txt" beside it, holding what the clip says, lets the model continue that speaker rather than only
    // imitate the sound, which keeps an accent much steadier. Otherwise a model with predefined voices speaks with
    // one (the first by default); others name the voice in the prompt.
    mtmd::bitmap_ptr speaker;
    std::string speaker_text;
    if (!voice.empty() && voice[0] == '@') {
        if (std::ifstream said(voice.substr(1) + ".txt"); said) {
            speaker_text.assign(std::istreambuf_iterator<char>(said), {});
            while (!speaker_text.empty() && std::isspace((unsigned char) speaker_text.back())) speaker_text.pop_back();
        }
        auto wrapper = mtmd_helper_bitmap_init_from_file(h->mctx, voice.c_str() + 1, false, mtmd_helper_init_opt_default());
        if (!wrapper.bitmap) return fail("failed to load the reference clip " + voice.substr(1), 2);
        speaker.reset(wrapper.bitmap);
        voice.clear();
    } else if (h->info.n_voices > 0) {
        if (voice.empty()) voice = mtmd_gen_audio_voice_name(h->mctx, 0);
        speaker.reset(mtmd_gen_audio_get_voice(h->mctx, voice.c_str()));
        if (!speaker) {
            std::string names;
            for (int32_t i = 0; const char * n = mtmd_gen_audio_voice_name(h->mctx, i); i++) names += std::string(i ? ", " : "") + n;
            return fail("unknown voice \"" + voice + "\", the model has: " + names, 2);
        }
    }

    std::string prompt;
    try {
        prompt = common_chat_get_tts_prompt(h->tmpls.get(), text, lang, voice, [&](const std::string & word) {
            char buf[512];
            return mtmd_gen_audio_g2p(h->mctx, word.c_str(), buf, sizeof(buf)) > 0 ? std::string(buf) : std::string();
        });
    } catch (const std::exception & e) {
        return fail(std::string("failed to prepare the text to speak: ") + e.what(), 3);
    }
    if (prompt.empty()) prompt = text;

    llama_memory_clear(llama_get_memory(h->lctx), true);
    mtmd_helper_gen_audio_reset(h->gen);
    llama_set_embeddings(h->lctx, true);

    const bool stream = cb != nullptr;
    mtmd_helper_gen_audio_inp inp = mtmd_helper_gen_audio_inp_default();
    inp.stream      = stream;
    inp.seq_id      = 0;
    inp.prompt      = prompt.c_str();
    inp.prompt_len  = prompt.size();
    inp.speaker_ref = speaker.get();
    inp.speaker_ref_text = speaker_text.empty() ? nullptr : speaker_text.c_str();
    inp.lang        = lang.empty() ? nullptr : lang.c_str();
    inp.top_k       = h->params.sampling.top_k;
    inp.top_p       = h->params.sampling.top_p;
    inp.seed        = seed;
    if (temperature >= 0) inp.temp = temperature;
    if (speed > 0) inp.speed = speed;
    inp.out_type    = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
    if (int r = mtmd_helper_gen_audio_set_input(h->gen, &inp); r != 0)
        return fail(r == 2 ? "the model does not take these settings (speed?)" : "set_input failed", 4);

    common_params_sampling sp = h->params.sampling;
    sp.seed = seed;
    if (temperature >= 0) sp.temp = temperature;
    common_sampler_ptr smpl(common_sampler_init(h->model, sp));
    if (!smpl) return fail("failed to initialise the sampler", 5);

    for (;;) {
        int32_t r = mtmd_helper_gen_audio_step_prompt(h->gen, llama_n_batch(h->lctx));
        if (r < 0) return fail("prompt processing failed", 6);
        if (r == 0) break;
    }

    std::vector<float> pcm;
    const int32_t ch = h->info.n_channels > 0 ? h->info.n_channels : 1;
    bool cancelled = false;
    // Appends the audio decoded since the last call; in stream mode also hands it to cb.
    auto take = [&]() -> int {
        int32_t rate = 0;
        const char * data = nullptr;
        size_t len = 0;
        if (mtmd_helper_gen_audio_get_output(h->gen, &rate, &data, &len, nullptr) != 0) return 1;
        if (len == 0) return 0;
        const float * f = (const float *) data;
        const size_t n = len / sizeof(float);
        if (stream && !cancelled && cb(f, (int64_t) (n / ch), ch, user) != 0) cancelled = true;
        pcm.insert(pcm.end(), f, f + n);
        return 0;
    };

    auto sample = [&]() {
        llama_token t = common_sampler_sample(smpl.get(), h->lctx, -1);
        common_sampler_accept(smpl.get(), t, true);
        return t;
    };
    const int32_t n_model = mtmd_helper_gen_audio_max_frames(h->gen);
    const int32_t limit = max_frames > 0 ? max_frames : n_model > 0 ? n_model : INT32_MAX;
    llama_token sampled = sample();
    // h_state is only valid until the next decode, the next step_gen() reads it after we sample
    const float * e = llama_get_embeddings_ith(h->lctx, -1);
    if (!e) return fail("the backbone gave no hidden state for the prompt", 11);
    std::vector<float> h_prompt(e, e + llama_model_n_embd(h->model));
    const float * h_state = h_prompt.data();
    int n_frames = 0;
    bool stop = false;
    while (!stop && !cancelled && n_frames < limit) {
        const float * h_next = nullptr;
        if (mtmd_helper_gen_audio_step_gen(h->gen, sampled, h_state, &h_next, &stop) != 0)
            return fail("step_gen failed at frame " + std::to_string(n_frames), 7);
        if (!h_next) break;  // stopped without a new frame
        n_frames++;
        h_state = h_next;
        if (!stop) sampled = sample();
        if (stream && take() != 0) return fail("streaming output failed", 8);
    }
    if (stream && mtmd_helper_gen_audio_flush(h->gen) != 0) return fail("flush failed", 9);
    if (take() != 0) return fail("get_output failed", 10);

    if (out_n_generated) *out_n_generated = n_frames;
    *out_n_frames = (int64_t) (pcm.size() / ch);
    *out_pcm = (float *) malloc(std::max<size_t>(1, pcm.size()) * sizeof(float));
    if (!*out_pcm) { *out_n_frames = 0; return fail("out of memory for the audio", 12); }
    if (!pcm.empty()) memcpy(*out_pcm, pcm.data(), pcm.size() * sizeof(float));
    return 0;
}

void kiln_tts_free(void * p) { free(p); }

void kiln_tts_destroy(kiln_tts * h) {
    if (!h) return;
    if (h->gen) mtmd_helper_gen_audio_free(h->gen);
    h->tmpls.reset();
    if (h->mctx) mtmd_free(h->mctx);
    h->init.reset();
    delete h;
}

}  // extern "C"
