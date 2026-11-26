#include "kiln_llama.h"
#include "llama.h"
#include "chat.h"
#include "common.h"
#include "json.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
    common_chat_templates_ptr tmpls;
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_batch_ext * batch = nullptr;
    mtmd_context *  mctx  = nullptr;  // only for a handle from kiln_llama_create_mm
};

namespace {
llama_sampler * simple_sampler(float temperature, float top_p, float min_p, uint32_t seed) {
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (temperature <= 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        if (top_p > 0 && top_p < 1) llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
        if (min_p > 0) llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    }
    return smpl;
}

// Decodes `cur` (one batch) at positions pos.., with logits for its last token.
int decode(kiln_llama * h, const llama_token * cur, int n_cur, int pos) {
    llama_batch_ext_clear(h->batch);
    for (int i = 0; i < n_cur; i++) {
        int idx = llama_batch_ext_add_token(h->batch, 0, cur[i]);
        llama_pos p = pos + i;
        llama_batch_ext_set_pos(h->batch, idx, &p);
    }
    llama_batch_ext_set_output_logits(h->batch, n_cur - 1, true);
    return llama_process(h->ctx, LLAMA_PROCESS_TYPE_DECODE, h->batch);
}

// Samples and decodes until end of turn, max_tokens or a full context, starting from logits already computed for
// the prompt (which ends at position pos and fills `cells` of the context). Positions and cells differ after an
// M-RoPE image (Qwen-VL: ~40 positions for ~1200 tokens), and it's the cells that run out.
int generate(kiln_llama * h, llama_sampler * smpl, int pos, int cells, int max_tokens, kiln_llama_token_cb cb,
             void * user, std::string & out, int & generated) {
    const int n_ctx = (int) llama_n_ctx(h->ctx);
    while (max_tokens <= 0 || generated < max_tokens) {
        llama_token tok = llama_sampler_sample(smpl, h->ctx, -1);
        if (llama_vocab_is_eog(h->vocab, tok)) break;
        char buf[256];
        int m = llama_token_to_piece(h->vocab, tok, buf, sizeof buf, 0, true);
        if (m < 0) return fail("token_to_piece failed", 4);
        std::string piece(buf, m);
        out += piece;
        generated++;
        if (cb && cb(piece.c_str(), user) != 0) break;
        if (cells + 1 > n_ctx) break;  // context full: return what we have
        if (int r = decode(h, &tok, 1, pos); r != 0) return fail("llama_process failed: " + std::to_string(r), 3);
        pos++;
        cells++;
    }
    return 0;
}

// Packed RGB buffers (3 bytes a pixel) as mtmd bitmaps, in order.
std::vector<mtmd::bitmap_ptr> rgb_bitmaps(const unsigned char ** rgb, const uint32_t * widths, const uint32_t * heights, int n) {
    std::vector<mtmd::bitmap_ptr> out;
    for (int i = 0; i < n; i++) {
        if (!rgb[i] || widths[i] == 0 || heights[i] == 0) throw std::invalid_argument("image " + std::to_string(i) + " is empty");
        out.emplace_back(mtmd_bitmap_init(widths[i], heights[i], rgb[i]));
    }
    return out;
}

std::vector<mtmd_bitmap *> raw(const std::vector<mtmd::bitmap_ptr> & v) {
    std::vector<mtmd_bitmap *> out;
    for (const auto & b : v) out.push_back(b.get());
    return out;
}

std::string base64_decode(const std::string & in) {
    static const std::string chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (c == '=') break;
        if (isspace(c)) continue;
        size_t d = chars.find(c == '-' ? '+' : c == '_' ? '/' : (char) c);
        if (d == std::string::npos) throw std::invalid_argument("invalid base64 in image_url");
        val = ((val << 6) | (int) d) & 0xFFFFFF;
        bits += 6;
        if (bits >= 0) { out.push_back((char) ((val >> bits) & 0xFF)); bits -= 8; }
    }
    return out;
}

// The marker for images inside a chat: no user text will contain it, so only Kiln's own image parts become images.
const std::string & private_marker() {
    static const std::string m = [] {
        std::random_device rd;
        char buf[64];
        snprintf(buf, sizeof buf, "<__kiln_media_%08x%08x__>", rd(), rd());
        return std::string(buf);
    }();
    return m;
}

// Images found in a request. `owned` keeps the ones decoded here alive; `all` is every image, in prompt order.
struct request_images {
    std::vector<mtmd::bitmap_ptr> owned;
    std::vector<mtmd_bitmap *> all;
};

// Replaces the image parts of the request's messages with media markers, collecting the images in order:
// {"type": "image", "index": i} is given[i]; {"type": "image_url", ...} is a base64 data: URI (PNG, JPEG, ...).
void take_images(kiln_llama * h, common_json & messages, const std::vector<mtmd_bitmap *> & given, request_images & out) {
    if (!messages.is_array()) return;
    for (auto & m : messages) {
        if (!m.is_object() || !m.contains("content") || !m.at("content").is_array()) continue;
        for (auto & p : m.at("content")) {
            std::string type = p.is_object() ? p.value("type", "") : "";
            if (type != "image" && type != "image_url") continue;
            if (!h->mctx || !mtmd_support_vision(h->mctx))
                throw std::invalid_argument("this model takes no images (load it with its mmproj)");
            if (type == "image") {
                int i = p.at("index").get<int>();
                if (i < 0 || i >= (int) given.size())
                    throw std::invalid_argument("image index " + std::to_string(i) + " out of range (" +
                                                std::to_string(given.size()) + " given)");
                out.all.push_back(given[i]);
            } else {
                std::string url = p.at("image_url").at("url").get<std::string>();
                size_t comma = url.find(',');
                if (url.rfind("data:", 0) != 0 || comma == std::string::npos || comma < 6 ||
                    url.compare(comma - 6, 6, "base64") != 0)
                    throw std::invalid_argument("image_url must be a base64 data: URI");
                std::string bytes = base64_decode(url.substr(comma + 1));
                auto w = mtmd_helper_bitmap_init_from_buf(h->mctx, (const unsigned char *) bytes.data(), bytes.size(),
                                                          false, mtmd_helper_init_opt_default());
                if (!w.bitmap) throw std::invalid_argument("could not decode an image_url image");
                out.owned.emplace_back(w.bitmap);
                out.all.push_back(w.bitmap);
            }
            p = common_json{{"type", "media_marker"}, {"text", private_marker()}};
        }
    }
}

// A rendered chat prompt with image markers, tokenized with the images in their places. Template text may carry
// special tokens; user-supplied text must not be parsed for them.
int tokenize_chat(kiln_llama * h, const common_chat_prompt & prompt, const std::vector<mtmd_bitmap *> & bmps,
                  mtmd::input_chunks_ptr & chunks) {
    const std::string & marker = private_marker();
    struct seg { std::string text; bool special; int image; };  // image: the one that follows the text, or -1
    std::vector<seg> segs;
    int next = 0;
    for (const auto & part : prompt.parts) {
        size_t start = 0, at;
        while ((at = part.text.find(marker, start)) != std::string::npos) {
            segs.push_back({part.text.substr(start, at - start), !part.is_input, next++});
            start = at + marker.size();
        }
        segs.push_back({part.text.substr(start), !part.is_input, -1});
    }
    if (next != (int) bmps.size())
        return fail("the chat template placed " + std::to_string(next) + " of " + std::to_string(bmps.size()) + " images", 9);
    std::vector<mtmd_input_text> texts(segs.size());
    std::vector<mtmd_input_part> parts;
    for (size_t i = 0; i < segs.size(); i++) {
        texts[i] = {segs[i].text.data(), segs[i].text.size(), false, segs[i].special};
        if (!segs[i].text.empty()) parts.push_back({&texts[i], nullptr});
        if (segs[i].image >= 0) parts.push_back({nullptr, bmps[segs[i].image]});
    }
    std::vector<const mtmd_input_part *> ptrs;
    for (const auto & p : parts) ptrs.push_back(&p);
    chunks.reset(mtmd_input_chunks_init());
    if (int r = mtmd_tokenize_from_parts(h->mctx, chunks.get(), ptrs.data(), ptrs.size(), true); r != 0)
        return fail("tokenizing the prompt with images failed: " + std::to_string(r), 10);
    return 0;
}

// Runs a prompt with images through the model from position 0 (images through the vision encoder, then everything
// through the language model), leaving the logits of its last token. *out_pos is the position after it, and
// *out_n_prompt the number of tokens (context cells) it took.
int eval_chunks(kiln_llama * h, const mtmd_input_chunks * chunks, int * out_pos, int * out_n_prompt) {
    const int n_ctx = (int) llama_n_ctx(h->ctx);
    const int n = (int) mtmd_helper_get_n_tokens(chunks);
    if (n >= n_ctx) return fail("prompt (" + std::to_string(n) + " tokens) exceeds context " + std::to_string(n_ctx), 2);
    llama_memory_clear(llama_get_memory(h->ctx), true);
    llama_pos new_pos = 0;
    if (int r = mtmd_helper_eval_chunks(h->mctx, h->ctx, chunks, 0, 0, (int) llama_n_batch(h->ctx), true, &new_pos); r != 0)
        return fail("evaluating the prompt and images failed: " + std::to_string(r), 8);
    *out_pos = (int) new_pos;
    *out_n_prompt = n;
    return 0;
}

// kiln_llama_chat_json, with the images that {"type": "image", "index": i} parts refer to.
int chat_json_impl(kiln_llama * h, const char * request_json, const std::vector<mtmd_bitmap *> & given,
                   kiln_llama_token_cb cb, void * user, char ** out_response_json) {
    *out_response_json = nullptr;
    common_sampler * smpl = nullptr;
    try {
        common_json req = common_json::parse(request_json);
        request_images images;
        if (req.contains("messages")) take_images(h, req.at("messages"), given, images);
        const bool with_images = !images.all.empty();

        common_chat_templates_inputs in;
        in.messages = common_chat_msgs_parse_oaicompat(req.at("messages"));
        if (req.contains("tools") && !req.at("tools").is_null()) in.tools = common_chat_tools_parse_oaicompat(req.at("tools"));
        if (req.contains("tool_choice") && req.at("tool_choice").is_string())
            in.tool_choice = common_chat_tool_choice_parse_oaicompat(req.at("tool_choice").get<std::string>());
        in.parallel_tool_calls = req.value("parallel_tool_calls", true);
        in.enable_thinking     = req.value("enable_thinking", true);
        in.reasoning_format    = COMMON_REASONING_FORMAT_DEEPSEEK;  // thinking goes to reasoning_content
        if (req.contains("json_schema") && !req.at("json_schema").is_null()) in.json_schema = req.at("json_schema").dump();

        common_chat_session session(h->tmpls.get(), h->vocab, in);

        // Template text may carry special tokens; user-supplied text must not be parsed for them.
        std::vector<llama_token> prompt;
        mtmd::input_chunks_ptr chunks;
        int n_prompt = 0;
        const int n_ctx = (int) llama_n_ctx(h->ctx);
        if (with_images) {
            if (int rc = tokenize_chat(h, session.prompt(), images.all, chunks); rc != 0) return rc;
            n_prompt = (int) mtmd_helper_get_n_tokens(chunks.get());
        } else {
            bool first = true;
            for (const auto & part : session.prompt().parts) {
                auto t = common_tokenize(h->vocab, part.text, first, !part.is_input);
                prompt.insert(prompt.end(), t.begin(), t.end());
                first = false;
            }
            n_prompt = (int) prompt.size();
        }
        if (n_prompt >= n_ctx)
            return fail("prompt (" + std::to_string(n_prompt) + " tokens) exceeds context " + std::to_string(n_ctx), 2);

        common_params_sampling sp;
        sp.temp  = (float) req.value("temperature", 0.7);
        sp.top_p = (float) req.value("top_p", 0.95);
        sp.min_p = (float) req.value("min_p", 0.05);
        sp.top_k = req.value("top_k", 40);
        int64_t seed = req.value("seed", (int64_t) -1);
        sp.seed = seed < 0 ? LLAMA_DEFAULT_SEED : (uint32_t) seed;
        session.apply_sampling(sp);
        smpl = common_sampler_init(h->model, sp);
        if (!smpl) return fail("failed to initialise the sampler (bad grammar?)", 6);

        const int max_tokens = req.value("max_tokens", -1);
        llama_memory_clear(llama_get_memory(h->ctx), true);
        std::string finish = "stop", raw, sent_content;
        int pos = 0, generated = 0, cells = 0;  // cells: of the context used (more than pos after an M-RoPE image)
        std::vector<llama_token> cur = prompt;  // empty when the prompt went in with its images
        if (with_images) {
            if (int rc = eval_chunks(h, chunks.get(), &pos, &cells); rc != 0) {
                common_sampler_free(smpl);
                return rc;
            }
        }
        while (true) {
            if (max_tokens > 0 && generated >= max_tokens) { finish = "length"; break; }
            if (!cur.empty()) {
                if (cells + (int) cur.size() > n_ctx) { finish = "length"; break; }
                if (int r = decode(h, cur.data(), (int) cur.size(), pos); r != 0) {
                    common_sampler_free(smpl);
                    return fail("llama_process failed: " + std::to_string(r), 3);
                }
                pos += (int) cur.size();
                cells += (int) cur.size();
            }
            llama_token tok = common_sampler_sample(smpl, h->ctx, -1);
            common_sampler_accept(smpl, tok, true);
            if (llama_vocab_is_eog(h->vocab, tok)) break;
            generated++;
            std::string piece = common_token_to_piece(h->vocab, tok, true);
            raw += piece;
            common_chat_input chunk;
            chunk.append(piece, tok);
            const common_chat_msg & m = session.feed(chunk);
            if (cb && m.content.size() > sent_content.size() && m.content.compare(0, sent_content.size(), sent_content) == 0) {
                std::string delta = m.content.substr(sent_content.size());
                sent_content = m.content;
                if (cb(delta.c_str(), user) != 0) break;
            }
            bool stop = false;
            for (const auto & s : session.additional_stops())
                if (!s.empty() && raw.size() >= s.size() && raw.compare(raw.size() - s.size(), s.size(), s) == 0) stop = true;
            if (stop) break;
            cur.assign(1, tok);
        }
        common_sampler_free(smpl);
        smpl = nullptr;

        common_chat_msg msg = session.finish();
        msg.role = "assistant";
        for (size_t i = 0; i < msg.tool_calls.size(); i++)
            if (msg.tool_calls[i].id.empty()) msg.tool_calls[i].id = "call_" + std::to_string(i);
        if (!msg.tool_calls.empty() && finish == "stop") finish = "tool_calls";

        common_json res = common_json::object();
        res["message"] = msg.to_json_oaicompat();
        res["finish_reason"] = finish;
        common_json usage = common_json::object();
        usage["prompt_tokens"] = (int64_t) n_prompt;
        usage["completion_tokens"] = (int64_t) generated;
        res["usage"] = usage;
        *out_response_json = dup(res.dump());
        return 0;
    } catch (const std::exception & e) {
        if (smpl) common_sampler_free(smpl);
        return fail(std::string("chat: ") + e.what(), 7);
    }
}

}  // namespace

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
    try {
        h->tmpls = common_chat_templates_init(h->model, "");
    } catch (const std::exception & e) {
        g_err = std::string("chat template: ") + e.what();
        kiln_llama_destroy(h);
        return nullptr;
    }
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

    llama_sampler * smpl = simple_sampler(temperature, top_p, min_p, seed);

    std::string out;
    int pos = 0, generated = 0, rc = 0;
    const llama_token * cur = toks.data();
    int n_cur = n;
    llama_token tok;
    while (max_tokens <= 0 || generated < max_tokens) {
        if (pos + n_cur > n_ctx) break;  // context full: return what we have
        if (int r = decode(h, cur, n_cur, pos); r != 0) {
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

int kiln_llama_chat_json(kiln_llama * h, const char * request_json, kiln_llama_token_cb cb, void * user,
                         char ** out_response_json) {
    return chat_json_impl(h, request_json, {}, cb, user, out_response_json);
}

kiln_llama * kiln_llama_create_mm(const char * model_path, const char * mmproj_path, int n_gpu_layers, int n_ctx) {
    kiln_llama * h = kiln_llama_create(model_path, n_gpu_layers, n_ctx);
    if (!h) return nullptr;
    if (!getenv("KILN_VERBOSE")) {
        mtmd_helper_log_set([](ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
        }, nullptr);
    }
    mtmd_context_params mp = mtmd_context_params_default();
    mp.use_gpu       = n_gpu_layers != 0;
    mp.print_timings = getenv("KILN_VERBOSE") != nullptr;
    mp.warmup        = false;
    try {
        // As llama-mtmd-cli does: a model whose image attention is non-causal needs the whole image in one ubatch.
        const int n_ubatch = (int) llama_n_ubatch(h->ctx);
        auto mem = mtmd_get_memory_usage(mmproj_path, mp);
        if (mem.use_non_causal && mem.image_max_tokens > n_ubatch) {
            mp.image_max_tokens = n_ubatch;
            mp.image_min_tokens = std::min(mp.image_min_tokens, n_ubatch);
        }
    } catch (const std::exception &) {}  // mtmd_init_from_file reports a bad mmproj below
    h->mctx = mtmd_init_from_file(mmproj_path, h->model, mp);
    if (!h->mctx) { g_err = std::string("unable to load mmproj: ") + mmproj_path; kiln_llama_destroy(h); return nullptr; }
    return h;
}

int kiln_llama_supports_vision(kiln_llama * h) { return h->mctx && mtmd_support_vision(h->mctx) ? 1 : 0; }

const char * kiln_llama_media_marker(void) { return mtmd_default_marker(); }

int kiln_llama_chat_images(kiln_llama * h, const char ** roles, const char ** contents, const int * image_counts,
                           int n_messages, const unsigned char ** rgb, const uint32_t * widths, const uint32_t * heights,
                           int n_images, int max_tokens, float temperature, float top_p, float min_p, uint32_t seed,
                           kiln_llama_token_cb cb, void * user, char ** out_text, int * out_n_prompt, int * out_n_generated) {
    *out_text = nullptr;
    char * res = nullptr;
    int rc;
    try {
        auto bmps = rgb_bitmaps(rgb, widths, heights, n_images);
        common_json msgs = common_json::array();
        int next = 0;
        for (int i = 0; i < n_messages; i++) {
            int k = image_counts ? image_counts[i] : 0;
            if (k <= 0) { msgs.push_back({{"role", roles[i]}, {"content", contents[i]}}); continue; }
            common_json parts = common_json::array();
            for (int j = 0; j < k; j++) parts.push_back({{"type", "image"}, {"index", next++}});
            if (contents[i] && *contents[i]) parts.push_back({{"type", "text"}, {"text", contents[i]}});
            msgs.push_back({{"role", roles[i]}, {"content", parts}});
        }
        if (next != n_images) return fail(std::to_string(n_images) + " images given, messages use " + std::to_string(next), 11);
        // the same sampling as kiln_llama_chat: no top-k; temperature <= 0 is greedy
        common_json req = {{"messages", msgs}, {"max_tokens", max_tokens}, {"temperature", temperature > 0 ? temperature : 0.0f},
                           {"top_p", top_p}, {"min_p", min_p}, {"top_k", 0},
                           {"seed", seed == 0xFFFFFFFFu ? (int64_t) -1 : (int64_t) seed}};
        rc = chat_json_impl(h, req.dump().c_str(), raw(bmps), cb, user, &res);
        if (rc != 0) return rc;
        std::string json(res);
        free(res);
        common_json out = common_json::parse(json);
        const auto & content = out.at("message").at("content");
        *out_text = dup(content.is_string() ? content.get<std::string>() : "");
        if (out_n_prompt) *out_n_prompt = out.at("usage").at("prompt_tokens").get<int>();
        if (out_n_generated) *out_n_generated = out.at("usage").at("completion_tokens").get<int>();
        return 0;
    } catch (const std::exception & e) {
        return fail(std::string("chat: ") + e.what(), 7);
    }
}

int kiln_llama_complete_images(kiln_llama * h, const char * prompt, const unsigned char ** rgb, const uint32_t * widths,
                               const uint32_t * heights, int n_images, int max_tokens, float temperature, float top_p,
                               float min_p, uint32_t seed, kiln_llama_token_cb cb, void * user, char ** out_text,
                               int * out_n_prompt, int * out_n_generated) {
    *out_text = nullptr;
    if (!h->mctx || !mtmd_support_vision(h->mctx)) return fail("this model takes no images (load it with its mmproj)", 12);
    std::vector<mtmd::bitmap_ptr> bmps;
    try { bmps = rgb_bitmaps(rgb, widths, heights, n_images); }
    catch (const std::exception & e) { return fail(e.what(), 11); }
    // Counted here: mtmd_tokenize reports a marker mismatch as 2, the same as a failed image, despite its header.
    const std::string marker = mtmd_default_marker(), text_in = prompt;
    int n_markers = 0;
    for (size_t at = text_in.find(marker); at != std::string::npos; at = text_in.find(marker, at + marker.size())) n_markers++;
    if (n_markers != n_images)
        return fail("the prompt has " + std::to_string(n_markers) + " image markers for " + std::to_string(n_images) + " images", 9);
    auto ptrs = raw(bmps);
    std::vector<const mtmd_bitmap *> cptrs(ptrs.begin(), ptrs.end());
    mtmd_input_text text = {prompt, strlen(prompt), true, true};
    mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
    int r = mtmd_tokenize(h->mctx, chunks.get(), &text, cptrs.data(), cptrs.size());
    if (r != 0) return fail("tokenizing the prompt with images failed: " + std::to_string(r), 10);
    int pos = 0, n_prompt = 0;
    if (int rc = eval_chunks(h, chunks.get(), &pos, &n_prompt); rc != 0) return rc;
    if (out_n_prompt) *out_n_prompt = n_prompt;
    llama_sampler * smpl = simple_sampler(temperature, top_p, min_p, seed);
    std::string out;
    int generated = 0;
    int rc = generate(h, smpl, pos, n_prompt, max_tokens, cb, user, out, generated);
    llama_sampler_free(smpl);
    if (out_n_generated) *out_n_generated = generated;
    *out_text = dup(out);
    return rc;
}

int kiln_llama_chat_json_images(kiln_llama * h, const char * request_json, const unsigned char ** rgb,
                                const uint32_t * widths, const uint32_t * heights, int n_images,
                                kiln_llama_token_cb cb, void * user, char ** out_response_json) {
    *out_response_json = nullptr;
    std::vector<mtmd::bitmap_ptr> bmps;
    try { bmps = rgb_bitmaps(rgb, widths, heights, n_images); }
    catch (const std::exception & e) { return fail(e.what(), 11); }
    return chat_json_impl(h, request_json, raw(bmps), cb, user, out_response_json);
}

void kiln_llama_free(void * p) { free(p); }

void kiln_llama_destroy(kiln_llama * h) {
    if (!h) return;
    h->tmpls.reset();
    if (h->mctx) mtmd_free(h->mctx);
    if (h->batch) llama_batch_ext_free(h->batch);
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    delete h;
}

}  // extern "C"
