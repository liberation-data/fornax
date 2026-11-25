package org.liberationdata.kiln.studio;

import org.liberationdata.kiln.Audio;
import org.liberationdata.kiln.acestep.AceStepModel;
import org.liberationdata.kiln.llama.LlamaModel;
import org.liberationdata.kiln.sd.DiffusionModel;
import org.liberationdata.kiln.yue2.Yue2Model;

import java.awt.Color;
import java.awt.Font;
import java.awt.GradientPaint;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * The studio's view of each engine: what a pooled model can do, behind an interface, so an engine can be swapped for a
 * stub that does no work on the GPU ({@code -Dkiln.studio.stub=image,song,video}). The real ones are Kiln's own classes,
 * called as an application would call them.
 */
final class Engines {
    private Engines() {}

    /** A loaded chat model. */
    interface Chat extends AutoCloseable {
        LlamaModel.Result chat(List<LlamaModel.Message> messages, LlamaModel.Options options, Consumer<String> onPiece);

        String chatJson(String requestJson, Consumer<String> onContent);

        @Override void close();
    }

    /** A loaded image model. onStep gets (step, steps, seconds per step). */
    interface Painter extends AutoCloseable {
        BufferedImage paint(DiffusionModel.Request request, StepSink onStep);

        @Override void close();
    }

    interface StepSink { void step(int step, int steps, float seconds); }

    /** A loaded song model: ACE-Step or YuE2, each taking its engine's own JSON request. */
    interface Singer extends AutoCloseable {
        Song sing(String requestJson);

        @Override void close();
    }

    /** A song made: the audio, the request as the engine resolved it, and the ABC score YuE2 composes first (or null). */
    record Song(Audio audio, String resolvedRequestJson, String score) {}

    /** A loaded video model (LTX-2.5). onStep gets each sampling step, of the draft and then of the refine. */
    interface Filmmaker extends AutoCloseable {
        DiffusionModel.Video film(DiffusionModel.VideoRequest request, StepSink onStep);

        @Override void close();
    }

    // ---- the real engines ----

    static Chat llama(Catalog.Model m, int contextSize) {
        var llm = new LlamaModel(m.path("gguf").toString(), -1, contextSize);
        return new Chat() {
            @Override public LlamaModel.Result chat(List<LlamaModel.Message> messages, LlamaModel.Options o, Consumer<String> onPiece) {
                return llm.chat(messages, o, onPiece);
            }

            @Override public String chatJson(String requestJson, Consumer<String> onContent) { return llm.chatJson(requestJson, onContent); }

            @Override public void close() { llm.close(); }
        };
    }

    static Painter diffusion(Catalog.Model m) {
        var cfg = DiffusionModel.Config.builder()
                .diffusionModel(m.path("diffusion_model").toString())
                .vae(m.path("vae").toString())
                .llm(m.path("llm").toString())
                .build();
        var model = new DiffusionModel(cfg);
        return new Painter() {
            @Override public BufferedImage paint(DiffusionModel.Request r, StepSink onStep) {
                // stable-diffusion.cpp's progress callback is process-wide; the pool lets one request use a model at a time.
                DiffusionModel.onProgress(onStep::step);
                try { return model.generate(r).get(0); }
                finally { DiffusionModel.onProgress((s, n, t) -> {}); }
            }

            @Override public void close() { model.close(); }
        };
    }

    static Singer aceStep(Catalog.Model m) {
        var lm = m.path("lm");
        var model = new AceStepModel(lm == null ? null : lm.toString(), m.path("text_encoder").toString(),
                m.path("dit").toString(), m.path("vae").toString(), true);
        return new Singer() {
            @Override public Song sing(String requestJson) {
                var r = model.generate(requestJson);
                return new Song(r.audio(), r.resolvedRequestJson(), null);
            }

            @Override public void close() { model.close(); }
        };
    }

    /** As Yue2Song loads it: kept loaded, the engine's own sequence limit. */
    static Singer yue2(Catalog.Model m) {
        var model = new Yue2Model(m.path("model").toString(), m.path("vae").toString(), true, 0);
        return new Singer() {
            @Override public Song sing(String requestJson) {
                var r = model.generate(requestJson);
                return new Song(r.audio(), r.resolvedRequestJson(), r.abcScore());
            }

            @Override public void close() { model.close(); }
        };
    }

    // ComfyUI's int8 LTX keeps biases, norms and embedders in bf16; ggml's Vulkan and CPU backends can't add bf16 to f32,
    // so those (~0.9 GB) are widened to f32 at load, as VideoClip does (docs/engine-notes.md). The int8 linears are untouched.
    static final String LTX_BF16_TO_F32 = ".*\\.bias=f32,.*_norm\\.weight=f32,.*to_gate_logits\\.weight=f32,"
            + ".*adaln_single.*\\.weight=f32,.*(patchify_proj|proj_out)\\.weight=f32,"
            + ".*(learnable_registers|keyframes_abs_pos_embedding)=f32";

    /** LTX-2.5, as VideoClip loads it; without an audio_vae part it makes video alone. */
    static Filmmaker ltx(Catalog.Model m) {
        var audioVae = m.path("audio_vae");
        var cfg = DiffusionModel.Config.builder()
                .diffusionModel(m.path("diffusion_model").toString())
                .llm(m.path("llm").toString())
                .vae(m.path("vae").toString())
                .set("audio_vae_path", audioVae == null ? "" : audioVae.toString())
                .set("tensor_type_rules", LTX_BF16_TO_F32)
                .build();
        var model = new DiffusionModel(cfg);
        return new Filmmaker() {
            @Override public DiffusionModel.Video film(DiffusionModel.VideoRequest r, StepSink onStep) {
                DiffusionModel.onProgress(onStep::step);
                try { return model.generateVideo(r); }
                finally { DiffusionModel.onProgress((s, n, t) -> {}); }
            }

            @Override public void close() { model.close(); }
        };
    }

    // ---- stubs: the same calls, answered without loading a thing ----

    /** Echoes the last user message back a word at a time. */
    static Chat stubChat() {
        return new Chat() {
            @Override public LlamaModel.Result chat(List<LlamaModel.Message> messages, LlamaModel.Options o, Consumer<String> onPiece) {
                String last = messages.isEmpty() ? "" : messages.getLast().content();
                String text = "(stub) You said: " + last;
                int n = 0;
                for (String w : text.split("(?<= )")) {
                    pause(20);
                    if (onPiece != null) onPiece.accept(w);
                    n++;
                }
                return new LlamaModel.Result(text, last.length() / 4, n);
            }

            @Override public String chatJson(String requestJson, Consumer<String> onContent) {
                String text = "(stub) No tools were called.";
                if (onContent != null) onContent.accept(text);
                return Json.write(Json.obj("message", Json.obj("role", "assistant", "content", text),
                        "finish_reason", "stop", "usage", Json.obj("prompt_tokens", 0L, "completion_tokens", 6L)));
            }

            @Override public void close() {}
        };
    }

    /** Takes a moment per step, then paints a gradient with the prompt on it, marked as a stub. */
    static Painter stubPainter() {
        return new Painter() {
            @Override public BufferedImage paint(DiffusionModel.Request r, StepSink onStep) {
                for (int s = 1; s <= r.steps(); s++) {
                    pause(120);
                    onStep.step(s, r.steps(), 0.12f);
                }
                var img = new BufferedImage(r.width(), r.height(), BufferedImage.TYPE_INT_RGB);
                var g = img.createGraphics();
                g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
                float hue = Math.floorMod(r.seed(), 360) / 360f;
                g.setPaint(new GradientPaint(0, 0, Color.getHSBColor(hue, 0.65f, 0.95f), r.width(), r.height(),
                        Color.getHSBColor((hue + 0.12f) % 1f, 0.8f, 0.25f)));
                g.fillRect(0, 0, r.width(), r.height());
                g.setColor(new Color(255, 255, 255, 230));
                g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, Math.max(14, r.width() / 14)));
                g.drawString("STUB", r.width() / 16, r.height() / 6);
                g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, Math.max(10, r.width() / 32)));
                int y = r.height() / 6 + r.width() / 12, lineH = Math.max(12, r.width() / 26);
                StringBuilder line = new StringBuilder();
                for (String w : r.prompt().split("\\s+")) {
                    if (line.length() + w.length() > 34) {
                        g.drawString(line.toString(), r.width() / 16, y);
                        y += lineH;
                        line.setLength(0);
                    }
                    line.append(w).append(' ');
                }
                g.drawString(line.toString(), r.width() / 16, y);
                g.dispose();
                return img;
            }

            @Override public void close() {}
        };
    }

    /** A plain tune (a major arpeggio, as long as the request asks, up to 20 s), after a short pause. */
    static Singer stubSinger() {
        return new Singer() {
            @Override public Song sing(String requestJson) {
                Map<String, Object> req = Json.object(requestJson);
                double seconds = Math.min(20, Math.max(1, Json.dbl(req, "duration", 10)));
                pause(800);
                int rate = 48000, n = (int) (seconds * rate);
                float[] l = new float[n], r = new float[n];
                double[] notes = {261.63, 329.63, 392.00, 523.25, 392.00, 329.63};
                double beat = 60.0 / Math.max(40, Json.dbl(req, "bpm", 96));
                for (int i = 0; i < n; i++) {
                    double t = i / (double) rate, inBeat = t % beat;
                    double f = notes[(int) (t / beat) % notes.length];
                    double env = Math.exp(-3 * inBeat / beat);
                    float v = (float) (0.25 * env * Math.sin(2 * Math.PI * f * t));
                    l[i] = v;
                    r[i] = v * 0.8f;
                }
                req.put("stub", true);
                // YuE2's requests have a style (ACE-Step's a caption), and YuE2 writes a score before it sings.
                String score = req.containsKey("style") ? "X:1\nT:(stub)\nM:4/4\nL:1/8\nK:C\n|: C E G c G E :|\n" : null;
                return new Song(new Audio(l, r, rate), Json.write(req), score);
            }

            @Override public void close() {}
        };
    }

    /**
     * A step at a time, as the request's schedule has them (its sigmas, then the refine's), then frames of a colour that
     * turns over the clip, marked as a stub, at twice the size when the request refines; a tone under them with audio.
     */
    static Filmmaker stubFilmmaker(boolean audio) {
        return new Filmmaker() {
            @Override public DiffusionModel.Video film(DiffusionModel.VideoRequest r, StepSink onStep) {
                var x = r.extra();
                int draft = x.get("sample.sigmas") instanceof float[] s ? s.length - 1 : r.steps();
                boolean hires = Boolean.TRUE.equals(x.get("hires.enabled"));
                int refine = !hires ? 0 : x.get("hires.sigmas") instanceof float[] s ? s.length - 1 : 3;
                for (int s = 1; s <= draft; s++) { pause(60); onStep.step(s, draft, 0.06f); }
                for (int s = 1; s <= refine; s++) { pause(60); onStep.step(s, refine, 0.06f); }
                int w = hires ? r.width() * 2 : r.width(), h = hires ? r.height() * 2 : r.height();
                List<BufferedImage> frames = new ArrayList<>();
                for (int i = 0; i < r.frames(); i++) {
                    var img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
                    var g = img.createGraphics();
                    g.setColor(Color.getHSBColor((Math.floorMod(r.seed(), 360) / 360f + i / (float) r.frames()) % 1f, 0.6f, 0.8f));
                    g.fillRect(0, 0, w, h);
                    g.setColor(Color.WHITE);
                    g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, Math.max(14, w / 14)));
                    g.drawString("STUB " + (i + 1), w / 16, h / 4);
                    g.dispose();
                    frames.add(img);
                }
                Audio sound = null;
                if (audio) {
                    int rate = 48000, n = (int) ((double) r.frames() / r.fps() * rate);
                    float[] l = new float[n], rr = new float[n];
                    for (int i = 0; i < n; i++) l[i] = rr[i] = (float) (0.2 * Math.sin(2 * Math.PI * 440 * i / (double) rate));
                    sound = new Audio(l, rr, rate);
                }
                return new DiffusionModel.Video(frames, r.fps(), sound);
            }

            @Override public void close() {}
        };
    }

    private static void pause(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }
}
