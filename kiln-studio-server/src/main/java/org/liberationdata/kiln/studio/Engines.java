package org.liberationdata.kiln.studio;

import org.liberationdata.kiln.Audio;
import org.liberationdata.kiln.acestep.AceStepModel;
import org.liberationdata.kiln.llama.LlamaModel;
import org.liberationdata.kiln.sd.DiffusionModel;

import java.awt.Color;
import java.awt.Font;
import java.awt.GradientPaint;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * The studio's view of each engine: what a pooled model can do, behind an interface, so an engine can be swapped for a
 * stub that does no work on the GPU ({@code -Dkiln.studio.stub=image,song}). The real ones are Kiln's own classes,
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

    /** A loaded song model. */
    interface Singer extends AutoCloseable {
        AceStepModel.Result sing(String requestJson);

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
            @Override public AceStepModel.Result sing(String requestJson) { return model.generate(requestJson); }

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
            @Override public AceStepModel.Result sing(String requestJson) {
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
                return new AceStepModel.Result(new Audio(l, r, rate), Json.write(req));
            }

            @Override public void close() {}
        };
    }

    private static void pause(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }
}
