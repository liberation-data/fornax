package org.liberationdata.fornax.server;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.whisper.Transcript;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.function.Consumer;

/** Engines without models, for the server's routes: they reply from what each test sets, and record what they were asked. */
class StubEngines implements Engines {
    /** The chat reply's content, streamed piece by piece, and the rest of its message. */
    volatile List<String> pieces = List.of("Hel", "lo", "!");
    volatile String toolCalls;      // the message's tool_calls JSON, or null
    volatile String finish = "stop";
    volatile long pieceDelayMillis;
    /** Thrown by chat after this many pieces have been sent (a model failing mid-reply), or null for none. */
    volatile RuntimeException failure;
    volatile int failAfter;
    /** What chat was asked: the engine request JSON of each call. */
    final List<String> requests = new CopyOnWriteArrayList<>();
    final List<String> models = new CopyOnWriteArrayList<>();
    /** Counted down when a generation is stopped (a stop sequence, or a client that went away). */
    final CountDownLatch stopped = new CountDownLatch(1);
    volatile boolean closed;

    @Override public String chat(String model, String requestJson, boolean images, Consumer<String> onContent, Stopper stop) {
        checkChat(model, images);
        requests.add(requestJson);
        models.add(model);
        boolean[] cancelled = {false};
        stop.attach(() -> { cancelled[0] = true; stopped.countDown(); });
        StringBuilder sent = new StringBuilder();
        int done = 0;
        try {
            for (String p : pieces) {
                if (cancelled[0]) break;
                if (pieceDelayMillis > 0) Thread.sleep(pieceDelayMillis);
                if (failure != null && done++ == failAfter) throw failure;
                if (cancelled[0]) break;
                sent.append(p);
                if (onContent != null) onContent.accept(p);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            stop.detach();
        }
        return "{\"message\": {\"role\": \"assistant\", \"content\": " + org.liberationdata.fornax.Json.quote(sent.toString())
                + (toolCalls == null ? "" : ", \"tool_calls\": " + toolCalls) + "}, \"finish_reason\": \"" + finish
                + "\", \"usage\": {\"prompt_tokens\": 7, \"completion_tokens\": " + pieces.size() + "}}";
    }

    @Override public void checkChat(String model, boolean images) {
        if (model.equals("missing")) throw ApiError.noSuchModel("no model missing");
        if (images && !model.equals("vision")) throw ApiError.badRequest(model + " takes no images", "messages");
    }

    @Override public float[][] embed(String model, List<String> texts) {
        models.add(model);
        float[][] out = new float[texts.size()][];
        for (int i = 0; i < out.length; i++) out[i] = new float[]{0.6f, 0.8f, 0f, i};
        return out;
    }

    @Override public List<BufferedImage> images(String model, ImageJob job) {
        models.add(model);
        requests.add(job.toString());
        List<BufferedImage> out = new ArrayList<>();
        for (int i = 0; i < job.n(); i++) out.add(new BufferedImage(job.width(), job.height(), BufferedImage.TYPE_INT_RGB));
        return out;
    }

    @Override public Audio speak(String model, String text, String voice, float speed) {
        models.add(model);
        requests.add(text + "|" + voice + "|" + speed);
        return Audio.ofMono(new float[2400], 24000);
    }

    @Override public Transcript transcribe(String model, Audio audio, String language, String prompt) {
        models.add(model);
        requests.add(audio.sampleRate() + "|" + language + "|" + prompt);
        var hello = new Transcript.Word(0.0, 0.4, " Hello", 0.9f);
        var there = new Transcript.Word(0.4, 0.8, " there.", 0.9f);
        return new Transcript(" Hello there.", "en", List.of(new Transcript.Segment(0.0, 0.8, " Hello there.", List.of(hello, there))));
    }

    @Override public List<Listed> models() {
        return List.of(new Listed("qwen3-4b", "chat", true), new Listed("flux2-klein", "image", false));
    }

    @Override public String defaultModel(Use use) {
        return switch (use) {
            case CHAT -> "qwen3-4b";
            case EMBED -> "qwen3-embedding-0.6b";
            case SPEECH -> "kokoro";
            case TRANSCRIPTION -> "whisper-base.en";
            case IMAGE -> null;
        };
    }

    @Override public void close() { closed = true; }
}
