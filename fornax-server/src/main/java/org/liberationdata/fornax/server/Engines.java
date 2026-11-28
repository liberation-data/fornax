package org.liberationdata.fornax.server;

import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.llama.LlamaModel.Embeddings;
import org.liberationdata.fornax.whisper.Transcript;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.function.Consumer;

/**
 * What the server's routes run, by model name: {@link PooledEngines} with Fornax's engines (and a {@code ModelPool}),
 * or a test's stand-in. A name the engine can't serve is an {@link ApiError}: a 404 for one it doesn't know, a 400
 * for one of the wrong kind.
 */
interface Engines extends AutoCloseable {

    /** What a model is loaded for, which is part of its key in the pool: one GGUF may be loaded for chat and embeddings. */
    enum Use { CHAT, EMBED, IMAGE, SPEECH, TRANSCRIPTION }

    /** One served model, for {@code GET /v1/models}. */
    record Listed(String id, String kind, boolean loaded) {}

    /**
     * An image request, checked. steps and cfg are null where the request leaves them to the model
     * ({@code ModelCatalog.imageSteps}, {@code imageCfg}).
     */
    record ImageJob(String prompt, int width, int height, Integer steps, Double cfg, long seed, int n) {}

    /**
     * {@code LlamaModel.chatJson}: the engine's response JSON for {@code requestJson}, with the content streamed to
     * {@code onContent} (null for none) as it is generated. {@code stop} is attached to the model while it runs.
     */
    String chat(String model, String requestJson, boolean images, Consumer<String> onContent, Stopper stop);

    /** Throws as {@link #chat} would for the model's name, before anything is loaded: for a stream's status. */
    void checkChat(String model, boolean images);

    /** One embedding per text, normalised, and the tokens they took. */
    Embeddings embed(String model, List<String> texts);

    List<BufferedImage> images(String model, ImageJob job);

    /** voice null for the model's own default; speed 0 for its own pace. */
    Audio speak(String model, String text, String voice, float speed);

    /** language null to detect it; prompt null for none. */
    Transcript transcribe(String model, Audio audio, String language, String prompt);

    /** The models this server would serve: those it was started with, those loaded, and what is downloaded. */
    List<Listed> models();

    /** The model a request for use that names none gets, or null. */
    String defaultModel(Use use);

    @Override void close();

    /**
     * Stops a running generation from another thread (a client that has gone) or from its own callback (a stop
     * sequence). A stop asked for before the model is attached takes effect when it is.
     */
    final class Stopper {
        private Runnable cancel;
        private boolean requested;

        synchronized void attach(Runnable cancel) {
            this.cancel = cancel;
            if (requested) cancel.run();
        }

        synchronized void detach() { cancel = null; }

        synchronized void stop() {
            requested = true;
            if (cancel != null) cancel.run();
        }

        synchronized boolean requested() { return requested; }
    }
}
