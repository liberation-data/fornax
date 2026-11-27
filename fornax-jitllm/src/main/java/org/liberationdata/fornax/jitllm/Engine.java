package org.liberationdata.fornax.jitllm;

import org.beehive.jitllm.api.CancellationToken;
import org.beehive.jitllm.api.ChatMessage;
import org.beehive.jitllm.api.ChatRole;
import org.beehive.jitllm.api.EmbeddingSession;
import org.beehive.jitllm.api.GenerationRequest;
import org.beehive.jitllm.api.GenerationResult;
import org.beehive.jitllm.api.GenerationSession;
import org.beehive.jitllm.api.LocalModel;
import org.beehive.jitllm.api.LocalModels;
import org.beehive.jitllm.api.ModelOptions;
import org.beehive.jitllm.api.TextGenerationModel;
import org.liberationdata.fornax.Call;
import org.liberationdata.fornax.llama.LlamaModel.Message;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;

/**
 * One jitllm model loaded in this JVM, and the session its calls run on. Calls come one at a time from the model's
 * {@code ModelThread}; {@link #cancel()} may come from any thread. Only this class touches jitllm's API, so a JVM that
 * runs the model in a worker never loads it.
 */
final class Engine implements AutoCloseable {
    private final LocalModel model;
    private final GenerationSession session;      // chat, or null
    private final EmbeddingSession embeddings;    // embed, or null
    private volatile CancellationToken running;   // the call in progress, or null

    Engine(Path gguf, int contextSize, boolean embed) {
        ModelOptions.Builder options = ModelOptions.builder();
        if (contextSize > 0) options.contextLength(contextSize);
        try { model = LocalModels.load(gguf, options.build()); }
        catch (IOException e) { throw new UncheckedIOException(gguf + ": " + e.getMessage(), e); }
        try {
            if (!(model instanceof TextGenerationModel text))
                throw new IllegalArgumentException(gguf + " isn't a language model jitllm can generate with");
            session = embed ? null : text.newSession();
            embeddings = embed ? EmbeddingSession.open(text) : null;
        } catch (RuntimeException e) {
            model.close();
            throw e;
        }
    }

    int embeddingSize() { return embeddings == null ? 0 : embeddings.dimension(); }

    /** A chat, from an empty context: the request carries the whole conversation. */
    ChatJson.Reply chat(Call call, ChatJson.Request r, Consumer<String> onPiece) {
        CancellationToken token = new CancellationToken();
        running = token;
        try {
            if (call.stopRequested()) token.cancel();   // a cancel that came before the call started
            session.reset();
            GenerationRequest.Builder b = GenerationRequest.builder()
                    .messages(messages(r.messages()))
                    .maxNewTokens(r.maxTokens())
                    .temperature(Math.max(0f, r.temperature()))
                    .cancellation(token)
                    .onToken(piece -> {
                        if (call.stopRequested()) token.cancel();
                        if (onPiece != null && !piece.isEmpty()) onPiece.accept(piece);
                    });
            if (r.topP() > 0f && r.topP() <= 1f) b.topP(r.topP());
            if (r.seed() >= 0) b.seed(r.seed());
            if (!r.stop().isEmpty()) b.stopSequences(r.stop());
            GenerationResult g = session.generate(b.build());
            return new ChatJson.Reply(g.text(), g.promptTokens(), g.generatedTokens(), finishReason(g));
        } finally {
            running = null;
        }
    }

    float[][] embed(List<String> texts) {
        return embeddings.embedAll(texts).toArray(float[][]::new);
    }

    /** Stops the call in progress: jitllm ends it with the text so far. */
    void cancel() {
        CancellationToken t = running;
        if (t != null) t.cancel();
    }

    @Override public void close() {
        try {
            if (session != null) session.close();
            if (embeddings != null) embeddings.close();
        } finally {
            model.close();
        }
    }

    /** OpenAI's finish reasons: "length" when the reply was cut short, else "stop". */
    private static String finishReason(GenerationResult g) {
        return switch (g.finishReason()) {
            case MAX_TOKENS, CONTEXT_FULL -> "length";
            default -> "stop";
        };
    }

    private static List<ChatMessage> messages(List<Message> messages) {
        List<ChatMessage> out = new ArrayList<>(messages.size());
        for (Message m : messages) out.add(ChatMessage.of(role(m.role()), m.content() == null ? "" : m.content()));
        return out;
    }

    private static ChatRole role(String role) {
        return switch (role.toLowerCase(Locale.ROOT)) {
            case "system", "developer" -> ChatRole.SYSTEM;
            case "user" -> ChatRole.USER;
            case "assistant" -> ChatRole.ASSISTANT;
            default -> throw new IllegalArgumentException("jitllm takes system, user and assistant messages, not " + role);
        };
    }
}
