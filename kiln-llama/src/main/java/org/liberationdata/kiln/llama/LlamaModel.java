package org.liberationdata.kiln.llama;

import org.liberationdata.kiln.Call;
import org.liberationdata.kiln.HfCache;
import org.liberationdata.kiln.ModelThread;
import org.liberationdata.kiln.NativeLibrary;
import org.liberationdata.kiln.Scored;
import org.liberationdata.kiln.StreamOverflowException;
import org.liberationdata.kiln.Vectors;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Flow;
import java.util.function.Consumer;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * A GGUF language model running in-process through llama.cpp (via libkiln_llama).
 * Loaded with an mmproj GGUF as well, it is a vision language model: user messages can carry {@link Image}s (through
 * llama.cpp's mtmd, as llama-mtmd-cli and llama-server run them). See {@link Weights} for the models Kiln knows by name.
 * Weights stay loaded until {@link #close()}.
 *
 * <p>Loaded in {@link Mode#EMBED} or {@link Mode#RERANK} instead, it is an embedding model ({@link #embed}) or a
 * reranker ({@link #rerank}), through llama.cpp's pooled embeddings output, as llama-embedding and llama-server run
 * them; {@link Vectors} searches the embeddings.
 *
 * <p>Threads: calls run one at a time, in arrival order, from any thread. The blocking methods ({@link #chat} and the
 * rest) run on the calling thread, callback included. The {@code ...Async} methods run on this model's own platform
 * thread and return a {@link CompletableFuture}, so a virtual thread awaiting one doesn't pin its carrier; streaming
 * goes to a {@link Flow.Subscriber}. {@link #cancel()} and {@link #close()} are safe from any thread.
 */
public final class LlamaModel implements AutoCloseable {
    private static final NativeLibrary LIB = NativeLibrary.load("kiln_llama");
    private static final MethodHandle LAST_ERROR = LIB.function("kiln_llama_last_error", FunctionDescriptor.of(ADDRESS));
    private static final MethodHandle CREATE = LIB.function("kiln_llama_create",
            FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT));
    private static final MethodHandle CHAT = LIB.function("kiln_llama_chat", FunctionDescriptor.of(JAVA_INT,
            ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_FLOAT, JAVA_FLOAT, JAVA_FLOAT, JAVA_INT,
            ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle COMPLETE = LIB.function("kiln_llama_complete", FunctionDescriptor.of(JAVA_INT,
            ADDRESS, ADDRESS, JAVA_INT, JAVA_FLOAT, JAVA_FLOAT, JAVA_FLOAT, JAVA_INT,
            ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle CHAT_JSON = LIB.function("kiln_llama_chat_json", FunctionDescriptor.of(JAVA_INT,
            ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle CREATE_MM = LIB.function("kiln_llama_create_mm",
            FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT));
    private static final MethodHandle SUPPORTS_VISION = LIB.function("kiln_llama_supports_vision",
            FunctionDescriptor.of(JAVA_INT, ADDRESS));
    private static final MethodHandle MEDIA_MARKER = LIB.function("kiln_llama_media_marker", FunctionDescriptor.of(ADDRESS));
    private static final MethodHandle CHAT_IMAGES = LIB.function("kiln_llama_chat_images", FunctionDescriptor.of(JAVA_INT,
            ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT,
            JAVA_INT, JAVA_FLOAT, JAVA_FLOAT, JAVA_FLOAT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle COMPLETE_IMAGES = LIB.function("kiln_llama_complete_images", FunctionDescriptor.of(JAVA_INT,
            ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_FLOAT, JAVA_FLOAT, JAVA_FLOAT, JAVA_INT,
            ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle CHAT_JSON_IMAGES = LIB.function("kiln_llama_chat_json_images", FunctionDescriptor.of(JAVA_INT,
            ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle CREATE_POOLED = LIB.function("kiln_llama_create_pooled",
            FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT));
    private static final MethodHandle N_EMBD = LIB.function("kiln_llama_n_embd", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    private static final MethodHandle EMBED = LIB.function("kiln_llama_embed", FunctionDescriptor.of(JAVA_INT,
            ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle RERANK = LIB.function("kiln_llama_rerank", FunctionDescriptor.of(JAVA_INT,
            ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    /** The shim's return code for a pooled call that cancel() stopped between batches. */
    private static final int CANCELLED = 13;
    private static final MethodHandle FREE = LIB.function("kiln_llama_free", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle DESTROY = LIB.function("kiln_llama_destroy", FunctionDescriptor.ofVoid(ADDRESS));

    private static final FunctionDescriptor TOKEN_CB = FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS);
    private static final MethodHandle ON_TOKEN;
    static {
        try {
            ON_TOKEN = MethodHandles.lookup().findStatic(LlamaModel.class, "onToken",
                    MethodType.methodType(int.class, MemorySegment.class, MemorySegment.class));
        } catch (ReflectiveOperationException e) { throw new ExceptionInInitializerError(e); }
    }
    private record Sink(Call call, Consumer<String> consumer) {}

    private static final ThreadLocal<Sink> SINK = new ThreadLocal<>();

    /**
     * The token upcall. Nothing may escape it: an exception thrown into llama.cpp's frames kills the JVM. So a
     * consumer's exception is recorded in the call, generation stops, and the Java method rethrows it on return.
     * An empty piece is the shim polling for a stop while no content is streaming (reasoning, tool calls, and
     * between embed and rerank batches).
     */
    private static int onToken(MemorySegment piece, MemorySegment user) {
        Sink sink = SINK.get();
        if (sink == null) return 0;
        try {
            if (sink.consumer() != null) {
                String s = NativeLibrary.string(piece);
                if (s != null && !s.isEmpty()) sink.consumer().accept(s);
            }
        } catch (Throwable t) {
            sink.call().fail(t);
        }
        return sink.call().stopRequested() ? 1 : 0;
    }

    /**
     * Stops the generation in progress (from any thread, including its own callback) after the current token; the
     * call returns what was generated so far ({@link #extract} throws a {@link CancellationException} instead, and so
     * do {@link #embed} and {@link #rerank}, which stop between batches). Works
     * for every call, streaming or not, blocking or async. A call that hasn't started yet
     * isn't affected: cancel its future instead.
     */
    public void cancel() { thread.cancel(); }

    /**
     * Where streamed pieces are delivered ({@link #chatAsync(List, Options, Flow.Subscriber)} and the like), and how
     * many a subscriber may fall behind before the generation stops with a {@link StreamOverflowException}.
     * By default: a virtual thread, and {@value ModelThread#DEFAULT_BUFFER} pieces.
     */
    public LlamaModel streamOn(Executor executor, int bufferSize) {
        thread.streamOn(executor, bufferSize);
        return this;
    }

    /** A chat message. A user message can carry images (for a model loaded with an mmproj); they go before its text. */
    public record Message(String role, String content, List<Image> images) {
        public Message {
            images = images == null ? List.of() : List.copyOf(images);
        }

        public Message(String role, String content) { this(role, content, List.of()); }

        public static Message system(String c) { return new Message("system", c); }
        public static Message user(String c) { return new Message("user", c); }
        /** A user message with images, for example {@code Message.user("What colour is the square?", Image.of(img))}. */
        public static Message user(String c, Image... images) { return new Message("user", c, List.of(images)); }
        public static Message assistant(String c) { return new Message("assistant", c); }
    }

    /**
     * An image for a vision language model, as packed RGB (3 bytes a pixel, row by row); alpha is dropped. The model's
     * mmproj resizes it to what its vision encoder takes, so there's no need to scale it first.
     */
    public record Image(int width, int height, byte[] rgb) {
        public Image {
            if (width <= 0 || height <= 0) throw new IllegalArgumentException("empty image: " + width + "x" + height);
            if ((long) width * height * 3 != rgb.length)
                throw new IllegalArgumentException("rgb has " + rgb.length + " bytes; " + width + "x" + height + " needs " + 3L * width * height);
        }

        /** From a Java2D image (any type). */
        public static Image of(BufferedImage img) {
            int w = img.getWidth(), h = img.getHeight();
            byte[] rgb = new byte[w * h * 3];
            int[] row = new int[w];
            for (int y = 0, o = 0; y < h; y++) {
                img.getRGB(0, y, w, 1, row, 0, w);
                for (int x = 0; x < w; x++) {
                    int p = row[x];
                    rgb[o++] = (byte) (p >> 16);
                    rgb[o++] = (byte) (p >> 8);
                    rgb[o++] = (byte) p;
                }
            }
            return new Image(w, h, rgb);
        }

        /** From encoded image bytes: whatever ImageIO reads (PNG, JPEG, BMP, GIF, ...). */
        public static Image decode(byte[] encoded) {
            try {
                BufferedImage img = ImageIO.read(new ByteArrayInputStream(encoded));
                if (img == null) throw new IllegalArgumentException("not an image ImageIO can read");
                return of(img);
            } catch (IOException e) { throw new UncheckedIOException(e); }
        }

        /** From an image file (PNG, JPEG, BMP, GIF, ...). */
        public static Image read(Path file) {
            try { return decode(Files.readAllBytes(file)); }
            catch (IOException e) { throw new UncheckedIOException(e); }
        }
    }

    /**
     * Weights on Hugging Face that Kiln knows by name. A vision language model has an {@link #mmprojFile} (its vision
     * encoder and projector); {@link LlamaModel#LlamaModel(Weights)} loads both. Fetch them with
     * {@link #downloadCommand()}; {@link #model()} and {@link #mmproj()} fail with the command for their own file
     * when it's missing.
     */
    public enum Weights {
        /** Qwen3 4B Instruct 2507, text only. Apache-2.0. */
        QWEN3_4B("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", null),
        /** Qwen3-VL 4B Instruct: image questions, documents, charts and grounding (boxes on a 0 to 1000 grid). Apache-2.0. */
        QWEN3_VL_4B("Qwen/Qwen3-VL-4B-Instruct-GGUF", "Qwen3VL-4B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"),
        /**
         * PaddleOCR-VL 1.6 (0.9B): document element recognition. It takes a task prompt rather than a question: "OCR:",
         * "Table Recognition:", "Formula Recognition:", "Chart Recognition:", "Seal Recognition:" or "Spotting:".
         * Use it greedy (temperature 0). Apache-2.0.
         */
        PADDLEOCR_VL("PaddlePaddle/PaddleOCR-VL-1.6-GGUF", "PaddleOCR-VL-1.6-GGUF.gguf", "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"),
        /**
         * Google MedGemma 1.5 4B (instruction tuned): medical images and text. Using it to diagnose or treat patients may need a health regulator's authorization (see the Health AI Developer Foundations terms).
         * Health AI Developer Foundations terms.
         */
        MEDGEMMA_4B("unsloth/medgemma-1.5-4b-it-GGUF", "medgemma-1.5-4b-it-Q4_K_M.gguf", "mmproj-F16.gguf"),
        /**
         * Qwen3-Embedding 0.6B (Q8_0): 1024-dimension text embeddings, 100+ languages; load it in {@link Mode#EMBED}.
         * Queries take an instruction ({@link EmbedOptions#query}); documents don't. Apache-2.0.
         */
        QWEN3_EMBEDDING_0_6B("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf", null),
        /** Qwen3-Reranker 0.6B (Q8_0): scores how well a document answers a query; load it in {@link Mode#RERANK}. Apache-2.0. */
        QWEN3_RERANKER_0_6B("ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF", "qwen3-reranker-0.6b-q8_0.gguf", null);

        public final String repo, modelFile, mmprojFile;

        Weights(String repo, String modelFile, String mmprojFile) {
            this.repo = repo; this.modelFile = modelFile; this.mmprojFile = mmprojFile;
        }

        /** Whether these weights take images (they come with an mmproj). */
        public boolean vision() { return mmprojFile != null; }

        public String model() { return HfCache.require(repo, modelFile).toString(); }

        /** The mmproj's path, or null for a text-only model. */
        public String mmproj() { return mmprojFile == null ? null : HfCache.require(repo, mmprojFile).toString(); }

        public boolean downloaded() {
            return HfCache.find(repo, modelFile).isPresent() && (mmprojFile == null || HfCache.find(repo, mmprojFile).isPresent());
        }

        /** The {@code hf download} command that fetches these weights. */
        public String downloadCommand() {
            return "hf download " + repo + " " + modelFile + (mmprojFile == null ? "" : " " + mmprojFile);
        }

        /** By name, ignoring case and '-' or '.' vs '_': "qwen3-vl-4b", "paddleocr-vl", "qwen3-embedding-0.6b". */
        public static Weights named(String name) {
            return valueOf(name.toUpperCase().replace('-', '_').replace('.', '_'));
        }
    }

    /** temperature <= 0 means greedy. seed -1 means random. maxTokens <= 0 means until end-of-turn or context full. */
    public record Options(int maxTokens, float temperature, float topP, float minP, int seed) {
        public static final Options DEFAULT = new Options(512, 0.7f, 0.9f, 0.05f, -1);
        public Options withSeed(int s) { return new Options(maxTokens, temperature, topP, minP, s); }
        public Options withMaxTokens(int n) { return new Options(n, temperature, topP, minP, seed); }
        public Options withTemperature(float t) { return new Options(maxTokens, t, topP, minP, seed); }
    }

    public record Result(String text, int promptTokens, int generatedTokens) {}

    /** What a model is loaded for. A model loaded for one can't do the others. */
    public enum Mode {
        /** Text generation: chat, chatJson, extract, complete (and images, with an mmproj). */
        CHAT,
        /** Embeddings: {@link #embed}, with the pooling type the GGUF sets (Qwen3-Embedding: the last token). */
        EMBED,
        /** Reranking: {@link #rerank}, for a reranker GGUF (rank pooling, with a classifier head). */
        RERANK
    }

    /**
     * How {@link #embed(List, EmbedOptions)} embeds. prefix (null for none) goes before each text, verbatim; normalize
     * scales each embedding to length 1, so that a dot product is the cosine similarity.
     */
    public record EmbedOptions(String prefix, boolean normalize) {
        public static final EmbedOptions DEFAULT = new EmbedOptions(null, true);

        /**
         * For a search query, Qwen3-Embedding's instruction format: {@code "Instruct: " + task + "\nQuery:"} before
         * the query, for example {@code query("Given a web search query, retrieve relevant passages that answer the
         * query")}. Embed the documents with {@link #DEFAULT}: in Qwen3-Embedding they take no instruction.
         */
        public static EmbedOptions query(String task) { return new EmbedOptions("Instruct: " + task + "\nQuery:", true); }

        public EmbedOptions withPrefix(String p) { return new EmbedOptions(p, normalize); }
        public EmbedOptions withNormalize(boolean n) { return new EmbedOptions(prefix, n); }
    }

    private final MemorySegment handle;
    private final boolean vision;
    private final Mode mode;
    private final int embeddingSize;  // 0 unless EMBED
    private final ModelThread thread;

    /** @param gpuLayers -1 offloads every layer. @param contextSize 0 uses the model's trained context. */
    public LlamaModel(String ggufPath, int gpuLayers, int contextSize) { this(ggufPath, null, gpuLayers, contextSize); }

    public LlamaModel(String ggufPath) { this(ggufPath, -1, 8192); }

    /**
     * A vision language model: the language model's GGUF plus its mmproj GGUF (the vision encoder and projector).
     * mmprojPath null loads the language model alone. The vision encoder runs on the GPU unless gpuLayers is 0.
     * @param gpuLayers -1 offloads every layer. @param contextSize 0 uses the model's trained context.
     */
    public LlamaModel(String ggufPath, String mmprojPath, int gpuLayers, int contextSize) {
        this(ggufPath, mmprojPath, Mode.CHAT, gpuLayers, contextSize);
    }

    /**
     * A model loaded for chat, embeddings or reranking (see {@link Mode}).
     * @param gpuLayers -1 offloads every layer. @param contextSize 0 uses the model's trained context. In EMBED and
     *        RERANK it is also the batch: the most tokens one text (or query and document) can have, and how many
     *        go through the model at once.
     */
    public LlamaModel(String ggufPath, int gpuLayers, int contextSize, Mode mode) {
        this(ggufPath, null, mode, gpuLayers, contextSize);
    }

    /** A model loaded for chat, embeddings or reranking, with 8192 tokens of context. */
    public LlamaModel(String ggufPath, Mode mode) { this(ggufPath, -1, 8192, mode); }

    private LlamaModel(String ggufPath, String mmprojPath, Mode mode, int gpuLayers, int contextSize) {
        this.mode = Objects.requireNonNull(mode, "mode");
        if (mode != Mode.CHAT && mmprojPath != null)
            throw new IllegalArgumentException("an mmproj is for chat with images; " + mode + " takes none");
        try (Arena a = Arena.ofConfined()) {
            if (mode != Mode.CHAT) {
                handle = (MemorySegment) CREATE_POOLED.invokeExact(a.allocateFrom(ggufPath), gpuLayers, contextSize,
                        mode == Mode.RERANK ? 1 : 0);
                if (handle.equals(MemorySegment.NULL)) throw new IllegalStateException("kiln_llama_create_pooled: " + error());
            } else {
                handle = mmprojPath == null
                        ? (MemorySegment) CREATE.invokeExact(a.allocateFrom(ggufPath), gpuLayers, contextSize)
                        : (MemorySegment) CREATE_MM.invokeExact(a.allocateFrom(ggufPath), a.allocateFrom(mmprojPath), gpuLayers, contextSize);
                if (handle.equals(MemorySegment.NULL))
                    throw new IllegalStateException((mmprojPath == null ? "kiln_llama_create: " : "kiln_llama_create_mm: ") + error());
            }
            vision = mmprojPath != null && (int) SUPPORTS_VISION.invokeExact(handle) != 0;
            embeddingSize = mode == Mode.EMBED ? (int) N_EMBD.invokeExact(handle) : 0;
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        MemorySegment h = handle;
        thread = new ModelThread(Path.of(ggufPath).getFileName().toString(), () -> {
            try { DESTROY.invokeExact(h); } catch (Throwable t) { throw new RuntimeException(t); }
        });
    }

    /** A vision language model and its mmproj, with 8192 tokens of context (images take their share of it). */
    public LlamaModel(String ggufPath, String mmprojPath) { this(ggufPath, mmprojPath, -1, 8192); }

    /** Weights Kiln knows by name, with their mmproj if they have one, and 8192 tokens of context. */
    public LlamaModel(Weights w) { this(w, -1, 8192); }

    public LlamaModel(Weights w, int gpuLayers, int contextSize) { this(w.model(), w.mmproj(), gpuLayers, contextSize); }

    /**
     * Weights Kiln knows by name, loaded for chat, embeddings or reranking, with 8192 tokens of context; for example
     * {@code new LlamaModel(Weights.QWEN3_EMBEDDING_0_6B, Mode.EMBED)}.
     */
    public LlamaModel(Weights w, Mode mode) { this(w, -1, 8192, mode); }

    public LlamaModel(Weights w, int gpuLayers, int contextSize, Mode mode) {
        this(w.model(), mode == Mode.CHAT ? w.mmproj() : null, mode, gpuLayers, contextSize);
    }

    /** Whether this model takes images: it was loaded with an mmproj that has a vision encoder. */
    public boolean supportsVision() { return vision; }

    /** What this model was loaded for. */
    public Mode mode() { return mode; }

    /** The length of one embedding (1024 for Qwen3-Embedding 0.6B). Only for a model loaded in {@link Mode#EMBED}. */
    public int embeddingSize() {
        requireMode(Mode.EMBED, "embed");
        return embeddingSize;
    }

    /** The marker {@link #complete(String, List, Options, Consumer)} replaces with an image (llama.cpp's default). */
    public static String mediaMarker() {
        try { return NativeLibrary.string((MemorySegment) MEDIA_MARKER.invokeExact()); }
        catch (Throwable t) { throw new RuntimeException(t); }
    }

    // ---- async: each call runs on this model's own platform thread (see ModelThread) ----
    //
    // A blocking call pins a virtual thread's carrier for the whole generation; these don't. Await the future from a
    // virtual thread and its carrier is free meanwhile. Cancelling the future (or orTimeout) stops the generation;
    // cancel() stops whichever call is running. Futures complete off the model's thread, so dependent stages never
    // hold up the next generation.

    /** {@link #chat(List, Options)}, on this model's thread. */
    public CompletableFuture<Result> chatAsync(List<Message> messages, Options o) {
        return thread.submit(c -> chatOn(c, messages, o, null));
    }

    /**
     * {@link #chat(List, Options, Consumer)}, on this model's thread, with each piece of text going to pieces through a
     * bounded buffer on the stream executor ({@link #streamOn}). A slow subscriber doesn't slow the generation; one
     * that falls a whole buffer behind stops it, and gets (as does the future) a {@link StreamOverflowException}.
     * Cancelling the subscription stops the generation, and the future completes with the text so far.
     */
    public CompletableFuture<Result> chatAsync(List<Message> messages, Options o, Flow.Subscriber<? super String> pieces) {
        return thread.<String, Result>submit((c, sink) -> chatOn(c, messages, o, sink), pieces);
    }

    /** A cold publisher of a chat reply's pieces: each subscription runs the chat once, as chatAsync with a subscriber does. */
    public Flow.Publisher<String> chatStream(List<Message> messages, Options o) {
        return thread.<String>publisher((c, sink) -> chatOn(c, messages, o, sink));
    }

    /** {@link #chatJson(String, Consumer)}, on this model's thread. */
    public CompletableFuture<String> chatJsonAsync(String requestJson) {
        return thread.submit(c -> chatJsonOn(c, requestJson, List.of(), null));
    }

    /** {@link #chatJson(String, List, Consumer)}, on this model's thread. */
    public CompletableFuture<String> chatJsonAsync(String requestJson, List<Image> images) {
        return thread.submit(c -> chatJsonOn(c, requestJson, images, null));
    }

    /** {@link #chatJson(String, List, Consumer)}, streaming the message's content to a subscriber as chatAsync does. */
    public CompletableFuture<String> chatJsonAsync(String requestJson, List<Image> images, Flow.Subscriber<? super String> content) {
        return thread.<String, String>submit((c, sink) -> chatJsonOn(c, requestJson, images, sink), content);
    }

    /** {@link #extract}, on this model's thread. */
    public CompletableFuture<String> extractAsync(List<Message> messages, String jsonSchema, Options o) {
        return thread.submit(c -> extractOn(c, messages, jsonSchema, o));
    }

    /** {@link #complete(String, List, Options, Consumer)}, on this model's thread. */
    public CompletableFuture<Result> completeAsync(String prompt, List<Image> images, Options o) {
        return thread.submit(c -> images.isEmpty() ? completeOn(c, prompt, o, null) : completeImages(c, prompt, images, o, null));
    }

    /** {@link #embed(List, EmbedOptions)}, on this model's thread. */
    public CompletableFuture<float[][]> embedAsync(List<String> texts, EmbedOptions o) {
        return thread.submit(c -> embedOn(c, texts, o));
    }

    /** {@link #rerank(String, List)}, on this model's thread. */
    public CompletableFuture<List<Scored>> rerankAsync(String query, List<String> documents) {
        return thread.submit(c -> rerankOn(c, query, documents));
    }

    // ---- blocking: on the calling thread ----

    public Result chat(List<Message> messages, Options o) { return chat(messages, o, null); }

    /**
     * Formats with the model's chat template and generates. onPiece (optional) streams text as it is produced.
     * Messages with images need a model loaded with its mmproj; such a chat is formatted with the model's own
     * (Jinja) chat template, as {@link #chatJson} does, and its prompt token count includes the image tokens.
     */
    public Result chat(List<Message> messages, Options o, Consumer<String> onPiece) {
        return thread.run(c -> chatOn(c, messages, o, onPiece));
    }

    private Result chatOn(Call call, List<Message> messages, Options o, Consumer<String> onPiece) {
        requireMode(Mode.CHAT, "chat");
        if (messages.stream().anyMatch(m -> !m.images().isEmpty())) return chatImages(call, messages, o, onPiece);
        try (Arena a = Arena.ofConfined()) {
            int n = messages.size();
            MemorySegment roles = a.allocate(ADDRESS, n), contents = a.allocate(ADDRESS, n);
            for (int i = 0; i < n; i++) {
                roles.setAtIndex(ADDRESS, i, a.allocateFrom(messages.get(i).role()));
                contents.setAtIndex(ADDRESS, i, a.allocateFrom(messages.get(i).content()));
            }
            Out out = new Out(a);
            MemorySegment cb = callback(a, call, onPiece);
            int rc = (int) CHAT.invokeExact(handle, roles, contents, n, o.maxTokens(), o.temperature(), o.topP(),
                    o.minP(), o.seed(), cb, MemorySegment.NULL, out.text, out.nPrompt, out.nGen);
            return out.result(rc, call);
        } catch (RuntimeException | Error e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { SINK.remove(); }
    }

    /**
     * OpenAI-style chat with tool calling, using the model's own chat template and tool-call format
     * (the same templates and parsers as llama-server). Request and response are JSON:
     * <pre>
     * request:  {"messages": [...], "tools": [...], "tool_choice": "auto", "max_tokens": 512, "temperature": 0.7, "seed": 1}
     * response: {"message": {"role": "assistant", "content": "...", "reasoning_content": "...", "tool_calls": [...]},
     *            "finish_reason": "stop|length|tool_calls", "usage": {"prompt_tokens": n, "completion_tokens": n}}
     * </pre>
     * Messages and tools use OpenAI's chat completions format. onContent (optional) streams the message content.
     */
    public String chatJson(String requestJson, Consumer<String> onContent) {
        return thread.run(c -> chatJsonOn(c, requestJson, List.of(), onContent));
    }

    private String chatJsonOn(Call call, String requestJson, List<Image> images, Consumer<String> onContent) {
        requireMode(Mode.CHAT, "chat");
        if (!images.isEmpty()) return chatJsonImages(call, requestJson, images, onContent);
        try (Arena a = Arena.ofConfined()) {
            MemorySegment out = a.allocate(ADDRESS);
            MemorySegment cb = callback(a, call, onContent);
            int rc = (int) CHAT_JSON.invokeExact(handle, a.allocateFrom(requestJson), cb, MemorySegment.NULL, out);
            return json(out, rc, call);
        } catch (RuntimeException | Error e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { SINK.remove(); }
    }

    private Result chatImages(Call call, List<Message> messages, Options o, Consumer<String> onPiece) {
        requireVision();
        try (Arena a = Arena.ofConfined()) {
            int n = messages.size();
            MemorySegment roles = a.allocate(ADDRESS, n), contents = a.allocate(ADDRESS, n), counts = a.allocate(JAVA_INT, n);
            List<Image> images = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                Message m = messages.get(i);
                roles.setAtIndex(ADDRESS, i, a.allocateFrom(m.role()));
                contents.setAtIndex(ADDRESS, i, a.allocateFrom(m.content() == null ? "" : m.content()));
                counts.setAtIndex(JAVA_INT, i, m.images().size());
                images.addAll(m.images());
            }
            NativeImages img = new NativeImages(a, images);
            Out out = new Out(a);
            MemorySegment cb = callback(a, call, onPiece);
            int rc = (int) CHAT_IMAGES.invokeExact(handle, roles, contents, counts, n, img.rgb, img.widths, img.heights, img.n,
                    o.maxTokens(), o.temperature(), o.topP(), o.minP(), o.seed(), cb, MemorySegment.NULL,
                    out.text, out.nPrompt, out.nGen);
            return out.result(rc, call);
        } catch (RuntimeException | Error e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { SINK.remove(); }
    }

    /**
     * {@link #chatJson(String, Consumer)} with images. In a message's content array, {@code {"type": "image", "index": i}}
     * stands for {@code images.get(i)}; an OpenAI {@code {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}}
     * part works too. Add {@code "json_schema"} to the request to constrain the reply to a JSON schema:
     * <pre>
     * {"messages": [{"role": "user", "content": [{"type": "image", "index": 0}, {"type": "text", "text": "Read this invoice."}]}],
     *  "json_schema": {"type": "object", "properties": {"total": {"type": "number"}}, "required": ["total"]},
     *  "temperature": 0}
     * </pre>
     */
    public String chatJson(String requestJson, List<Image> images, Consumer<String> onContent) {
        return thread.run(c -> chatJsonOn(c, requestJson, images, onContent));
    }

    private String chatJsonImages(Call call, String requestJson, List<Image> images, Consumer<String> onContent) {
        requireVision();
        try (Arena a = Arena.ofConfined()) {
            NativeImages img = new NativeImages(a, images);
            MemorySegment out = a.allocate(ADDRESS);
            MemorySegment cb = callback(a, call, onContent);
            int rc = (int) CHAT_JSON_IMAGES.invokeExact(handle, a.allocateFrom(requestJson), img.rgb, img.widths, img.heights,
                    img.n, cb, MemorySegment.NULL, out);
            return json(out, rc, call);
        } catch (RuntimeException | Error e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { SINK.remove(); }
    }

    /**
     * Structured extraction: the reply to messages (with or without images), constrained to a JSON schema, as JSON
     * text; for example an invoice's vendor, date, total and line items from a picture of it. Reasoning, for a model
     * that thinks, is left out. Parse the result with any JSON library, or {@link Json#parse}.
     * @param jsonSchema a JSON Schema document, such as {@code {"type": "object", "properties": {...}, "required": [...]}}
     * @throws IllegalStateException if the reply ran out of tokens before the JSON was complete
     * @throws CancellationException if {@link #cancel()} stopped it: unlike chat, the text so far is no use here,
     *         because it is JSON cut off part way (and the engine still reports it as finished)
     */
    public String extract(List<Message> messages, String jsonSchema, Options o) {
        return thread.run(c -> extractOn(c, messages, jsonSchema, o));
    }

    private String extractOn(Call call, List<Message> messages, String jsonSchema, Options o) {
        List<Image> images = new ArrayList<>();
        String request = ChatRequests.extract(messages, jsonSchema, o, images);
        Object response = Json.parse(chatJsonOn(call, request, images, null));
        if (call.stopRequested()) throw new CancellationException("extract was cancelled");
        Object finish = Json.get(response, "finish_reason");
        if (!"stop".equals(finish)) throw new IllegalStateException("the reply was cut off (" + finish + "); raise maxTokens");
        return Json.get(response, "message", "content") instanceof String s ? s : "";
    }

    /**
     * Raw completion of an already-formatted prompt with images: the prompt holds {@link #mediaMarker()} once for each
     * image, in order, where the model's own image tokens go. Needs a model loaded with its mmproj.
     */
    public Result complete(String prompt, List<Image> images, Options o, Consumer<String> onPiece) {
        return thread.run(c -> images.isEmpty() ? completeOn(c, prompt, o, onPiece) : completeImages(c, prompt, images, o, onPiece));
    }

    private Result completeImages(Call call, String prompt, List<Image> images, Options o, Consumer<String> onPiece) {
        requireMode(Mode.CHAT, "complete");
        requireVision();
        try (Arena a = Arena.ofConfined()) {
            NativeImages img = new NativeImages(a, images);
            Out out = new Out(a);
            MemorySegment cb = callback(a, call, onPiece);
            int rc = (int) COMPLETE_IMAGES.invokeExact(handle, a.allocateFrom(prompt), img.rgb, img.widths, img.heights, img.n,
                    o.maxTokens(), o.temperature(), o.topP(), o.minP(), o.seed(), cb, MemorySegment.NULL,
                    out.text, out.nPrompt, out.nGen);
            return out.result(rc, call);
        } catch (RuntimeException | Error e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { SINK.remove(); }
    }

    private void requireVision() {
        if (!vision) throw new IllegalStateException("this model takes no images: load it with its mmproj");
    }

    /** Images as the shim takes them: parallel arrays of RGB pointers, widths and heights. */
    private static final class NativeImages {
        final MemorySegment rgb, widths, heights;
        final int n;

        NativeImages(Arena a, List<Image> images) {
            n = images.size();
            rgb = a.allocate(ADDRESS, Math.max(1, n));
            widths = a.allocate(JAVA_INT, Math.max(1, n));
            heights = a.allocate(JAVA_INT, Math.max(1, n));
            for (int i = 0; i < n; i++) {
                Image img = images.get(i);
                rgb.setAtIndex(ADDRESS, i, a.allocateFrom(JAVA_BYTE, img.rgb()));
                widths.setAtIndex(JAVA_INT, i, img.width());
                heights.setAtIndex(JAVA_INT, i, img.height());
            }
        }
    }

    /** Raw completion of an already-formatted prompt. */
    public Result complete(String prompt, Options o, Consumer<String> onPiece) {
        return thread.run(c -> completeOn(c, prompt, o, onPiece));
    }

    private Result completeOn(Call call, String prompt, Options o, Consumer<String> onPiece) {
        requireMode(Mode.CHAT, "complete");
        try (Arena a = Arena.ofConfined()) {
            Out out = new Out(a);
            MemorySegment cb = callback(a, call, onPiece);
            int rc = (int) COMPLETE.invokeExact(handle, a.allocateFrom(prompt), o.maxTokens(), o.temperature(),
                    o.topP(), o.minP(), o.seed(), cb, MemorySegment.NULL, out.text, out.nPrompt, out.nGen);
            return out.result(rc, call);
        } catch (RuntimeException | Error e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { SINK.remove(); }
    }

    // ---- embeddings and reranking (Mode.EMBED, Mode.RERANK) ----

    /** {@link #embed(List, EmbedOptions)} with the defaults: no prefix, normalised. */
    public float[][] embed(List<String> texts) { return embed(texts, EmbedOptions.DEFAULT); }

    /** One text's embedding, normalised. */
    public float[] embed(String text) { return embed(List.of(text))[0]; }

    /**
     * Embeds texts, one vector of {@link #embeddingSize()} floats each, in order; L2-normalised unless
     * {@code o.normalize()} is false. Texts go through the model several at a time, each as its own sequence, as
     * llama-embedding runs them. Text isn't parsed for special tokens. A text longer than the context fails
     * (split it first). {@link #cancel()} stops it between batches, with a {@link CancellationException}.
     */
    public float[][] embed(List<String> texts, EmbedOptions o) {
        return thread.run(c -> embedOn(c, texts, o));
    }

    private float[][] embedOn(Call call, List<String> texts, EmbedOptions o) {
        requireMode(Mode.EMBED, "embed");
        int n = texts.size(), dim = embeddingSize;
        if (n == 0) return new float[0][];
        try (Arena a = Arena.ofConfined()) {
            MemorySegment in = strings(a, texts, o.prefix());
            MemorySegment out = a.allocate(JAVA_FLOAT, (long) n * dim);
            MemorySegment cb = callback(a, call, null);
            int rc = (int) EMBED.invokeExact(handle, in, n, o.normalize() ? 1 : 0, out, cb, MemorySegment.NULL);
            pooledResult(rc, call, "embed");
            float[][] v = new float[n][];
            for (int i = 0; i < n; i++) v[i] = out.asSlice((long) i * dim * Float.BYTES, (long) dim * Float.BYTES).toArray(JAVA_FLOAT);
            return v;
        } catch (RuntimeException | Error e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { SINK.remove(); }
    }

    /**
     * Scores how well each document answers the query, as llama-server's /rerank does (the GGUF's own rerank prompt),
     * best first; {@link Scored#index()} is the document's position in documents. For Qwen3-Reranker a score is the
     * probability, 0 to 1, that the document answers the query. {@link #cancel()} stops it between batches, with a
     * {@link CancellationException}.
     */
    public List<Scored> rerank(String query, List<String> documents) {
        return thread.run(c -> rerankOn(c, query, documents));
    }

    private List<Scored> rerankOn(Call call, String query, List<String> documents) {
        requireMode(Mode.RERANK, "rerank");
        int n = documents.size();
        if (n == 0) return List.of();
        try (Arena a = Arena.ofConfined()) {
            MemorySegment docs = strings(a, documents, null);
            MemorySegment out = a.allocate(JAVA_FLOAT, n);
            MemorySegment cb = callback(a, call, null);
            int rc = (int) RERANK.invokeExact(handle, a.allocateFrom(query), docs, n, out, cb, MemorySegment.NULL);
            pooledResult(rc, call, "rerank");
            return Vectors.rank(out.toArray(JAVA_FLOAT));
        } catch (RuntimeException | Error e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { SINK.remove(); }
    }

    /** Strings as a C array, each with prefix (if any) before it. */
    private static MemorySegment strings(Arena a, List<String> texts, String prefix) {
        MemorySegment arr = a.allocate(ADDRESS, texts.size());
        for (int i = 0; i < texts.size(); i++) {
            String t = Objects.requireNonNull(texts.get(i), "text " + i);
            arr.setAtIndex(ADDRESS, i, a.allocateFrom(prefix == null ? t : prefix + t));
        }
        return arr;
    }

    private static void pooledResult(int rc, Call call, String what) {
        call.rethrow();
        if (rc == CANCELLED && call.stopRequested()) throw new CancellationException(what + " was cancelled");
        if (rc != 0) throw new IllegalStateException("llama " + what + " failed (rc=" + rc + "): " + error());
    }

    private void requireMode(Mode needed, String what) {
        if (mode != needed)
            throw new IllegalStateException("this model was loaded for " + mode + ", so it can't " + what
                    + ": load it with Mode." + needed);
    }

    /** Always an upcall, even with no consumer, so that cancel() reaches every call. */
    private static MemorySegment callback(Arena a, Call call, Consumer<String> onPiece) {
        SINK.set(new Sink(call, onPiece));
        return NativeLibrary.linker().upcallStub(ON_TOKEN, TOKEN_CB, a);
    }

    /** A chatJson reply: freed, then the callback's exception if one threw, then the engine's error. */
    private static String json(MemorySegment out, int rc, Call call) throws Throwable {
        MemorySegment r = out.get(ADDRESS, 0);
        String json = NativeLibrary.string(r);
        if (!r.equals(MemorySegment.NULL)) FREE.invokeExact(r);
        call.rethrow();
        if (rc != 0) throw new IllegalStateException("llama chat failed (rc=" + rc + "): " + error());
        return json;
    }

    private static final class Out {
        final MemorySegment text, nPrompt, nGen;
        Out(Arena a) { text = a.allocate(ADDRESS); nPrompt = a.allocate(JAVA_INT); nGen = a.allocate(JAVA_INT); }

        Result result(int rc, Call call) throws Throwable {
            MemorySegment t = text.get(ADDRESS, 0);
            String s = NativeLibrary.string(t);
            if (!t.equals(MemorySegment.NULL)) FREE.invokeExact(t);
            call.rethrow();
            if (rc != 0) throw new IllegalStateException("llama generation failed (rc=" + rc + "): " + error());
            return new Result(s, nPrompt.get(JAVA_INT, 0), nGen.get(JAVA_INT, 0));
        }
    }

    private static String error() {
        try { return NativeLibrary.string((MemorySegment) LAST_ERROR.invokeExact()); }
        catch (Throwable t) { return t.toString(); }
    }

    /**
     * Frees the weights. Idempotent, and safe while a call is in flight on another thread: it cancels that call,
     * fails queued async calls with a CancellationException, waits for the call to return, then frees them.
     * @throws IllegalStateException if called from inside one of this model's own callbacks
     */
    @Override public void close() { thread.close(); }
}
