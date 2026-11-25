package org.liberationdata.kiln.llama;

import org.liberationdata.kiln.HfCache;
import org.liberationdata.kiln.NativeLibrary;

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
import java.util.function.Consumer;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * A GGUF language model running in-process through llama.cpp (via libkiln_llama).
 * Loaded with an mmproj GGUF as well, it is a vision language model: user messages can carry {@link Image}s (through
 * llama.cpp's mtmd, as llama-mtmd-cli and llama-server run them). See {@link Weights} for the models Kiln knows by name.
 * Weights stay loaded until {@link #close()}. Not thread-safe: use one instance per thread.
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
    private record Sink(LlamaModel model, Consumer<String> consumer) {}

    private static final ThreadLocal<Sink> SINK = new ThreadLocal<>();

    private static int onToken(MemorySegment piece, MemorySegment user) {
        Sink sink = SINK.get();
        if (sink == null) return 0;
        if (sink.consumer() != null) sink.consumer().accept(NativeLibrary.string(piece));
        return sink.model().cancelled ? 1 : 0;
    }

    private volatile boolean cancelled;

    /**
     * Stops the generation in progress (from any thread) after the current token; the call returns what was
     * generated so far. Takes effect for calls that stream (a non-null onPiece/onContent callback).
     */
    public void cancel() { cancelled = true; }

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
     * {@link #downloadCommand()}, which {@link #model()} also prints when they're missing.
     */
    public enum Weights {
        /** Qwen3 4B Instruct 2507, text only. Apache-2.0. */
        QWEN3_4B("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", null),
        /** Qwen3-VL 4B Instruct: image questions, documents, charts and grounding (boxes on a 0 to 1000 grid). Apache-2.0. */
        QWEN3_VL_4B("Qwen/Qwen3-VL-4B-Instruct-GGUF", "Qwen3VL-4B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf"),
        /**
         * PaddleOCR-VL 1.6 (0.9B): document parsing. It takes a task prompt rather than a question, such as "OCR:",
         * "Table Recognition:", "Formula Recognition:" or "Chart Recognition:". Apache-2.0.
         */
        PADDLEOCR_VL("PaddlePaddle/PaddleOCR-VL-1.6-GGUF", "PaddleOCR-VL-1.6-GGUF.gguf", "PaddleOCR-VL-1.6-GGUF-mmproj.gguf"),
        /**
         * Google MedGemma 1.5 4B (instruction tuned): medical images and text. For research, not diagnosis.
         * Health AI Developer Foundations terms.
         */
        MEDGEMMA_4B("unsloth/medgemma-1.5-4b-it-GGUF", "medgemma-1.5-4b-it-Q4_K_M.gguf", "mmproj-F16.gguf");

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

        /** By name, ignoring case and '-' vs '_': "qwen3-vl-4b", "paddleocr-vl", "medgemma-4b", "qwen3-4b". */
        public static Weights named(String name) {
            return valueOf(name.toUpperCase().replace('-', '_'));
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

    private final MemorySegment handle;
    private final boolean vision;

    /** @param gpuLayers -1 offloads every layer. @param contextSize 0 uses the model's trained context. */
    public LlamaModel(String ggufPath, int gpuLayers, int contextSize) { this(ggufPath, null, gpuLayers, contextSize); }

    public LlamaModel(String ggufPath) { this(ggufPath, -1, 8192); }

    /**
     * A vision language model: the language model's GGUF plus its mmproj GGUF (the vision encoder and projector).
     * mmprojPath null loads the language model alone. The vision encoder runs on the GPU unless gpuLayers is 0.
     * @param gpuLayers -1 offloads every layer. @param contextSize 0 uses the model's trained context.
     */
    public LlamaModel(String ggufPath, String mmprojPath, int gpuLayers, int contextSize) {
        try (Arena a = Arena.ofConfined()) {
            handle = mmprojPath == null
                    ? (MemorySegment) CREATE.invokeExact(a.allocateFrom(ggufPath), gpuLayers, contextSize)
                    : (MemorySegment) CREATE_MM.invokeExact(a.allocateFrom(ggufPath), a.allocateFrom(mmprojPath), gpuLayers, contextSize);
            if (handle.equals(MemorySegment.NULL))
                throw new IllegalStateException((mmprojPath == null ? "kiln_llama_create: " : "kiln_llama_create_mm: ") + error());
            vision = mmprojPath != null && (int) SUPPORTS_VISION.invokeExact(handle) != 0;
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    /** A vision language model and its mmproj, with 8192 tokens of context (images take their share of it). */
    public LlamaModel(String ggufPath, String mmprojPath) { this(ggufPath, mmprojPath, -1, 8192); }

    /** Weights Kiln knows by name, with their mmproj if they have one, and 8192 tokens of context. */
    public LlamaModel(Weights w) { this(w, -1, 8192); }

    public LlamaModel(Weights w, int gpuLayers, int contextSize) { this(w.model(), w.mmproj(), gpuLayers, contextSize); }

    /** Whether this model takes images: it was loaded with an mmproj that has a vision encoder. */
    public boolean supportsVision() { return vision; }

    /** The marker {@link #complete(String, List, Options, Consumer)} replaces with an image (llama.cpp's default). */
    public static String mediaMarker() {
        try { return NativeLibrary.string((MemorySegment) MEDIA_MARKER.invokeExact()); }
        catch (Throwable t) { throw new RuntimeException(t); }
    }

    public Result chat(List<Message> messages, Options o) { return chat(messages, o, null); }

    /**
     * Formats with the model's chat template and generates. onPiece (optional) streams text as it is produced.
     * Messages with images need a model loaded with its mmproj; such a chat is formatted with the model's own
     * (Jinja) chat template, as {@link #chatJson} does, and its prompt token count includes the image tokens.
     */
    public Result chat(List<Message> messages, Options o, Consumer<String> onPiece) {
        if (messages.stream().anyMatch(m -> !m.images().isEmpty())) return chatImages(messages, o, onPiece);
        try (Arena a = Arena.ofConfined()) {
            int n = messages.size();
            MemorySegment roles = a.allocate(ADDRESS, n), contents = a.allocate(ADDRESS, n);
            for (int i = 0; i < n; i++) {
                roles.setAtIndex(ADDRESS, i, a.allocateFrom(messages.get(i).role()));
                contents.setAtIndex(ADDRESS, i, a.allocateFrom(messages.get(i).content()));
            }
            Out out = new Out(a);
            MemorySegment cb = callback(a, onPiece);
            int rc = (int) CHAT.invokeExact(handle, roles, contents, n, o.maxTokens(), o.temperature(), o.topP(),
                    o.minP(), o.seed(), cb, MemorySegment.NULL, out.text, out.nPrompt, out.nGen);
            return out.result(rc);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
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
        try (Arena a = Arena.ofConfined()) {
            MemorySegment out = a.allocate(ADDRESS);
            MemorySegment cb = callback(a, onContent);
            int rc = (int) CHAT_JSON.invokeExact(handle, a.allocateFrom(requestJson), cb, MemorySegment.NULL, out);
            MemorySegment r = out.get(ADDRESS, 0);
            String json = NativeLibrary.string(r);
            if (!r.equals(MemorySegment.NULL)) FREE.invokeExact(r);
            if (rc != 0) throw new IllegalStateException("llama chat failed (rc=" + rc + "): " + error());
            return json;
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { SINK.remove(); }
    }

    private Result chatImages(List<Message> messages, Options o, Consumer<String> onPiece) {
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
            MemorySegment cb = callback(a, onPiece);
            int rc = (int) CHAT_IMAGES.invokeExact(handle, roles, contents, counts, n, img.rgb, img.widths, img.heights, img.n,
                    o.maxTokens(), o.temperature(), o.topP(), o.minP(), o.seed(), cb, MemorySegment.NULL,
                    out.text, out.nPrompt, out.nGen);
            return out.result(rc);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
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
        if (images.isEmpty()) return chatJson(requestJson, onContent);
        requireVision();
        try (Arena a = Arena.ofConfined()) {
            NativeImages img = new NativeImages(a, images);
            MemorySegment out = a.allocate(ADDRESS);
            MemorySegment cb = callback(a, onContent);
            int rc = (int) CHAT_JSON_IMAGES.invokeExact(handle, a.allocateFrom(requestJson), img.rgb, img.widths, img.heights,
                    img.n, cb, MemorySegment.NULL, out);
            MemorySegment r = out.get(ADDRESS, 0);
            String json = NativeLibrary.string(r);
            if (!r.equals(MemorySegment.NULL)) FREE.invokeExact(r);
            if (rc != 0) throw new IllegalStateException("llama chat failed (rc=" + rc + "): " + error());
            return json;
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { SINK.remove(); }
    }

    /**
     * Structured extraction: the reply to messages (with or without images), constrained to a JSON schema, as JSON
     * text; for example an invoice's vendor, date, total and line items from a picture of it. Reasoning, for a model
     * that thinks, is left out. Parse the result with any JSON library, or {@link Json#parse}.
     * @param jsonSchema a JSON Schema document, such as {@code {"type": "object", "properties": {...}, "required": [...]}}
     * @throws IllegalStateException if the reply ran out of tokens before the JSON was complete
     */
    public String extract(List<Message> messages, String jsonSchema, Options o) {
        List<Image> images = new ArrayList<>();
        String request = ChatRequests.extract(messages, jsonSchema, o, images);
        Object response = Json.parse(chatJson(request, images, null));
        Object finish = Json.get(response, "finish_reason");
        if (!"stop".equals(finish)) throw new IllegalStateException("the reply was cut off (" + finish + "); raise maxTokens");
        return Json.get(response, "message", "content") instanceof String s ? s : "";
    }

    /**
     * Raw completion of an already-formatted prompt with images: the prompt holds {@link #mediaMarker()} once for each
     * image, in order, where the model's own image tokens go. Needs a model loaded with its mmproj.
     */
    public Result complete(String prompt, List<Image> images, Options o, Consumer<String> onPiece) {
        if (images.isEmpty()) return complete(prompt, o, onPiece);
        requireVision();
        try (Arena a = Arena.ofConfined()) {
            NativeImages img = new NativeImages(a, images);
            Out out = new Out(a);
            MemorySegment cb = callback(a, onPiece);
            int rc = (int) COMPLETE_IMAGES.invokeExact(handle, a.allocateFrom(prompt), img.rgb, img.widths, img.heights, img.n,
                    o.maxTokens(), o.temperature(), o.topP(), o.minP(), o.seed(), cb, MemorySegment.NULL,
                    out.text, out.nPrompt, out.nGen);
            return out.result(rc);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
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
        try (Arena a = Arena.ofConfined()) {
            Out out = new Out(a);
            MemorySegment cb = callback(a, onPiece);
            int rc = (int) COMPLETE.invokeExact(handle, a.allocateFrom(prompt), o.maxTokens(), o.temperature(),
                    o.topP(), o.minP(), o.seed(), cb, MemorySegment.NULL, out.text, out.nPrompt, out.nGen);
            return out.result(rc);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
        finally { SINK.remove(); }
    }

    private MemorySegment callback(Arena a, Consumer<String> onPiece) {
        cancelled = false;
        if (onPiece == null) return MemorySegment.NULL;
        SINK.set(new Sink(this, onPiece));
        return NativeLibrary.linker().upcallStub(ON_TOKEN, TOKEN_CB, a);
    }

    private static final class Out {
        final MemorySegment text, nPrompt, nGen;
        Out(Arena a) { text = a.allocate(ADDRESS); nPrompt = a.allocate(JAVA_INT); nGen = a.allocate(JAVA_INT); }

        Result result(int rc) throws Throwable {
            MemorySegment t = text.get(ADDRESS, 0);
            String s = NativeLibrary.string(t);
            if (!t.equals(MemorySegment.NULL)) FREE.invokeExact(t);
            if (rc != 0) throw new IllegalStateException("llama generation failed (rc=" + rc + "): " + error());
            return new Result(s, nPrompt.get(JAVA_INT, 0), nGen.get(JAVA_INT, 0));
        }
    }

    private static String error() {
        try { return NativeLibrary.string((MemorySegment) LAST_ERROR.invokeExact()); }
        catch (Throwable t) { return t.toString(); }
    }

    @Override public void close() {
        try { DESTROY.invokeExact(handle); } catch (Throwable t) { throw new RuntimeException(t); }
    }
}
