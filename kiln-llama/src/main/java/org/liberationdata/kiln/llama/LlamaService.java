package org.liberationdata.kiln.llama;

import org.liberationdata.kiln.Call;
import org.liberationdata.kiln.Wire;
import org.liberationdata.kiln.WorkerService;
import org.liberationdata.kiln.llama.LlamaModel.Image;
import org.liberationdata.kiln.llama.LlamaModel.Message;
import org.liberationdata.kiln.llama.LlamaModel.Options;
import org.liberationdata.kiln.llama.LlamaModel.Result;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * The worker side of an isolated {@link LlamaModel} ({@link LlamaModel#isolated}): an in-process LlamaModel in the
 * worker process, and the encoding of its calls. Not for direct use; the worker creates it by name.
 */
public final class LlamaService implements WorkerService {
    private final LlamaModel model;

    /** config: {@link #config}. */
    public LlamaService(byte[] config) {
        Wire.In in = new Wire.In(config);
        model = LlamaModel.inProcess(in.string(), in.string(), in.integer(), in.integer());
    }

    @Override public byte[] call(String method, byte[] args, Call call, Consumer<byte[]> events) {
        Wire.In in = new Wire.In(args == null ? new byte[0] : args);
        return switch (method) {
            case "vision" -> new Wire.Out().bool(model.supportsVision()).toBytes();
            case "chat" -> {
                List<Message> messages = messages(in);
                Options o = options(in);
                yield result(model.chat(messages, o, in.bool() ? pieces(events) : null));
            }
            case "chatJson" -> {
                String request = in.string();
                List<Image> images = images(in);
                yield utf8(model.chatJson(request, images, in.bool() ? pieces(events) : null));
            }
            case "complete" -> {
                String prompt = in.string();
                List<Image> images = images(in);
                Options o = options(in);
                yield result(model.complete(prompt, images, o, in.bool() ? pieces(events) : null));
            }
            default -> throw new UnsupportedOperationException("LlamaService has no method " + method);
        };
    }

    @Override public void cancel() { model.cancel(); }

    @Override public void close() { model.close(); }

    private static Consumer<String> pieces(Consumer<byte[]> events) { return s -> events.accept(utf8(s)); }

    // ---- the encoding, shared with LlamaModel's isolated side ----

    static byte[] config(String gguf, String mmproj, int gpuLayers, int contextSize) {
        return new Wire.Out().string(gguf).string(mmproj).integer(gpuLayers).integer(contextSize).toBytes();
    }

    static byte[] chatArgs(List<Message> messages, Options o, boolean stream) {
        Wire.Out out = new Wire.Out().integer(messages.size());
        for (Message m : messages) {
            out.string(m.role()).string(m.content());
            images(out, m.images());
        }
        return options(out, o).bool(stream).toBytes();
    }

    static byte[] chatJsonArgs(String request, List<Image> images, boolean stream) {
        Wire.Out out = new Wire.Out().string(request);
        return images(out, images).bool(stream).toBytes();
    }

    static byte[] completeArgs(String prompt, List<Image> images, Options o, boolean stream) {
        Wire.Out out = new Wire.Out().string(prompt);
        return options(images(out, images), o).bool(stream).toBytes();
    }

    static Result decodeResult(byte[] b) {
        Wire.In in = new Wire.In(b);
        return new Result(in.string(), in.integer(), in.integer());
    }

    static String decodeString(byte[] b) { return new String(b, StandardCharsets.UTF_8); }

    private static byte[] result(Result r) {
        return new Wire.Out().string(r.text()).integer(r.promptTokens()).integer(r.generatedTokens()).toBytes();
    }

    private static byte[] utf8(String s) { return s == null ? new byte[0] : s.getBytes(StandardCharsets.UTF_8); }

    private static Wire.Out images(Wire.Out out, List<Image> images) {
        out.integer(images.size());
        for (Image img : images) out.integer(img.width()).integer(img.height()).bytes(img.rgb());
        return out;
    }

    private static List<Image> images(Wire.In in) {
        int n = in.integer();
        List<Image> images = new ArrayList<>(n);
        for (int i = 0; i < n; i++) images.add(new Image(in.integer(), in.integer(), in.bytes()));
        return images;
    }

    private static List<Message> messages(Wire.In in) {
        int n = in.integer();
        List<Message> messages = new ArrayList<>(n);
        for (int i = 0; i < n; i++) messages.add(new Message(in.string(), in.string(), images(in)));
        return messages;
    }

    private static Wire.Out options(Wire.Out out, Options o) {
        return out.integer(o.maxTokens()).floatValue(o.temperature()).floatValue(o.topP()).floatValue(o.minP()).integer(o.seed());
    }

    private static Options options(Wire.In in) {
        return new Options(in.integer(), in.floatValue(), in.floatValue(), in.floatValue(), in.integer());
    }
}
