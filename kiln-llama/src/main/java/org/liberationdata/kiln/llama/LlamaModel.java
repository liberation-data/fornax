package org.liberationdata.kiln.llama;

import org.liberationdata.kiln.NativeLibrary;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.util.List;
import java.util.function.Consumer;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * A GGUF language model running in-process through llama.cpp (via libkiln_llama).
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
    private static final ThreadLocal<Consumer<String>> SINK = new ThreadLocal<>();

    private static int onToken(MemorySegment piece, MemorySegment user) {
        Consumer<String> sink = SINK.get();
        if (sink != null) sink.accept(NativeLibrary.string(piece));
        return 0;
    }

    public record Message(String role, String content) {
        public static Message system(String c) { return new Message("system", c); }
        public static Message user(String c) { return new Message("user", c); }
        public static Message assistant(String c) { return new Message("assistant", c); }
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

    /** @param gpuLayers -1 offloads every layer. @param contextSize 0 uses the model's trained context. */
    public LlamaModel(String ggufPath, int gpuLayers, int contextSize) {
        try (Arena a = Arena.ofConfined()) {
            handle = (MemorySegment) CREATE.invokeExact(a.allocateFrom(ggufPath), gpuLayers, contextSize);
        } catch (Throwable t) { throw new RuntimeException(t); }
        if (handle.equals(MemorySegment.NULL)) throw new IllegalStateException("kiln_llama_create: " + error());
    }

    public LlamaModel(String ggufPath) { this(ggufPath, -1, 8192); }

    public Result chat(List<Message> messages, Options o) { return chat(messages, o, null); }

    /** Formats with the model's chat template and generates. onPiece (optional) streams text as it is produced. */
    public Result chat(List<Message> messages, Options o, Consumer<String> onPiece) {
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

    private static MemorySegment callback(Arena a, Consumer<String> onPiece) {
        if (onPiece == null) return MemorySegment.NULL;
        SINK.set(onPiece);
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
