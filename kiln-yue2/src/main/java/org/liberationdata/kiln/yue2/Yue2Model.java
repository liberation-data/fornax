package org.liberationdata.kiln.yue2;

import org.liberationdata.kiln.Audio;
import org.liberationdata.kiln.NativeLibrary;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

/**
 * YuE2 song generation in-process via yue2.cpp (through libkiln_yue2).
 * Request is yue2.cpp's JSON (style, lyrics, duration, cot, seed, lm_seed, steps, ...); output is 48 kHz stereo.
 * YuE2 weights are CC BY-NC 4.0 (no commercial use). Not thread-safe: use one instance per thread.
 */
public final class Yue2Model implements AutoCloseable {
    private static final NativeLibrary LIB = NativeLibrary.load("kiln_yue2");

    private static MethodHandle fn(String n, FunctionDescriptor fd) { return LIB.function(n, fd); }

    private static final MethodHandle LAST_ERROR = fn("kiln_yue2_last_error", FunctionDescriptor.of(ADDRESS));
    private static final MethodHandle CREATE = fn("kiln_yue2_create", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT));
    private static final MethodHandle GENERATE = fn("kiln_yue2_generate", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle FREE = fn("kiln_yue2_free", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle DESTROY = fn("kiln_yue2_destroy", FunctionDescriptor.ofVoid(ADDRESS));

    /** abcScore is the ABC-notation score YuE2 writes before rendering audio, when present. */
    public record Result(Audio audio, String resolvedRequestJson, String abcScore) {}

    private static String str(MemorySegment s) { return s.equals(MemorySegment.NULL) ? null : s.reinterpret(Long.MAX_VALUE).getString(0); }

    private static String err() {
        try { return str((MemorySegment) LAST_ERROR.invokeExact()); } catch (Throwable t) { return t.toString(); }
    }

    private final MemorySegment handle;

    public Yue2Model(String modelPath, String vaePath, boolean keepLoaded, int maxSeq) {
        try (Arena a = Arena.ofConfined()) {
            handle = (MemorySegment) CREATE.invokeExact(a.allocateFrom(modelPath), a.allocateFrom(vaePath), keepLoaded ? 1 : 0, maxSeq);
        } catch (Throwable t) { throw new RuntimeException(t); }
        if (handle.equals(MemorySegment.NULL)) throw new IllegalStateException("kiln_yue2_create: " + err());
    }

    public Result generate(String requestJson) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment pcmOut = a.allocate(ADDRESS), nOut = a.allocate(JAVA_INT), jsonOut = a.allocate(ADDRESS), scoreOut = a.allocate(ADDRESS);
            int rc = (int) GENERATE.invokeExact(handle, a.allocateFrom(requestJson), pcmOut, nOut, jsonOut, scoreOut);
            if (rc != 0) throw new IllegalStateException("kiln_yue2_generate rc=" + rc + ": " + err());
            int n = nOut.get(JAVA_INT, 0);
            MemorySegment pcm = pcmOut.get(ADDRESS, 0).reinterpret((long) n * 8);
            float[] l = pcm.asSlice(0, (long) n * 4).toArray(JAVA_FLOAT);
            float[] r = pcm.asSlice((long) n * 4, (long) n * 4).toArray(JAVA_FLOAT);
            MemorySegment js = jsonOut.get(ADDRESS, 0), sc = scoreOut.get(ADDRESS, 0);
            String json = str(js), score = str(sc);
            FREE.invokeExact(pcm);
            if (!js.equals(MemorySegment.NULL)) FREE.invokeExact(js);
            if (!sc.equals(MemorySegment.NULL)) FREE.invokeExact(sc);
            return new Result(new Audio(l, r, 48000), json, score);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    @Override public void close() {
        try { DESTROY.invokeExact(handle); } catch (Throwable t) { throw new RuntimeException(t); }
    }
}
