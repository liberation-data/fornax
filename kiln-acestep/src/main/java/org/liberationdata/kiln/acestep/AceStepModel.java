package org.liberationdata.kiln.acestep;

import org.liberationdata.kiln.Audio;
import org.liberationdata.kiln.NativeLibrary;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.*;

/**
 * ACE-Step 1.5 song generation in-process via acestep.cpp (through libkiln_acestep).
 * Request is acestep.cpp's JSON (caption, lyrics, bpm, duration, keyscale, seed, ...); output is 48 kHz stereo.
 * Not thread-safe: use one instance per thread.
 */
public final class AceStepModel implements AutoCloseable {
    private static final NativeLibrary LIB = NativeLibrary.load("kiln_acestep");

    private static MethodHandle fn(String n, FunctionDescriptor fd) { return LIB.function(n, fd); }

    private static final MethodHandle LAST_ERROR = fn("kiln_acestep_last_error", FunctionDescriptor.of(ADDRESS));
    private static final MethodHandle CREATE = fn("kiln_acestep_create", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, JAVA_INT));
    private static final MethodHandle GENERATE = fn("kiln_acestep_generate", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
    private static final MethodHandle FREE = fn("kiln_acestep_free", FunctionDescriptor.ofVoid(ADDRESS));
    private static final MethodHandle DESTROY = fn("kiln_acestep_destroy", FunctionDescriptor.ofVoid(ADDRESS));

    /** resolvedRequestJson is the request after the LM filled in metadata, lyrics and seeds. */
    public record Result(Audio audio, String resolvedRequestJson) {}

    private static String err() {
        try { return ((MemorySegment) LAST_ERROR.invokeExact()).reinterpret(Long.MAX_VALUE).getString(0); }
        catch (Throwable t) { return t.toString(); }
    }

    private final MemorySegment handle;

    /** lmPath may be null to skip the 5 Hz LM (DiT-only: caption + lyrics drive generation directly). */
    public AceStepModel(String lmPath, String textEncPath, String ditPath, String vaePath, boolean keepLoaded) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment lm = lmPath == null ? MemorySegment.NULL : a.allocateFrom(lmPath);
            handle = (MemorySegment) CREATE.invokeExact(lm,
                    a.allocateFrom(textEncPath), a.allocateFrom(ditPath), a.allocateFrom(vaePath), keepLoaded ? 1 : 0);
        } catch (Throwable t) { throw new RuntimeException(t); }
        if (handle.equals(MemorySegment.NULL)) throw new IllegalStateException("kiln_acestep_create: " + err());
    }

    public Result generate(String requestJson) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment pcmOut = a.allocate(ADDRESS), nOut = a.allocate(JAVA_INT), jsonOut = a.allocate(ADDRESS);
            int rc = (int) GENERATE.invokeExact(handle, a.allocateFrom(requestJson), pcmOut, nOut, jsonOut);
            if (rc != 0) throw new IllegalStateException("kiln_acestep_generate rc=" + rc + ": " + err());
            int n = nOut.get(JAVA_INT, 0);
            MemorySegment pcm = pcmOut.get(ADDRESS, 0).reinterpret((long) n * 2 * 4);
            MemorySegment js = jsonOut.get(ADDRESS, 0);
            float[] l = pcm.asSlice(0, (long) n * 4).toArray(JAVA_FLOAT);
            float[] r = pcm.asSlice((long) n * 4, (long) n * 4).toArray(JAVA_FLOAT);
            String resolved = js.equals(MemorySegment.NULL) ? null : js.reinterpret(Long.MAX_VALUE).getString(0);
            FREE.invokeExact(pcm);
            if (!js.equals(MemorySegment.NULL)) FREE.invokeExact(js);
            return new Result(new Audio(l, r, 48000), resolved);
        } catch (RuntimeException e) { throw e; } catch (Throwable t) { throw new RuntimeException(t); }
    }

    @Override public void close() {
        try { DESTROY.invokeExact(handle); } catch (Throwable t) { throw new RuntimeException(t); }
    }
}
