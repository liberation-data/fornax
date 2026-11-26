package org.liberationdata.kiln.sd;

import org.liberationdata.kiln.Call;
import org.liberationdata.kiln.NativeLibrary;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;

import static java.lang.foreign.ValueLayout.*;

/**
 * Hand-written Java FFM bindings for stable-diffusion.cpp's C API (include/stable-diffusion.h).
 * The API is option-key based (sd_opt_set_*), so no struct layouts are needed for basic use.
 */
public final class Sd {
    private static final NativeLibrary LIB = NativeLibrary.load("stable-diffusion");

    private static MethodHandle fn(String name, FunctionDescriptor fd) { return LIB.function(name, fd); }

    // status/diagnostics
    static final MethodHandle get_last_error = fn("sd_get_last_error", FunctionDescriptor.of(ADDRESS));
    // options
    static final MethodHandle ctx_opt_create = fn("sd_ctx_opt_create", FunctionDescriptor.of(ADDRESS));
    static final MethodHandle img_gen_opt_create = fn("sd_img_gen_opt_create", FunctionDescriptor.of(ADDRESS));
    static final MethodHandle vid_gen_opt_create = fn("sd_vid_gen_opt_create", FunctionDescriptor.of(ADDRESS));
    static final MethodHandle opt_free = fn("sd_opt_free", FunctionDescriptor.ofVoid(ADDRESS));
    static final MethodHandle opt_set_bool = fn("sd_opt_set_bool", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_BOOLEAN));
    static final MethodHandle opt_set_int = fn("sd_opt_set_int", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG));
    static final MethodHandle opt_set_float = fn("sd_opt_set_float", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_DOUBLE));
    static final MethodHandle opt_set_string = fn("sd_opt_set_string", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle opt_set_floats = fn("sd_opt_set_floats", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG));
    static final MethodHandle opt_set_tensor = fn("sd_opt_set_tensor", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle opt_set_audios = fn("sd_opt_set_audios", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG));
    static final MethodHandle opt_set_tensors = fn("sd_opt_set_tensors", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_LONG));
    // context + generation
    static final MethodHandle ctx_create = fn("sd_ctx_create", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
    static final MethodHandle ctx_free = fn("sd_ctx_free", FunctionDescriptor.ofVoid(ADDRESS));
    static final MethodHandle generate_image = fn("sd_generate_image", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle generate_video = fn("sd_generate_video", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
    static final MethodHandle get_model_version_name = fn("sd_get_model_version_name", FunctionDescriptor.of(ADDRESS, ADDRESS));
    /** Thread-safe; only reaches a run that has started executing (an idle context ignores it). */
    static final MethodHandle cancel_generation = fn("sd_cancel_generation", FunctionDescriptor.ofVoid(ADDRESS, JAVA_INT));
    // results
    static final MethodHandle result_get_image_count = fn("sd_result_get_image_count", FunctionDescriptor.of(JAVA_LONG, ADDRESS));
    static final MethodHandle result_get_image = fn("sd_result_get_image", FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG));
    static final MethodHandle result_get_audio = fn("sd_result_get_audio", FunctionDescriptor.of(ADDRESS, ADDRESS));
    static final MethodHandle result_get_fps = fn("sd_result_get_fps", FunctionDescriptor.of(JAVA_INT, ADDRESS));
    static final MethodHandle result_free = fn("sd_result_free", FunctionDescriptor.ofVoid(ADDRESS));
    // tensors
    static final MethodHandle tensor_create = fn("sd_tensor_create", FunctionDescriptor.of(ADDRESS, JAVA_INT, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG));
    static final MethodHandle tensor_free = fn("sd_tensor_free", FunctionDescriptor.ofVoid(ADDRESS));
    static final MethodHandle tensor_rank = fn("sd_tensor_rank", FunctionDescriptor.of(JAVA_LONG, ADDRESS));
    static final MethodHandle tensor_dim = fn("sd_tensor_dim", FunctionDescriptor.of(JAVA_LONG, ADDRESS, JAVA_LONG));
    static final MethodHandle tensor_byte_size = fn("sd_tensor_byte_size", FunctionDescriptor.of(JAVA_LONG, ADDRESS));
    static final MethodHandle tensor_data = fn("sd_tensor_data", FunctionDescriptor.of(ADDRESS, ADDRESS));
    // callbacks
    static final MethodHandle set_log_callback = fn("sd_set_log_callback", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
    static final MethodHandle set_progress_callback = fn("sd_set_progress_callback", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));

    /** sd_audio_t: { uint32_t sample_rate; const sd_tensor_t* data; } */
    static final StructLayout AUDIO = MemoryLayout.structLayout(JAVA_INT.withName("sample_rate"), MemoryLayout.paddingLayout(4), ADDRESS.withName("data"));

    public static final int SD_TENSOR_U8 = 1;
    public static final int SD_TENSOR_F32 = 4;
    /** sd_status_t of a run stopped by sd_cancel_generation. */
    static final int SD_STATUS_CANCELLED = 10;
    /** sd_cancel_mode_t: stop the current generation as soon as possible. */
    static final int SD_CANCEL_ALL = 0;

    private Sd() {}

    static String lastError() {
        try {
            MemorySegment s = (MemorySegment) get_last_error.invokeExact();
            return s.equals(MemorySegment.NULL) ? "(no error text)" : s.reinterpret(Long.MAX_VALUE).getString(0);
        } catch (Throwable t) { throw new RuntimeException(t); }
    }

    static void check(int status, String what) {
        if (status != 0) throw new SdException(what + " failed: status " + status + ": " + lastError());
    }

    public static final class SdException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        SdException(String m) { super(m); }
    }

    // ---- callbacks (upcalls) ----
    public interface LogSink { void log(int level, String text); }
    public interface ProgressSink { void progress(int step, int steps, float seconds); }

    // sd log levels: 0 debug, 1 info, 2 warn, 3 error. Quiet unless KILN_VERBOSE is set.
    private static LogSink logSink = (l, t) -> { if (l >= logThreshold()) System.err.print(t); };

    /** KILN_VERBOSE unset: errors only; set: info and up; "debug": everything, including the compiled DGL program. */
    private static int logThreshold() {
        String v = System.getenv("KILN_VERBOSE");
        return v == null ? 3 : "debug".equalsIgnoreCase(v) ? 0 : 1;
    }
    private static ProgressSink progressSink = (s, n, t) -> {};

    /** The DiffusionModel call running on this thread, if any: where a sink's exception goes. */
    static final ThreadLocal<Call> CALL = new ThreadLocal<>();

    /*
     * The log and progress upcalls. Nothing may escape them: an exception thrown into stable-diffusion.cpp's frames
     * kills the JVM. During a generate call a sink's exception is recorded in the call and the run is cancelled;
     * generate rethrows it on return. Outside one (a log line while loading, say) there is no caller to give it to,
     * so it is printed.
     */

    static void onLog(int level, MemorySegment text, MemorySegment data) {
        try { logSink.log(level, text.reinterpret(Long.MAX_VALUE).getString(0)); } catch (Throwable t) { failed(t); }
    }

    static void onProgress(int step, int steps, float time, MemorySegment data) {
        try {
            progressSink.progress(step, steps, time);
            // sd.cpp ignores a cancel that comes before its run begins executing, so a stopped call asks again here.
            Call call = CALL.get();
            if (call != null && call.stopRequested()) call.cancel();
        } catch (Throwable t) { failed(t); }
    }

    private static void failed(Throwable t) {
        try {
            Call call = CALL.get();
            if (call != null) call.fail(t);
            else { System.err.println("Kiln: a stable-diffusion.cpp log or progress sink threw outside a generation:"); t.printStackTrace(); }
        } catch (Throwable ignored) {}  // last resort: never unwind into native code
    }

    private static final MemorySegment LOG_STUB, PROGRESS_STUB;
    static {
        try {
            var lookup = MethodHandles.lookup();
            LOG_STUB = NativeLibrary.linker().upcallStub(
                    lookup.findStatic(Sd.class, "onLog", MethodType.methodType(void.class, int.class, MemorySegment.class, MemorySegment.class)),
                    FunctionDescriptor.ofVoid(JAVA_INT, ADDRESS, ADDRESS), Arena.global());
            PROGRESS_STUB = NativeLibrary.linker().upcallStub(
                    lookup.findStatic(Sd.class, "onProgress", MethodType.methodType(void.class, int.class, int.class, float.class, MemorySegment.class)),
                    FunctionDescriptor.ofVoid(JAVA_INT, JAVA_INT, JAVA_FLOAT, ADDRESS), Arena.global());
            set_log_callback.invokeExact(LOG_STUB, MemorySegment.NULL);
            set_progress_callback.invokeExact(PROGRESS_STUB, MemorySegment.NULL);
        } catch (Throwable t) { throw new ExceptionInInitializerError(t); }
    }

    public static void setLogSink(LogSink s) { logSink = s; }
    public static void setProgressSink(ProgressSink s) { progressSink = s; }
}
