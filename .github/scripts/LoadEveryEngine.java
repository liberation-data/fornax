// Loads each engine's native library through Fornax's NativeLibrary, as an app would, and finds one C function in it:
// a check that a platform's natives load and export Fornax's API (run by windows.yml, with -Dfornax.library.path).
import java.lang.foreign.FunctionDescriptor;
import java.util.Map;
import org.liberationdata.fornax.NativeLibrary;

public class LoadEveryEngine {
    public static void main(String[] args) {
        Map<String, String> oneFunction = Map.of(
                "fornax_llama", "fornax_llama_last_error",
                "fornax_tts", "fornax_tts_last_error",
                "stable-diffusion", "sd_ctx_free",
                "fornax_acestep", "fornax_acestep_last_error",
                "fornax_yue2", "fornax_yue2_last_error",
                "fornax_whisper", "fornax_whisper_last_error",
                "fornax_demucs", "fornax_demucs_last_error");
        oneFunction.forEach((lib, fn) -> {
            NativeLibrary.load(lib).function(fn, FunctionDescriptor.ofVoid());
            System.out.println("loaded " + lib + ", found " + fn);
        });
    }
}
