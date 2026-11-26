package org.liberationdata.fornax.snippets;

// --8<-- [start:hfcache]
import org.liberationdata.fornax.HfCache;

import java.nio.file.Path;
import java.util.Optional;
// --8<-- [end:hfcache]

/** The snippets of website/src/content/docs/docs/getting-started.mdx: compiled, never run. */
class GettingStarted {
    void hfCache() {
        // --8<-- [start:hfcache]
        // The file inside any downloaded snapshot of the repository, if there is one.
        Optional<Path> maybe = HfCache.find("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf");

        // The same, or an IllegalStateException that names the hf download command to run.
        Path gguf = HfCache.require("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf");

        // The first of several (repository, path) pairs that is present.
        Path encoder = HfCache.requireAny(
                "Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
                "Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors");
        // --8<-- [end:hfcache]
    }
}
