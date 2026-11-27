package org.liberationdata.fornax.snippets;

// --8<-- [start:pool]
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.ModelPool;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
// --8<-- [end:pool]
// --8<-- [start:mixed]
import org.liberationdata.fornax.sd.DiffusionModel;
// --8<-- [end:mixed]

/**
 * The snippets of website/src/content/docs/docs/guides/model-pool.md: compiled, never run. Named ModelPoolGuide, not
 * ModelPool, so it doesn't clash with the imported {@code org.liberationdata.fornax.ModelPool}.
 */
class ModelPoolGuide {
    void pool() {
        // --8<-- [start:pool]
        Path gguf = HfCache.require("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf");

        try (var pool = new ModelPool(96L << 30, Duration.ofMinutes(10))) {   // a 96 GB budget; unload after 10 idle minutes
            try (var lease = pool.acquire("qwen3-4b", ModelPool.sizeOf(gguf), () -> new LlamaModel(gguf.toString()))) {
                lease.model().chat(List.of(Message.user("Hello!")), Options.DEFAULT, System.out::print);
            }
            // The model is still loaded: the next acquire("qwen3-4b", ...) gets it at once.
        }
        // --8<-- [end:pool]
    }

    void mixed(ModelPool pool) {
        // --8<-- [start:mixed]
        Path flux = HfCache.require("black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors");
        Path vae = HfCache.require("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors");
        Path qwen3 = HfCache.require("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors");

        try (var lease = pool.acquire("flux2-klein", ModelPool.sizeOf(flux, vae, qwen3), () -> new DiffusionModel(
                DiffusionModel.Config.builder().diffusionModel(flux.toString()).vae(vae.toString()).llm(qwen3.toString()).build()))) {
            var img = lease.model().generate(new DiffusionModel.Request("a fornax at dusk", 768, 768, 12, 4.0, 7)).get(0);
        }
        // --8<-- [end:mixed]
    }

    void entries(ModelPool pool) {
        // --8<-- [start:entries]
        for (ModelPool.Entry e : pool.entries()) {
            System.out.printf("%s: %d MB, loaded=%s, inUse=%s, last used %s%n",
                    e.key(), e.sizeBytes() >> 20, e.loaded(), e.inUse(), e.lastUsed());
        }
        System.out.println((pool.loadedBytes() >> 20) + " MB loaded");

        pool.unload("flux2-klein");   // waits until it is free, then unloads it; false if it wasn't loaded
        pool.unloadAll();             // unloads every model that isn't in use
        // --8<-- [end:entries]
    }
}
