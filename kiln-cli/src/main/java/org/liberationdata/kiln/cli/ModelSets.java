package org.liberationdata.kiln.cli;

import java.util.List;
import java.util.Optional;

/**
 * Short names for the models Kiln's examples use, and the files each one needs. This is the one place they are
 * defined: {@code kiln pull}, {@code list}, {@code chat} and {@code test} all read it.
 */
final class ModelSets {
    private ModelSets() {}

    enum Kind { LLM, IMAGE, SONG }

    /** A file in a Hugging Face repo. */
    record Source(String repo, String file) {
        String fileName() { return file.substring(file.lastIndexOf('/') + 1); }
    }

    /**
     * One file a model needs, by role. {@code pull} fetches the first source; the others are accepted when
     * they are already in the cache (the same file published in another repo).
     */
    record Component(String role, List<Source> sources) {
        Source primary() { return sources.get(0); }

        Optional<java.nio.file.Path> locate(HubCache cache) {
            for (Source s : sources) {
                var p = cache.find(s.repo(), s.file());
                if (p.isPresent()) return p;
            }
            return Optional.empty();
        }
    }

    record ModelSet(String name, Kind kind, String description, List<Component> components) {
        Component component(String role) {
            return components.stream().filter(c -> c.role().equals(role)).findFirst()
                    .orElseThrow(() -> new IllegalStateException(name + " has no " + role));
        }
    }

    private static final String ACE = "Serveurperso/ACE-Step-1.5-GGUF", YUE = "Serveurperso/YuE2-GGUF";
    private static final Component FLUX2_VAE = c("vae", "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors");
    private static final Component QWEN_IMAGE_VAE = c("vae", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors");
    private static final Component QWEN_VL = new Component("llm", List.of(
            new Source("Comfy-Org/Qwen-Image_ComfyUI", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
            new Source("Comfy-Org/HunyuanVideo_1.5_repackaged", "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors")));

    static final List<ModelSet> ALL = List.of(
            new ModelSet("qwen3-0.6b", Kind.LLM, "Qwen3 0.6B, Q8_0 (tiny; good for smoke tests)",
                    List.of(c("model", "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf"))),
            new ModelSet("qwen3-4b", Kind.LLM, "Qwen3 4B Instruct 2507, Q4_K_M",
                    List.of(c("model", "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"))),
            new ModelSet("ace-step", Kind.SONG, "ACE-Step 1.5 turbo with the 4B lyric LM", List.of(
                    c("lm", ACE, "acestep-5Hz-lm-4B-Q8_0.gguf"),
                    c("text-encoder", ACE, "Qwen3-Embedding-0.6B-Q8_0.gguf"),
                    c("dit", ACE, "acestep-v15-turbo-Q8_0.gguf"),
                    c("vae", ACE, "vae-BF16.gguf"))),
            new ModelSet("ace-step-xl", Kind.SONG, "ACE-Step 1.5 XL turbo with the 4B lyric LM", List.of(
                    c("lm", ACE, "acestep-5Hz-lm-4B-Q8_0.gguf"),
                    c("text-encoder", ACE, "Qwen3-Embedding-0.6B-Q8_0.gguf"),
                    c("dit", ACE, "acestep-v15-xl-turbo-Q8_0.gguf"),
                    c("vae", ACE, "vae-BF16.gguf"))),
            new ModelSet("yue2", Kind.SONG, "YuE2 3B, Q8_0 (weights CC BY-NC 4.0)", List.of(
                    c("model", YUE, "YuE2-3B-Q8_0.gguf"),
                    c("vae", YUE, "YuE2-Vae-F32.gguf"))),
            new ModelSet("flux2-klein", Kind.IMAGE, "FLUX.2 [klein] base 4B, fp8", List.of(
                    c("diffusion", "black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors"),
                    FLUX2_VAE,
                    c("llm", "Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors"))),
            new ModelSet("qwen-image", Kind.IMAGE, "Qwen-Image 2512, fp8", List.of(
                    c("diffusion", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"),
                    QWEN_IMAGE_VAE, QWEN_VL)),
            new ModelSet("qwen-image-edit", Kind.IMAGE, "Qwen-Image-Edit 2511, bf16 (edits a reference image)", List.of(
                    c("diffusion", "Comfy-Org/Qwen-Image-Edit_ComfyUI", "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors"),
                    QWEN_IMAGE_VAE, QWEN_VL)));

    static Optional<ModelSet> byName(String name) {
        return ALL.stream().filter(m -> m.name().equalsIgnoreCase(name)).findFirst();
    }

    /** The LLM alias whose single file this is, if any (so {@code kiln list} can show its short name). */
    static Optional<ModelSet> llmFor(String repo, String file) {
        return ALL.stream().filter(m -> m.kind() == Kind.LLM)
                .filter(m -> m.components().get(0).sources().contains(new Source(repo, file))).findFirst();
    }

    /** Repos whose files belong to image or song engines: their GGUFs aren't chat models. */
    static boolean isEngineRepo(String repo) {
        return ALL.stream().filter(m -> m.kind() != Kind.LLM).flatMap(m -> m.components().stream())
                .flatMap(c -> c.sources().stream()).anyMatch(s -> s.repo().equals(repo));
    }

    private static Component c(String role, String repo, String file) {
        return new Component(role, List.of(new Source(repo, file)));
    }
}
