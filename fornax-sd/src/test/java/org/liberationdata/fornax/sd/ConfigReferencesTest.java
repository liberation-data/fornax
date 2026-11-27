package org.liberationdata.fornax.sd;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelFolders;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelSet.Part;

import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** A Config's catalogue sets and model references, against a cache on disk and an offline hub: no native library needed. */
class ConfigReferencesTest {
    static final String COMMIT = "0123456789abcdef0123456789abcdef01234567";
    @TempDir Path dir;

    /** A file in the Hugging Face cache layout, as a download leaves it. */
    Path cached(String repo, String file) throws Exception {
        Path repoDir = dir.resolve("models--" + repo.replace("/", "--"));
        Path f = repoDir.resolve("snapshots").resolve(COMMIT).resolve(file);
        Files.createDirectories(f.getParent());
        Files.writeString(f, file);
        Files.createDirectories(repoDir.resolve("refs"));
        Files.writeString(repoDir.resolve("refs").resolve("main"), COMMIT);
        return f;
    }

    FornaxHub hub(List<ModelSet> catalogue) {
        return FornaxHub.builder().cacheRoot(dir).modelFolders(ModelFolders.none()).token(null).offline(true)
                .catalogue(catalogue).build();
    }

    static final List<ModelSet> SETS = List.of(
            ModelSet.of("flux", Kind.IMAGE, "Flux", "stable-diffusion.cpp",
                    Part.of("diffusion_model", "o/flux", "flux.safetensors"), Part.of("vae", "o/flux", "vae.safetensors"),
                    Part.of("llm", "o/flux", "llm.safetensors")).withAliases("image:flux"),
            ModelSet.of("ltx-2.5-tiny", Kind.VIDEO, "LTX", "stable-diffusion.cpp",
                    Part.of("diffusion_model", "o/ltx", "ltx.safetensors"), Part.of("audio_vae", "o/ltx", "audio.safetensors"),
                    Part.of("upscaler", "o/ltx", "up.safetensors")).withAliases("video:ltx-2.5-tiny"),
            ModelSet.of("song", Kind.SONG, "Song", "acestep.cpp", Part.of("dit", "o/song", "dit.gguf")));

    @Test void aSetSetsEveryPathOption() throws Exception {
        Path model = cached("o/flux", "flux.safetensors"), vae = cached("o/flux", "vae.safetensors"), llm = cached("o/flux", "llm.safetensors");
        var cfg = DiffusionModel.Config.builder().model("image:flux", hub(SETS)).set("vae_path", "/mine/vae.safetensors").build();
        assertEquals(Map.of("diffusion_model_path", model.toString(), "vae_path", "/mine/vae.safetensors", "llm_path", llm.toString()),
                cfg.options());   // a later option overrides the set's
        assertTrue(Files.isRegularFile(vae));
    }

    @Test void anLtxSetNeedsNoTypeRules() throws Exception {
        Path model = cached("o/ltx", "ltx.safetensors"), audio = cached("o/ltx", "audio.safetensors");
        Path up = cached("o/ltx", "up.safetensors");
        for (String name : List.of("ltx-2.5-tiny", "Video:LTX-2.5-tiny")) {   // by id or alias, in any case
            var o = DiffusionModel.Config.builder().model(name, hub(SETS)).build().options();
            assertEquals(model.toString(), o.get("diffusion_model_path"));
            assertEquals(audio.toString(), o.get("audio_vae_path"));
            assertFalse(o.containsKey("tensor_type_rules"), name);   // leejet/stable-diffusion.cpp#2432 made them unneeded
            assertFalse(o.containsValue(up.toString()));   // the upscaler is a request's hires.model_path
            assertEquals(2, o.size());
        }
        for (String f : List.of("flux.safetensors", "vae.safetensors", "llm.safetensors")) cached("o/flux", f);
        assertFalse(DiffusionModel.Config.builder().model("image:flux", hub(SETS)).build().options().containsKey("tensor_type_rules"));
    }

    @Test void aSetOfAnotherKindIsRefused() {
        var e = assertThrows(IllegalArgumentException.class, () -> DiffusionModel.Config.builder().model("song", hub(SETS)));
        assertEquals("song is a song model (acestep.cpp); DiffusionModel loads image and video models", e.getMessage());
    }

    @Test void aReferenceInAPathOptionBecomesItsFile() throws Exception {
        Path gguf = cached("city/FLUX-gguf", "flux-Q4_K_S.gguf");
        Path local = Files.writeString(dir.resolve("vae.safetensors"), "x");
        var cfg = new DiffusionModel.Config(Map.of("diffusion_model_path", "city/FLUX-gguf:Q4_K_S", "vae_path", local.toString(),
                "audio_vae_path", "", "flow_shift", 3.0, "lora_model_dir", "city/FLUX-gguf:Q4_K_S"));
        assertEquals(Map.of("diffusion_model_path", gguf.toString(), "vae_path", local.toString(), "audio_vae_path", "",
                "flow_shift", 3.0, "lora_model_dir", "city/FLUX-gguf:Q4_K_S"), cfg.resolved(hub(List.of())));   // only *_path options
    }

    @Test void everyPathOptionIsCheckedBeforeAnyIsFetched() {
        var cfg = new DiffusionModel.Config(new java.util.LinkedHashMap<>(Map.of("diffusion_model_path", "o/missing:Q8_0")));
        assertThrows(UncheckedIOException.class, () -> cfg.resolved(hub(List.of())));   // offline, not cached: the fetch was tried
        var both = new java.util.LinkedHashMap<String, Object>();
        both.put("diffusion_model_path", "o/missing:Q8_0");
        both.put("vae_path", "image:flux");
        var e = assertThrows(IllegalArgumentException.class, () -> new DiffusionModel.Config(both).resolved(hub(SETS)));
        assertTrue(e.getMessage().startsWith("image:flux is an image model (stable-diffusion.cpp)"), e.getMessage());
    }
}
