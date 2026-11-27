package org.liberationdata.fornax.snippets;

// --8<-- [start:pull]
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelCatalog;
import org.liberationdata.fornax.hub.ModelSet;

import java.nio.file.Path;
import java.util.Map;
// --8<-- [end:pull]
import org.liberationdata.fornax.hub.Eviction;
import org.liberationdata.fornax.hub.ModelStatus;
import org.liberationdata.fornax.hub.PullListener;
import org.liberationdata.fornax.hub.PullPlan;
import org.liberationdata.fornax.hub.PullProgress;
import org.liberationdata.fornax.hub.PullTask;

/** The snippets of website/src/content/docs/docs/guides/models.md: compiled, never run. */
class Models {
    interface Ui { void update(PullProgress progress); }

    void fetch(FornaxHub hub) throws Exception {
        // --8<-- [start:fetch]
        Map<String, Path> files = hub.fetch("unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M", PullListener.logging());
        Path gguf = files.get("gguf");

        Map<String, Path> vlm = hub.fetch("qwen3-vl-4b", PullListener.logging());   // an alias: "gguf" and "mmproj"
        // --8<-- [end:fetch]
        System.out.println(gguf + " " + vlm); // --8<-- [skip]
    }

    void pull() throws Exception {
        // --8<-- [start:pull]
        FornaxHub hub = FornaxHub.standard();
        ModelSet flux = ModelCatalog.require("flux2-klein");

        Map<String, Path> files = hub.pull(flux, p ->
                System.out.printf("%s %s %.0f%%%n", p.part().role(), p.phase(), p.fraction() * 100));

        Path diffusionModel = files.get("diffusion_model");
        Path vae = files.get("vae");
        Path textEncoder = files.get("llm");
        // --8<-- [end:pull]
        System.out.println(diffusionModel + " " + vae + " " + textEncoder); // --8<-- [skip]
    }

    Map<String, Path> async(FornaxHub hub, ModelSet flux, Ui ui) {
        // --8<-- [start:async]
        PullTask task = hub.pullAsync(flux, progress -> ui.update(progress));
        // later, if the user clicks Cancel:
        task.cancel();
        // or wait for it:
        Map<String, Path> files = task.result().join();
        // --8<-- [end:async]
        return files;
    }

    long plan(FornaxHub hub, ModelSet flux) throws Exception {
        // --8<-- [start:plan]
        PullPlan plan = hub.plan(flux);
        long bytes = plan.bytesToDownload();   // what is left to fetch, minus what an earlier pull left
        // --8<-- [end:plan]
        return bytes;
    }

    void status(FornaxHub hub, ModelSet flux) throws Exception {
        // --8<-- [start:status]
        ModelStatus s = hub.status(flux);
        s.state();         // PRESENT, PARTIAL or MISSING
        for (var part : s.parts())
            System.out.println(part.part().role() + " " + part.state() + " " + part.bytesOnDisk() + " / " + part.size());
        // --8<-- [end:status]
    }

    void evict(FornaxHub hub, ModelSet flux) throws Exception {
        // --8<-- [start:evict]
        Eviction e = hub.evict(flux);
        e.bytesFreed();
        e.kept();   // parts left in place because another model on disk uses the same file
        // --8<-- [end:evict]
    }

    void custom(FornaxHub hub) throws Exception {
        // --8<-- [start:custom]
        ModelSet mine = ModelSet.of("my-model", ModelSet.Kind.IMAGE, "My model", "stable-diffusion.cpp",
                ModelSet.Part.of("diffusion_model", "me/my-model", "model.safetensors"),
                ModelSet.Part.of("vae", "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"));
        hub.pull(mine, PullListener.NONE);
        // --8<-- [end:custom]
    }
}
