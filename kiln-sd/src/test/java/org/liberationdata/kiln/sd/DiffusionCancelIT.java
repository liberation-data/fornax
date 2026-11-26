package org.liberationdata.kiln.sd;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.HfCache;

import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Cancel and a throwing progress sink against FLUX.2 [klein] 4B at 512x512, when its files are in the Hugging Face
 * cache; skipped otherwise. No run is allowed to finish: each stops within its first steps.
 */
class DiffusionCancelIT {
    static final int STEPS = 8;
    static DiffusionModel model;

    @BeforeAll
    static void load() {
        var dit = HfCache.find("black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors");
        var vae = HfCache.find("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors");
        var llm = HfCache.find("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors");
        assumeTrue(dit.isPresent() && vae.isPresent() && llm.isPresent(), "FLUX.2 klein not downloaded");
        model = new DiffusionModel(DiffusionModel.Config.builder()
                .diffusionModel(dit.get().toString()).vae(vae.get().toString()).llm(llm.get().toString()).build());
    }

    @AfterAll
    static void close() {
        if (model == null) return;
        model.close();
        model.close();  // idempotent
    }

    @AfterEach
    void quiet() { DiffusionModel.onProgress((s, n, t) -> {}); }

    static DiffusionModel.Request request() { return new DiffusionModel.Request("a kiln at dusk", 512, 512, STEPS, 4.0, 7); }

    @Test
    void cancelFromTheProgressSinkStopsTheRun() {
        AtomicInteger last = new AtomicInteger();
        DiffusionModel.onProgress((step, steps, t) -> {
            if (steps != STEPS) return;  // the same callback reports loading the weights, on the first run
            last.set(step);
            if (step >= 2) model.cancel();
        });
        assertThrows(CancellationException.class, () -> model.generate(request()));
        assertTrue(last.get() < STEPS, "ran to step " + last.get());
    }

    @Test
    void cancellingTheFutureStopsAnAsyncRun() throws Exception {
        var firstStep = new java.util.concurrent.CountDownLatch(1);
        AtomicInteger last = new AtomicInteger();
        DiffusionModel.onProgress((step, steps, t) -> {
            if (steps != STEPS) return;
            last.set(step);
            firstStep.countDown();
        });
        var f = model.generateAsync(request());
        assertTrue(firstStep.await(120, TimeUnit.SECONDS));
        f.cancel(true);
        assertThrows(CancellationException.class, f::join);
        // the next call waits for the cancelled run to return; it, too, is cancelled at its first step
        DiffusionModel.onProgress((step, steps, t) -> model.cancel());
        assertThrows(CancellationException.class, () -> model.generate(request()));
        assertTrue(last.get() < STEPS, "the cancelled run went on to step " + last.get());
    }

    @Test
    void aThrowingProgressSinkSurfacesAndStopsTheRun() {
        var boom = new IllegalStateException("progress sink broke");
        DiffusionModel.onProgress((step, steps, t) -> { throw boom; });
        assertSame(boom, assertThrows(IllegalStateException.class, () -> model.generate(request())));
        var e = assertThrows(ExecutionException.class, () -> model.generateAsync(request()).get(120, TimeUnit.SECONDS));
        assertSame(boom, e.getCause());
    }
}
