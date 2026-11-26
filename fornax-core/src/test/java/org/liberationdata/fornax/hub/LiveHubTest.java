package org.liberationdata.fornax.hub;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelSet.Part;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Against the real Hub: one tiny public repo, a 454 KB LFS file and its config.json (a git file), into a temporary cache.
 * Off unless {@code FORNAX_LIVE_HUB=1}, so the build doesn't need the network.
 */
@EnabledIfEnvironmentVariable(named = "FORNAX_LIVE_HUB", matches = "1")
class LiveHubTest {
    @TempDir Path cache;

    @Test void pullsATinyRepoFromHuggingFace() throws Exception {
        String repo = "hf-internal-testing/tiny-random-gpt2";
        var set = ModelSet.of("tiny", Kind.CHAT, "tiny", "none", Part.of("model", repo, "model.safetensors"), Part.of("config", repo, "config.json"));
        FornaxHub hub = FornaxHub.builder().cacheRoot(cache).token(null).modelFolders(ModelFolders.none()).offline(false).build();

        assertEquals(453_864, hub.plan(set).items().get(0).size());
        Map<String, Path> files = hub.pull(set, p -> System.out.println(p.phase() + " " + p.part().role() + " " + p.bytesDone() + "/" + p.bytesTotal()));

        assertEquals(453_864, Files.size(files.get("model")));
        assertTrue(Files.isSymbolicLink(files.get("model")));
        assertTrue(Files.exists(cache.resolve("models--hf-internal-testing--tiny-random-gpt2/blobs/8111d5afb0715dbf5a31396d31432cb56370ba23f6650a035ea0fc8a20b4e500")));
        assertTrue(Files.size(files.get("config")) > 0);
        assertEquals(ModelStatus.State.PRESENT, hub.status(set).state());
    }
}
