package org.liberationdata.fornax.hub;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelSet.Part;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** {@link FornaxHub#fetch} by model reference and alias, against a fake Hub. */
class FetchTest {
    static final String REPO = "org/Model-GGUF";
    @TempDir Path dir;
    FakeHub fake;
    Path cache;

    @BeforeEach void start() throws Exception {
        fake = new FakeHub();
        cache = dir.resolve("hub");
        fake.add(REPO, "m-Q4_K_M.gguf", 20_000, true);
        fake.add(REPO, "m-Q8_0.gguf", 30_000, true);
        fake.add(REPO, "mmproj-m-F16.gguf", 5_000, true);
        fake.add(REPO, "README.md", 100, false);
    }

    @AfterEach void stop() { fake.close(); }

    FornaxHub.Builder builder() {
        return FornaxHub.builder().cacheRoot(cache).endpoint(fake.endpoint()).token(null).modelFolders(ModelFolders.none())
                .offline(false).reserveBytes(0).retryDelay(Duration.ofMillis(10)).catalogue(List.of());
    }

    FornaxHub hub() { return builder().build(); }

    @Test void aTagDownloadsThatQuantAndTheNextFetchNeedsNoNetwork() throws Exception {
        Map<String, Path> files = hub().fetch(REPO + ":q4_k_m", PullListener.NONE);
        assertEquals(List.of("gguf"), List.copyOf(files.keySet()));
        assertEquals("m-Q4_K_M.gguf", files.get("gguf").getFileName().toString());
        assertEquals(20_000, Files.size(files.get("gguf")));
        assertEquals(List.of(REPO + "@main"), fake.listings);

        int heads = fake.heads.get();
        var offline = builder().offline(true).build();
        assertEquals(files, offline.fetch(REPO + ":Q4_K_M", PullListener.NONE));   // from the cache, no listing or HEAD
        assertEquals(1, fake.listings.size());
        assertEquals(heads, fake.heads.get());
    }

    @Test void aFileByName() throws Exception {
        assertEquals("README.md", hub().fetch(REPO + ":README.md", PullListener.NONE).get("model").getFileName().toString());
        assertEquals("m-Q8_0.gguf", hub().fetch(REPO + ":m-Q8_0.gguf", PullListener.NONE).get("gguf").getFileName().toString());
    }

    @Test void aRepoWithOneModelNeedsNoTagAndTheMmprojIsntAModel() throws Exception {
        fake.add("solo/One-GGUF", "one-Q4_0.gguf", 1_000, true);
        fake.add("solo/One-GGUF", "mmproj-one.gguf", 1_000, true);
        assertEquals("one-Q4_0.gguf", hub().fetch("solo/One-GGUF", PullListener.NONE).get("gguf").getFileName().toString());
    }

    @Test void aSplitModelBringsEveryPart() throws Exception {
        fake.add("big/Split-GGUF", "Q6_K/big-Q6_K-00001-of-00002.gguf", 1_000, true);
        fake.add("big/Split-GGUF", "Q6_K/big-Q6_K-00002-of-00002.gguf", 1_000, true);
        Map<String, Path> files = hub().fetch("big/Split-GGUF:Q6_K", PullListener.NONE);
        assertEquals("big-Q6_K-00001-of-00002.gguf", files.get("gguf").getFileName().toString());
        assertTrue(Files.isRegularFile(files.get("Q6_K/big-Q6_K-00002-of-00002.gguf")));
        assertEquals(files.get("gguf").getParent(), files.get("Q6_K/big-Q6_K-00002-of-00002.gguf").getParent());
    }

    @Test void aPartlyDownloadedSplitModelIsFinishedNotUsed() throws Exception {
        fake.add("big/Split-GGUF", "big-00001-of-00002.gguf", 1_000, true);
        fake.add("big/Split-GGUF", "big-00002-of-00002.gguf", 1_000, true);
        hub().pull(ModelSet.of("p1", Kind.CHAT, "p1", "t", Part.of("gguf", "big/Split-GGUF", "big-00001-of-00002.gguf")), PullListener.NONE);
        Map<String, Path> files = hub().fetch("big/Split-GGUF", PullListener.NONE);
        assertTrue(Files.isRegularFile(files.get("big-00002-of-00002.gguf")));
        assertEquals(1, fake.listings.size());   // part 1 alone in the cache wasn't taken as the model
    }

    @Test void anAmbiguousTagNamesTheChoices() {
        fake.add(REPO, "m-UD-Q4_K_M.gguf", 1_000, true);
        var e = assertThrows(IllegalArgumentException.class, () -> hub().fetch(REPO + ":Q4_K_M", PullListener.NONE));
        assertTrue(e.getMessage().contains(REPO + ":m-UD-Q4_K_M.gguf") && e.getMessage().contains(REPO + ":m-Q4_K_M.gguf"), e.getMessage());
        var none = assertThrows(IllegalArgumentException.class, () -> hub().fetch(REPO, PullListener.NONE));
        assertTrue(none.getMessage().contains("several GGUF models"), none.getMessage());
    }

    @Test void aTagThatMatchesNothingListsTheModels() {
        var e = assertThrows(HubException.class, () -> hub().fetch(REPO + ":IQ2_XS", PullListener.NONE));
        assertEquals(HubException.Reason.NOT_FOUND, e.reason());
        assertTrue(e.getMessage().contains("m-Q8_0.gguf"), e.getMessage());
        var repo = assertThrows(HubException.class, () -> hub().fetch("no/Such-GGUF:Q4_K_M", PullListener.NONE));
        assertEquals(HubException.Reason.NOT_FOUND, repo.reason());
    }

    @Test void aRevisionAsksTheHubForIt() throws Exception {
        hub().fetch(REPO + "@" + FakeHub.commit(REPO) + ":Q8_0", PullListener.NONE);
        assertEquals(List.of(REPO + "@" + FakeHub.commit(REPO)), fake.listings);
        assertThrows(HubException.class, () -> hub().fetch(REPO + "@nope:Q8_0", PullListener.NONE));
    }

    @Test void aCommitInTheCacheNeedsNoHubOffline() throws Exception {
        String ref = REPO + "@" + FakeHub.commit(REPO) + ":Q8_0";
        Map<String, Path> files = hub().fetch(ref, PullListener.NONE);
        assertEquals(files, builder().offline(true).build().fetch(ref, PullListener.NONE));
        assertEquals(1, fake.listings.size());
    }

    @Test void aNamedRevisionIsntAnsweredByMainsOlderCopy() throws Exception {
        // an older main in the cache: snapshot "b...", refs/main naming it, its own m-Q8_0.gguf
        Path repo = cache.resolve("models--org--Model-GGUF"), old = repo.resolve("snapshots/" + "b".repeat(40) + "/m-Q8_0.gguf");
        Files.createDirectories(old.getParent());
        Files.writeString(old, "older");
        Files.createDirectories(repo.resolve("refs"));
        Files.writeString(repo.resolve("refs/main"), "b".repeat(40));
        Path got = hub().fetch(REPO + "@main:Q8_0", PullListener.NONE).get("gguf");
        assertEquals(30_000, Files.size(got));   // the commit main names on the Hub, downloaded
        assertEquals(List.of(REPO + "@main"), fake.listings);
    }

    @Test void anAliasFetchesItsWholeSet() throws Exception {
        var vlm = ModelSet.of("tiny-vl", Kind.VISION, "Tiny VL", "llama.cpp",
                Part.of("gguf", REPO, "m-Q4_K_M.gguf"), Part.of("mmproj", REPO, "mmproj-m-F16.gguf")).withAliases("tiny");
        var hub = builder().catalogue(List.of(vlm)).build();
        Map<String, Path> files = hub.fetch("TINY", PullListener.NONE);
        assertEquals(List.of("gguf", "mmproj"), List.copyOf(files.keySet()));
        assertTrue(fake.listings.isEmpty());   // an alias names its files: no listing
    }

    @Test void urlsAndNonsenseAreRefused() {
        var e = assertThrows(IllegalArgumentException.class, () -> hub().fetch(
                "https://huggingface.co/" + REPO + "/resolve/main/m-Q8_0.gguf", PullListener.NONE));
        assertTrue(e.getMessage().contains("use " + REPO + ":m-Q8_0.gguf"), e.getMessage());
        assertThrows(IllegalArgumentException.class, () -> hub().fetch("not a model", PullListener.NONE));
    }

    @Test void isReferenceTellsPathsFromNames() throws Exception {
        var vlm = ModelSet.of("tiny-vl", Kind.VISION, "Tiny VL", "llama.cpp", Part.of("gguf", REPO, "m-Q4_K_M.gguf"));
        var hub = builder().catalogue(List.of(vlm)).build();
        Path real = Files.writeString(dir.resolve("x.gguf"), "");
        assertFalse(hub.isReference(real.toString()));
        assertFalse(hub.isReference("models/missing.gguf"));
        assertFalse(hub.isReference(""));
        assertTrue(hub.isReference("tiny-vl"));
        assertTrue(hub.isReference(REPO + ":Q8_0"));
        assertTrue(hub.isReference(REPO));
    }

    @Test void theListingGoesThroughTheBuildersClient() throws Exception {
        // a host only the client's proxy can reach: the listing must use the client given, as downloads do
        var viaProxy = java.net.http.HttpClient.newBuilder().followRedirects(java.net.http.HttpClient.Redirect.NEVER)
                .proxy(java.net.ProxySelector.of(new java.net.InetSocketAddress(fake.endpoint().getHost(), fake.endpoint().getPort())))
                .build();
        var hub = builder().endpoint(java.net.URI.create("http://hub.invalid")).httpClient(viaProxy).build();
        assertEquals("m-Q8_0.gguf", hub.resolve(ModelRef.parse(REPO + ":Q8_0")).parts().get(0).primary().file());
        assertEquals(List.of(REPO + "@main"), fake.listings);
    }

    @Test void offlineWithNothingCachedSaysSo() {
        var e = assertThrows(HubException.class, () -> builder().offline(true).build().fetch(REPO + ":Q8_0", PullListener.NONE));
        assertEquals(HubException.Reason.OFFLINE, e.reason());
    }
}
