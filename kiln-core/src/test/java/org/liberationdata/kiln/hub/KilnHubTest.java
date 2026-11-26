package org.liberationdata.kiln.hub;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.kiln.hub.HubException.Reason;
import org.liberationdata.kiln.hub.ModelSet.Kind;
import org.liberationdata.kiln.hub.ModelSet.Part;
import org.liberationdata.kiln.hub.ModelStatus.State;
import org.liberationdata.kiln.hub.ModelStatus.Where;
import org.liberationdata.kiln.hub.PullProgress.Phase;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class KilnHubTest {
    static final String REPO = "test-org/Model-GGUF", OTHER = "other/vae-repo";

    @TempDir Path dir;
    FakeHub fake;
    Path cache;

    @BeforeEach void start() throws Exception {
        fake = new FakeHub();
        cache = dir.resolve("hub");
    }

    @AfterEach void stop() { fake.close(); }

    KilnHub hub() { return hub(fake.token).build(); }

    KilnHub.Builder hub(String token) {
        return KilnHub.builder().cacheRoot(cache).endpoint(fake.endpoint()).token(token).modelFolders(ModelFolders.none())
                .offline(false).reserveBytes(0).retryDelay(Duration.ofMillis(10)).catalogue(List.of());
    }

    static ModelSet set(String id, Part... parts) { return ModelSet.of(id, Kind.IMAGE, id, "test", parts); }

    @Test void pullsIntoTheHuggingFaceCacheLayout() throws Exception {
        var model = fake.add(REPO, "model-Q8_0.gguf", 300_000, true);
        var readme = fake.add(OTHER, "sub/config.json", 1_234, false);   // a git file: its etag is the git blob sha1
        var set = set("two", Part.of("model", REPO, "model-Q8_0.gguf"), Part.of("config", OTHER, "sub/config.json"));

        Map<String, Path> files = hub().pull(set, PullListener.NONE);

        String commit = FakeHub.commit(REPO);
        Path repoDir = cache.resolve("models--test-org--Model-GGUF");
        Path snap = repoDir.resolve("snapshots").resolve(commit).resolve("model-Q8_0.gguf");
        assertEquals(snap, files.get("model"));
        assertTrue(Files.isSymbolicLink(snap));
        assertEquals(Path.of("../../blobs/" + model.etag()), Files.readSymbolicLink(snap));
        assertArrayEquals(model.content(), Files.readAllBytes(snap));
        assertEquals(commit, Files.readString(repoDir.resolve("refs/main")));
        Path cfg = cache.resolve("models--other--vae-repo/snapshots").resolve(FakeHub.commit(OTHER)).resolve("sub/config.json");
        assertEquals(Path.of("../../../blobs/" + readme.etag()), Files.readSymbolicLink(cfg));
        assertArrayEquals(readme.content(), Files.readAllBytes(files.get("config")));
        assertFalse(Files.exists(repoDir.resolve("blobs").resolve(model.etag() + ".incomplete")));
        assertEquals(List.of(), fake.cdnAuth, "the token stays with the Hub");

        // The same files, found again: no request at all.
        int heads = fake.heads.get();
        List<Phase> phases = new ArrayList<>();
        assertEquals(files, hub().pull(set, p -> phases.add(p.phase())));
        assertEquals(heads, fake.heads.get());
        assertEquals(List.of(Phase.PRESENT, Phase.PRESENT), phases);
        assertEquals(State.PRESENT, hub().status(set).state());
        assertEquals(Where.HF_CACHE, hub().status(set).parts().get(0).where());
    }

    @Test void aGitFileTheHubServesItselfIsSizedByContentLengthAndNamedByItsETag() throws Exception {
        var f = fake.add(REPO, "config.json", 3_000, false);
        fake.gitFilesDirect = true;
        var set = set("one", Part.of("c", REPO, "config.json"));

        assertEquals(3_000, hub().plan(set).items().get(0).size());
        Path p = hub().pull(set, PullListener.NONE).get("c");

        assertArrayEquals(f.content(), Files.readAllBytes(p));
        assertEquals(Path.of("../../blobs/" + f.etag()), Files.readSymbolicLink(p));
        assertEquals(1, fake.gets(f));
        assertEquals(State.PRESENT, hub().status(set).state());
    }

    @Test void anEmptyFileIsPulledToo() throws Exception {
        var f = fake.add(REPO, "added_tokens.json", 0, false);
        var set = set("empty", Part.of("t", REPO, "added_tokens.json"));

        Path p = hub().pull(set, PullListener.NONE).get("t");

        assertEquals(0, Files.size(p));
        assertEquals(Path.of("../../blobs/" + f.etag()), Files.readSymbolicLink(p));
        assertEquals(State.PRESENT, hub().status(set).state());
    }

    @Test void reportsProgressForABar() throws Exception {
        var a = fake.add(REPO, "a.gguf", 2_000_000, true);
        var b = fake.add(REPO, "b.gguf", 1_000_000, true);
        var set = set("bar", Part.of("a", REPO, "a.gguf"), Part.of("b", REPO, "b.gguf"));
        List<PullProgress> events = Collections.synchronizedList(new ArrayList<>());

        hub().pull(set, events::add);

        assertEquals(Phase.CHECKING, events.get(0).phase());
        PullProgress last = events.get(events.size() - 1);
        assertEquals(Phase.DONE, last.phase());
        assertEquals("b", last.part().role());
        assertEquals(1, last.index());
        assertEquals(2, last.count());
        long total = a.content().length + b.content().length;
        assertEquals(total, last.setBytesTotal());
        assertEquals(total, last.setBytesDone());
        assertEquals(1.0, last.fraction());
        assertTrue(last.bytesPerSecond() > 0);
        assertEquals(Duration.ZERO, last.eta().orElseThrow());
        assertTrue(events.stream().anyMatch(e -> e.phase() == Phase.DOWNLOADING));
        long prev = -1;
        for (PullProgress e : events) {
            assertTrue(e.setBytesDone() >= prev, "the bar never goes back");
            prev = e.setBytesDone();
        }
    }

    @Test void resumesADroppedTransferWithARangeRequest() throws Exception {
        var f = fake.add(REPO, "big.gguf", 500_000, true);
        fake.dropNextAfter.set(200_000);

        Map<String, Path> files = hub().pull(set("one", Part.of("model", REPO, "big.gguf")), PullListener.NONE);

        assertArrayEquals(f.content(), Files.readAllBytes(files.get("model")));
        assertEquals(1, fake.ranges.size(), fake.ranges.toString());
        long from = Long.parseLong(fake.ranges.get(0).replaceAll(".* bytes=(\\d+)-", "$1"));
        assertTrue(from > 0 && from <= 200_000, "resumed from the bytes that arrived: " + from);
        assertEquals(2, fake.gets(f));
        assertEquals(200_000 + 500_000 - from, fake.bytesServed.get(), "only the missing bytes were sent again");
    }

    @Test void aChunkedTransferThatEndsShortIsResumedNotDeleted() throws Exception {
        var f = fake.add(REPO, "big.gguf", 300_000, true);
        fake.chunked = true;   // no Content-Length, so the short body ends cleanly rather than with an error
        fake.dropNextAfter.set(100_000);

        Map<String, Path> files = hub().pull(set("one", Part.of("model", REPO, "big.gguf")), PullListener.NONE);

        assertArrayEquals(f.content(), Files.readAllBytes(files.get("model")));
        assertEquals(List.of(f.etag() + " bytes=100000-"), fake.ranges);
    }

    @Test void aConnectionThatDropsOftenButMakesProgressEachTimeKeepsGoing() throws Exception {
        var f = fake.add(REPO, "big.gguf", 300_000, true);
        fake.chunked = true;
        fake.dropEveryAfter = 10_000;   // thirty short responses, far more than five

        Map<String, Path> files = hub().pull(set("one", Part.of("model", REPO, "big.gguf")), PullListener.NONE);

        assertArrayEquals(f.content(), Files.readAllBytes(files.get("model")));
        assertEquals(30, fake.gets(f));
    }

    @Test void aTransferThatStopsGettingAnywhereGivesUpAfterFiveTriesAndKeepsThePartialFile() throws Exception {
        var f = fake.add(REPO, "big.gguf", 300_000, true);
        var set = set("one", Part.of("model", REPO, "big.gguf"));
        fake.chunked = true;
        fake.dropEveryAfter = 10_000;
        fake.stallAt = 50_000;   // five tries that add bytes, then five that add none

        IOException e = assertThrows(IOException.class, () -> hub().pull(set, PullListener.NONE));

        assertFalse(e instanceof HubException, "a connection that keeps failing is not the Hub's answer: " + e);
        assertEquals(10, fake.gets(f), "five tries with progress, then five without, then it gives up");
        Path part = new CacheLayout(cache).incomplete(REPO, f.etag());
        assertEquals(50_000, Files.size(part), "each attempt's bytes are kept");

        fake.dropEveryAfter = 0;
        fake.stallAt = 0;
        Map<String, Path> files = hub().pull(set, PullListener.NONE);
        assertArrayEquals(f.content(), Files.readAllBytes(files.get("model")));
        assertTrue(fake.ranges.contains(f.etag() + " bytes=50000-"), "the next pull resumes: " + fake.ranges);
    }

    @Test void aPathWithSpacesUnicodeAndReservedCharactersIsEncoded() throws Exception {
        var a = fake.add(REPO, "dir one/naïve #1+2%.gguf", 1, true);
        var b = fake.add(REPO, "sub/x y?.json", 1, false);
        var set = set("odd", Part.of("a", REPO, a.path()), Part.of("b", REPO, b.path()));

        Map<String, Path> files = hub().pull(set, PullListener.NONE);

        assertArrayEquals(a.content(), Files.readAllBytes(files.get("a")));
        assertArrayEquals(b.content(), Files.readAllBytes(files.get("b")));
        assertEquals(State.PRESENT, hub().status(set).state());
    }

    @Test void aCancelledPullLeavesAPartialFileThatTheNextPullResumes() throws Exception {
        var f = fake.add(REPO, "big.gguf", 1_000_000, true);
        var set = set("one", Part.of("model", REPO, "big.gguf"));
        fake.pauseMillis = 20;
        CountDownLatch someBytes = new CountDownLatch(1);

        PullTask task = hub().pullAsync(set, p -> { if (p.bytesDone() > 100_000) someBytes.countDown(); });
        assertTrue(someBytes.await(30, TimeUnit.SECONDS));
        task.cancel();
        assertThrows(CancellationException.class, () -> task.result().get(30, TimeUnit.SECONDS));

        var status = hub().status(set);
        assertEquals(State.PARTIAL, status.state());
        var part = status.parts().get(0);
        assertEquals(State.PARTIAL, part.state());
        assertEquals(1_000_000, part.size(), "the expected size is known from the pull that started it");
        long partial = part.bytesOnDisk();
        assertTrue(partial > 100_000 && partial < 1_000_000, "partial: " + partial);
        assertEquals(partial, hub().plan(set).items().get(0).bytesOnDisk());
        assertEquals(1_000_000 - partial, hub().plan(set).bytesToDownload());

        fake.pauseMillis = 0;
        Map<String, Path> files = hub().pull(set, PullListener.NONE);
        assertArrayEquals(f.content(), Files.readAllBytes(files.get("model")));
        assertEquals(List.of(f.etag() + " bytes=" + partial + "-"), fake.ranges);
        assertEquals(State.PRESENT, hub().status(set).state());
    }

    @Test void interruptingABlockingPullCancelsIt() throws Exception {
        fake.add(REPO, "big.gguf", 1_000_000, true);
        fake.pauseMillis = 20;
        var set = set("one", Part.of("model", REPO, "big.gguf"));
        List<Throwable> thrown = new ArrayList<>();
        CountDownLatch someBytes = new CountDownLatch(1);
        Thread t = Thread.ofPlatform().start(() -> {
            try { hub().pull(set, p -> { if (p.bytesDone() > 0) someBytes.countDown(); }); }
            catch (Throwable e) { thrown.add(e); }
        });
        assertTrue(someBytes.await(30, TimeUnit.SECONDS));
        t.interrupt();
        t.join(30_000);
        assertFalse(t.isAlive());
        assertInstanceOf(CancellationException.class, thrown.get(0));
        assertEquals(State.PARTIAL, hub().status(set).state());
    }

    @Test void aWrongChecksumDeletesTheDownload() throws Exception {
        var f = fake.add(REPO, "model.gguf", 100_000, true);
        fake.corrupt(f);
        var set = set("one", Part.of("model", REPO, "model.gguf"));

        var e = assertThrows(HubException.class, () -> hub().pull(set, PullListener.NONE));

        assertEquals(Reason.CHECKSUM_MISMATCH, e.reason());
        assertTrue(e.getMessage().contains(f.etag()), e.getMessage());
        Path blobs = cache.resolve("models--test-org--Model-GGUF/blobs");
        assertFalse(Files.exists(blobs.resolve(f.etag())));
        assertFalse(Files.exists(blobs.resolve(f.etag() + ".incomplete")));
        assertEquals(State.MISSING, hub().status(set).state());
    }

    @Test void aWrongGitBlobHashIsCaughtToo() throws Exception {
        var f = fake.add(REPO, "config.json", 5_000, false);
        fake.corrupt(f);
        var e = assertThrows(HubException.class, () -> hub().pull(set("one", Part.of("c", REPO, "config.json")), PullListener.NONE));
        assertEquals(Reason.CHECKSUM_MISMATCH, e.reason());
        assertTrue(e.getMessage().contains("git blob sha1"), e.getMessage());
    }

    @Test void aGatedRepoWithoutATokenSaysToAcceptItsTerms() throws Exception {
        fake.add("gated/LTX", "vae.safetensors", 10_000, true);
        fake.gate("gated/LTX");
        var set = set("ltx", Part.of("vae", "gated/LTX", "vae.safetensors"));

        var e = assertThrows(HubException.class, () -> hub(null).build().pull(set, PullListener.NONE));
        assertEquals(Reason.ACCESS_DENIED, e.reason());
        assertEquals("gated/LTX", e.repo());
        assertTrue(e.getMessage().contains("Accept its terms on " + fake.endpoint() + "/gated/LTX"), e.getMessage());
        assertTrue(e.getMessage().contains("HF_TOKEN"), e.getMessage());

        var wrong = assertThrows(HubException.class, () -> hub("hf_wrong").build().pull(set, PullListener.NONE));
        assertEquals(Reason.ACCESS_DENIED, wrong.reason());
        assertTrue(wrong.getMessage().contains("A token was sent"), wrong.getMessage());

        assertTrue(Files.exists(hub().pull(set, PullListener.NONE).get("vae")));
        assertEquals(List.of(), fake.cdnAuth, "the token stays with the Hub");
    }

    @Test void aMissingFileOrRepoIsNotFound() {
        fake.add(REPO, "model.gguf", 10, true);
        var file = assertThrows(HubException.class, () -> hub().pull(set("x", Part.of("m", REPO, "nope.gguf")), PullListener.NONE));
        assertEquals(Reason.NOT_FOUND, file.reason());
        var repo = assertThrows(HubException.class, () -> hub().pull(set("x", Part.of("m", "no/such-repo", "m.gguf")), PullListener.NONE));
        assertEquals(Reason.NOT_FOUND, repo.reason());
        assertTrue(repo.getMessage().startsWith("no/such-repo: no such repo"), repo.getMessage());
    }

    @Test void followsARenamedRepoButKeepsTheNameAskedFor() throws Exception {
        var f = fake.add("Comfy-Org/Flux-2-Dev", "split_files/vae/flux2-vae.safetensors", 20_000, true);
        fake.rename("Comfy-Org/flux2-dev", "Comfy-Org/Flux-2-Dev");
        Path p = hub().pull(set("vae", Part.of("vae", "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors")), PullListener.NONE).get("vae");
        assertTrue(p.startsWith(cache.resolve("models--Comfy-Org--flux2-dev")), p.toString());
        assertArrayEquals(f.content(), Files.readAllBytes(p));
    }

    @Test void checksFreeDiskSpaceBeforeDownloading() {
        var f = fake.add(REPO, "model.gguf", 10_000, true);
        var e = assertThrows(HubException.class, () -> hub(fake.token).reserveBytes(Long.MAX_VALUE / 2).build()
                .pull(set("one", Part.of("m", REPO, "model.gguf")), PullListener.NONE));
        assertEquals(Reason.DISK_SPACE, e.reason());
        assertTrue(e.getMessage().contains("needs 10 KB"), e.getMessage());
        assertEquals(0, fake.gets(f));
    }

    @Test void offlineFailsAtOnceForAMissingPart() {
        fake.add(REPO, "model.gguf", 10, true);
        var e = assertThrows(HubException.class, () -> hub(null).offline(true).build()
                .pull(set("one", Part.of("m", REPO, "model.gguf")), PullListener.NONE));
        assertEquals(Reason.OFFLINE, e.reason());
        assertEquals(0, fake.heads.get());
    }

    @Test void twoPullsOfTheSameFileAtOnceDownloadItOnce() throws Exception {
        var f = fake.add(REPO, "big.gguf", 400_000, true);
        fake.pauseMillis = 10;
        var set = set("one", Part.of("model", REPO, "big.gguf"));
        List<Phase> second = Collections.synchronizedList(new ArrayList<>());
        CountDownLatch firstStarted = new CountDownLatch(1);

        PullTask a = hub().pullAsync(set, p -> { if (p.phase() == Phase.DOWNLOADING) firstStarted.countDown(); });
        assertTrue(firstStarted.await(30, TimeUnit.SECONDS));
        PullTask b = hub().pullAsync(set, p -> second.add(p.phase()));   // another KilnHub: only the lock file joins them

        Path pa = a.result().get(60, TimeUnit.SECONDS).get("model"), pb = b.result().get(60, TimeUnit.SECONDS).get("model");
        assertEquals(pa, pb);
        assertArrayEquals(f.content(), Files.readAllBytes(pa));
        assertEquals(1, fake.gets(f));
        assertTrue(second.contains(Phase.WAITING), second.toString());
        assertFalse(second.contains(Phase.DOWNLOADING), second.toString());
    }

    @Test void aPullInAnotherProcessIsWaitedFor() throws Exception {
        var f = fake.add(REPO, "big.gguf", 400_000, true);
        fake.pauseMillis = 25;   // 25 chunks: about 0.6 s of transfer, after the other JVM's start
        List<String> cmd = List.of(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-cp", System.getProperty("java.class.path"), PullMain.class.getName(),
                cache.toString(), fake.endpoint().toString(), REPO, "big.gguf");
        Process other = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        try {
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(60);
            while (fake.gets(f) == 0 && System.nanoTime() < deadline) Thread.sleep(20);
            assertEquals(1, fake.gets(f), "the other process started downloading");

            List<Phase> phases = Collections.synchronizedList(new ArrayList<>());
            Path p = hub().pull(set("one", Part.of("model", REPO, "big.gguf")), p2 -> phases.add(p2.phase())).get("model");

            assertTrue(other.waitFor(60, TimeUnit.SECONDS));
            String out = new String(other.getInputStream().readAllBytes());
            assertEquals(0, other.exitValue(), out);
            assertTrue(out.contains("pulled " + p), out);
            assertArrayEquals(f.content(), Files.readAllBytes(p));
            assertEquals(1, fake.gets(f), "downloaded once, by the other process");
            assertTrue(phases.contains(Phase.WAITING), phases.toString());
        } finally {
            other.destroyForcibly();
        }
    }

    @Test void findsPartsInAModelFolderAndDoesNotDownloadThem() throws Exception {
        fake.add(REPO, "flux.safetensors", 1_000, true);
        fake.add("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors", 1_000, true);
        Path comfy = dir.resolve("ComfyUI");
        Files.createDirectories(comfy.resolve("models/vae"));
        Files.write(comfy.resolve("models/vae/flux2-vae.safetensors"), new byte[7]);
        var set = set("flux", Part.of("diffusion_model", REPO, "flux.safetensors"),
                Part.of("vae", "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"));
        KilnHub hub = hub(fake.token).modelFolders(new ModelFolders(List.of(comfy))).build();

        var before = hub.status(set);
        assertEquals(State.PARTIAL, before.state());
        assertEquals(Where.MODEL_FOLDER, before.parts().get(1).where());
        assertEquals(7, before.parts().get(1).size());
        assertEquals(State.MISSING, before.parts().get(0).state());

        Map<String, Path> files = hub.pull(set, PullListener.NONE);
        assertEquals(comfy.resolve("models/vae/flux2-vae.safetensors"), files.get("vae"));
        assertEquals(1, fake.heads.get(), "only the missing part was asked about");
        assertEquals(files, hub.require(set));
    }

    @Test void theCacheComesBeforeAModelFolder() throws Exception {
        fake.add(OTHER, "split_files/vae/v.safetensors", 1_000, true);
        var set = set("v", Part.of("vae", OTHER, "split_files/vae/v.safetensors"));
        Path inCache = hub().pull(set, PullListener.NONE).get("vae");
        Path comfy = dir.resolve("ComfyUI");
        Files.createDirectories(comfy.resolve("vae"));
        Files.write(comfy.resolve("vae/v.safetensors"), new byte[7]);
        KilnHub hub = hub(fake.token).modelFolders(new ModelFolders(List.of(comfy))).build();

        assertEquals(inCache, hub.find(set.parts().get(0)).orElseThrow());
        assertEquals(Where.HF_CACHE, hub.status(set).parts().get(0).where());
        assertEquals(1_000, hub.status(set).parts().get(0).size());
    }

    @Test void aResumedPullsNumbersCountThePartialFileAndNotThePresentPart() throws Exception {
        var a = fake.add(REPO, "a.gguf", 300_000, true);
        var b = fake.add(REPO, "b.gguf", 1_000_000, true);
        hub().pull(set("a", Part.of("a", REPO, "a.gguf")), PullListener.NONE);
        var set = set("ab", Part.of("a", REPO, "a.gguf"), Part.of("b", REPO, "b.gguf"));
        fake.pauseMillis = 20;
        CountDownLatch some = new CountDownLatch(1);
        PullTask t = hub().pullAsync(set, p -> { if (p.phase() == Phase.DOWNLOADING && p.bytesDone() > 100_000) some.countDown(); });
        assertTrue(some.await(30, TimeUnit.SECONDS));
        t.cancel();
        assertThrows(CancellationException.class, () -> t.result().join());
        fake.pauseMillis = 0;
        long partial = hub().status(set).parts().get(1).bytesOnDisk();

        PullPlan plan = hub().plan(set);
        assertEquals(b.content().length, plan.bytesTotal(), "the present part isn't in the total");
        assertEquals(b.content().length - partial, plan.bytesToDownload());
        assertEquals(a.content().length, plan.items().get(0).size());

        List<PullProgress> events = Collections.synchronizedList(new ArrayList<>());
        hub().pull(set, events::add);
        List<PullProgress> after = events.stream().filter(e -> e.phase() != Phase.CHECKING).toList();
        assertEquals(Phase.PRESENT, after.get(0).phase());
        assertEquals(partial, after.get(0).setBytesDone(), "the bar starts where the last pull stopped");
        long prev = -1;
        for (PullProgress e : after) {
            assertEquals(b.content().length, e.setBytesTotal());
            assertTrue(e.setBytesDone() >= prev, "the bar never goes back");
            prev = e.setBytesDone();
        }
        PullProgress last = after.get(after.size() - 1);
        assertEquals(Phase.DONE, last.phase());
        assertEquals(b.content().length, last.setBytesDone());
        assertEquals(1.0, last.fraction());
    }

    @Test void requireNamesTheMissingParts() {
        var set = set("flux", Part.of("vae", REPO, "vae.safetensors"));
        var e = assertThrows(IllegalStateException.class, () -> hub().require(set));
        assertTrue(e.getMessage().contains("vae (" + REPO + " vae.safetensors)"), e.getMessage());
        assertTrue(e.getMessage().contains("kiln pull flux"), e.getMessage());
    }

    @Test void anAlternativeSourceAlreadyInTheCacheCounts() throws Exception {
        fake.add("second/repo", "te.safetensors", 1_000, true);
        hub().pull(set("x", Part.of("llm", "second/repo", "te.safetensors")), PullListener.NONE);
        int heads = fake.heads.get();
        var part = Part.anyOf("llm", new ModelSet.Source("first/repo", "te.safetensors"), new ModelSet.Source("second/repo", "te.safetensors"));
        assertTrue(hub().pull(set("y", part), PullListener.NONE).get("llm").toString().contains("models--second--repo"));
        assertEquals(heads, fake.heads.get());
    }

    @Test void evictKeepsWhatAnotherSetOnDiskUses() throws Exception {
        var vae = fake.add(OTHER, "vae.safetensors", 3_000, true);
        var ditA = fake.add(REPO, "a.gguf", 5_000, true);
        var ditB = fake.add(REPO, "b.gguf", 7_000, true);
        var a = set("a", Part.of("dit", REPO, "a.gguf"), Part.of("vae", OTHER, "vae.safetensors"));
        var b = set("b", Part.of("dit", REPO, "b.gguf"), Part.of("vae", OTHER, "vae.safetensors"));
        KilnHub hub = hub(fake.token).catalogue(List.of(a, b)).build();
        hub.pull(a, PullListener.NONE);
        hub.pull(b, PullListener.NONE);

        Eviction ea = hub.evict(a);
        assertEquals(5_000, ea.bytesFreed());
        assertEquals(List.of("vae"), ea.kept().stream().map(Part::role).toList());
        assertEquals(State.MISSING, hub.status(a).parts().get(0).state());
        assertEquals(State.PRESENT, hub.status(b).state());
        Path blobs = cache.resolve("models--test-org--Model-GGUF/blobs");
        assertFalse(Files.exists(blobs.resolve(ditA.etag())));
        assertTrue(Files.exists(blobs.resolve(ditB.etag())));

        Eviction eb = hub.evict(b);
        assertEquals(10_000, eb.bytesFreed());
        assertEquals(List.of(), eb.kept());
        assertEquals(State.MISSING, hub.status(b).state());
        assertFalse(Files.exists(cache.resolve("models--test-org--Model-GGUF")), "an empty repo folder goes too");
        assertFalse(Files.exists(cache.resolve("models--other--vae-repo")));
        assertNotNull(vae);
    }

    @Test void evictRemovesOnlyTheSourceThePartResolvesTo() throws Exception {
        var first = fake.add("first/repo", "te.safetensors", 1_000, true);
        var second = fake.add("second/repo", "te.safetensors", 2_000, true);
        var part = Part.anyOf("llm", new ModelSet.Source("first/repo", "te.safetensors"), new ModelSet.Source("second/repo", "te.safetensors"));
        var set = set("y", part);
        KilnHub hub = hub();
        // Another tool's download of the same file from the other repo, which Kiln did not pull for this set.
        hub.pull(set("x", Part.of("llm", "second/repo", "te.safetensors")), PullListener.NONE);
        hub.pull(set("z", Part.of("llm", "first/repo", "te.safetensors")), PullListener.NONE);
        assertTrue(hub.status(set).parts().get(0).path().toString().contains("models--first--repo"));

        Eviction e = hub.evict(set);
        assertEquals(1_000, e.bytesFreed());
        assertFalse(Files.exists(cache.resolve("models--first--repo")));
        assertTrue(Files.exists(cache.resolve("models--second--repo/blobs").resolve(second.etag())), "the other repo's file stays");
        assertTrue(hub.status(set).parts().get(0).path().toString().contains("models--second--repo"));
        assertNotNull(first);
    }

    @Test void evictRemovesAPartialOfAnotherSourceAndItsEmptiedRepo() throws Exception {
        fake.add("first/repo", "te.safetensors", 1_000, true);
        var part = Part.anyOf("llm", new ModelSet.Source("first/repo", "te.safetensors"), new ModelSet.Source("second/repo", "te.safetensors"));
        var set = set("y", part);
        KilnHub hub = hub();
        hub.pull(set("z", Part.of("llm", "first/repo", "te.safetensors")), PullListener.NONE);
        // A Kiln pull of the same file from the second repo that stopped part way.
        String etag = "a".repeat(64);
        Path incomplete = cache.resolve("models--second--repo/blobs").resolve(etag + ".incomplete");
        Files.createDirectories(incomplete.getParent());
        Files.write(incomplete, new byte[300]);
        Path pending = cache.resolve(".locks/models--second--repo").resolve(etag + ".pending");
        Files.createDirectories(pending.getParent());
        Files.writeString(pending, "te.safetensors\n2000\n");

        Eviction e = hub.evict(set);
        assertEquals(1_300, e.bytesFreed());
        assertFalse(Files.exists(pending));
        assertFalse(Files.exists(cache.resolve("models--first--repo")));
        assertFalse(Files.exists(cache.resolve("models--second--repo")), "the repo the partial emptied goes too");
        assertEquals(State.MISSING, hub.status(set).state());
    }

    @Test void evictKeepsABlobAnotherSnapshotStillLinksTo() throws Exception {
        var f = fake.add(REPO, "model.gguf", 4_000, true);
        var set = set("one", Part.of("m", REPO, "model.gguf"));
        KilnHub hub = hub();
        hub.pull(set, PullListener.NONE);
        // The same blob under another name, as an older snapshot (or `hf download` of a copy) can have it.
        Path other = cache.resolve("models--test-org--Model-GGUF/snapshots/0000000000000000000000000000000000000000/renamed.gguf");
        Files.createDirectories(other.getParent());
        Files.createSymbolicLink(other, Path.of("../../blobs/" + f.etag()));

        Eviction e = hub.evict(set);
        assertEquals(0, e.bytesFreed());
        assertTrue(Files.exists(cache.resolve("models--test-org--Model-GGUF/blobs").resolve(f.etag())));
        assertTrue(Files.isRegularFile(other));
        assertEquals(State.MISSING, hub.status(set).state());
    }

    @Test void evictRemovesAPartialDownload() throws Exception {
        fake.add(REPO, "big.gguf", 1_000_000, true);
        var set = set("one", Part.of("m", REPO, "big.gguf"));
        fake.pauseMillis = 20;
        CountDownLatch some = new CountDownLatch(1);
        PullTask t = hub().pullAsync(set, p -> { if (p.bytesDone() > 50_000) some.countDown(); });
        assertTrue(some.await(30, TimeUnit.SECONDS));
        t.cancel();
        assertThrows(CancellationException.class, () -> t.result().join());
        assertEquals(State.PARTIAL, hub().status(set).state());

        Eviction e = hub().evict(set);
        assertTrue(e.bytesFreed() > 0);
        assertEquals(State.MISSING, hub().status(set).state());
        assertFalse(Files.exists(cache.resolve("models--test-org--Model-GGUF")), "the emptied repo folder goes too");
    }

    @Test void aServerThatIgnoresRangeRestartsTheFileFromZero() throws Exception {
        var f = fake.add(REPO, "big.gguf", 500_000, true);
        fake.dropNextAfter.set(200_000);
        fake.ignoreRange = true;

        Map<String, Path> files = hub().pull(set("one", Part.of("model", REPO, "big.gguf")), PullListener.NONE);

        assertArrayEquals(f.content(), Files.readAllBytes(files.get("model")), "the 200 replaced the partial file, not appended to it");
        assertEquals(1, fake.ranges.size(), fake.ranges.toString());
        assertEquals(2, fake.gets(f));
    }

    @Test void aServerThatIgnoresRangeAndKeepsDroppingGivesUpOnceNoTryGetsFurther() {
        var f = fake.add(REPO, "big.gguf", 300_000, true);
        fake.ignoreRange = true;
        fake.chunked = true;
        fake.dropCycle = new long[] {20_000, 10_000};   // each restart from zero ends short, some shorter than the last
        var set = set("one", Part.of("model", REPO, "big.gguf"));

        assertTimeoutPreemptively(Duration.ofSeconds(20), () -> assertThrows(IOException.class, () -> hub().pull(set, PullListener.NONE)));
        assertEquals(6, fake.gets(f), "one try reaches 20 000 bytes, five more get no further, then it gives up");
    }

    @Test void aPinnedCommitIsItsOwnSnapshotAndWritesNoRef() throws Exception {
        var f = fake.add(REPO, "model.gguf", 2_000, true);
        String commit = FakeHub.commit(REPO);
        var pinned = set("pinned", new Part("m", List.of(new ModelSet.Source(REPO, "model.gguf", commit))));

        Path p = hub().pull(pinned, PullListener.NONE).get("m");

        Path repoDir = cache.resolve("models--test-org--Model-GGUF");
        assertEquals(repoDir.resolve("snapshots").resolve(commit).resolve("model.gguf"), p);
        assertArrayEquals(f.content(), Files.readAllBytes(p));
        assertFalse(Files.exists(repoDir.resolve("refs")), "a commit is not a branch: no refs/<commit>");
        // A commit that isn't on disk is not satisfied by another snapshot.
        var other = set("other", new Part("m", List.of(new ModelSet.Source(REPO, "model.gguf", "1".repeat(40)))));
        assertEquals(State.MISSING, hub().status(other).state());
    }

    @Test void aRevisionOrHubCommitThatLeavesTheCacheIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> new ModelSet.Source(REPO, "m.gguf", "../../../outside"));
        assertThrows(IllegalArgumentException.class, () -> new ModelSet.Source(REPO, "m.gguf", "/etc/x"));
        assertDoesNotThrow(() -> new ModelSet.Source(REPO, "m.gguf", "refs/pr/1"));

        fake.add(REPO, "model.gguf", 2_000, true);
        fake.commitHeader = "../../../../outside";
        var e = assertThrows(HubException.class, () -> hub().pull(set("one", Part.of("m", REPO, "model.gguf")), PullListener.NONE));
        assertEquals(Reason.PROTOCOL, e.reason());
        assertFalse(Files.exists(dir.resolve("outside")));
        assertFalse(Files.exists(cache.resolve("outside")));
    }

    @Test void anotherThreadGivingUpOnALockDoesNotReleaseItForOtherProcesses() throws Exception {
        var f = fake.add(REPO, "big.gguf", 1_000_000, true);
        var set = set("one", Part.of("m", REPO, "big.gguf"));
        fake.pauseMillis = 20;
        CountDownLatch some = new CountDownLatch(1);
        PullTask t = hub().pullAsync(set, p -> { if (p.bytesDone() > 50_000) some.countDown(); });
        try {
            assertTrue(some.await(30, TimeUnit.SECONDS));
            // Evict, in this JVM, tries the downloading blob's lock and finds it held. Closing a channel on a file
            // drops every lock this process has on it, so the evict must not have opened (and closed) one.
            hub().evict(set);
            Path lock = cache.resolve(".locks/models--test-org--Model-GGUF/" + f.etag() + ".lock");
            Process probe = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                    "-cp", System.getProperty("java.class.path"), LockProbe.class.getName(), lock.toString())
                    .redirectErrorStream(true).start();
            assertTrue(probe.waitFor(60, TimeUnit.SECONDS));
            assertEquals("held", new String(probe.getInputStream().readAllBytes()).trim());
        } finally {
            t.cancel();
        }
        assertThrows(CancellationException.class, () -> t.result().join());
    }
}
