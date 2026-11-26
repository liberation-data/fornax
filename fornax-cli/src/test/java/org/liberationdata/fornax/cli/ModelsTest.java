package org.liberationdata.fornax.cli;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.HubException;
import org.liberationdata.fornax.hub.ModelFolders;
import org.liberationdata.fornax.hub.ModelSet.Kind;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class ModelsTest {
    @TempDir Path root;

    FornaxHub hub() throws Exception {
        Path cache = root.resolve("hub");
        put(cache, "Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q8_0.gguf");
        put(cache, "some/Multi-GGUF", "m-Q4_K_M.gguf");
        put(cache, "some/Multi-GGUF", "m-Q8_0.gguf");
        put(cache, "some/Multi-GGUF", "mmproj-m-F16.gguf");
        put(cache, "Serveurperso/ACE-Step-1.5-GGUF", "vae-BF16.gguf");   // one part of the ACE-Step sets
        put(cache, "ggerganov/whisper.cpp", "ggml-base.en.bin");
        return FornaxHub.builder().cacheRoot(cache).modelFolders(ModelFolders.none()).offline(true).build();
    }

    /** A file in the hub layout: a blob, and a snapshot link to it. */
    static void put(Path cache, String repo, String file) throws Exception {
        Path repoDir = cache.resolve("models--" + repo.replace("/", "--"));
        Path blob = repoDir.resolve("blobs/" + file + ".blob");
        Files.createDirectories(blob.getParent());
        Files.writeString(blob, file);
        Path snap = repoDir.resolve("snapshots/abc/" + file);
        Files.createDirectories(snap.getParent());
        Files.createSymbolicLink(snap, snap.getParent().relativize(blob));
    }

    @Test
    void resolvesPathsAliasesAndRepos() throws Exception {
        var hub = hub();
        Path file = Files.writeString(root.resolve("local.gguf"), "x");
        assertEquals(file, ((Models.Llm) Models.resolve(file.toString(), hub)).path());

        var alias = (Models.Llm) Models.resolve("qwen3-0.6b", hub);
        assertEquals("Qwen3-0.6B-Q8_0.gguf", alias.path().getFileName().toString());
        assertEquals(alias.path(), ((Models.Llm) Models.resolve("Qwen/Qwen3-0.6B-GGUF", hub)).path());   // one model in the repo
        assertEquals("m-Q8_0.gguf", ((Models.Llm) Models.resolve("some/Multi-GGUF:q8_0", hub)).path().getFileName().toString());
        assertEquals("m-Q4_K_M.gguf", ((Models.Llm) Models.resolve("some/Multi-GGUF:m-Q4_K_M.gguf", hub)).path().getFileName().toString());
    }

    @Test
    void explainsWhatIsMissing() throws Exception {
        var hub = hub();
        var notPulled = assertThrows(Main.Failure.class, () -> Models.resolve("qwen3-4b", hub));
        assertTrue(notPulled.getMessage().contains("fornax pull qwen3-4b"), notPulled.getMessage());
        var partOfASet = assertThrows(Main.Failure.class, () -> Models.resolve("ace-step", hub));
        assertTrue(partOfASet.getMessage().contains("acestep-v15-turbo-Q8_0.gguf"), partOfASet.getMessage());
        assertFalse(partOfASet.getMessage().contains("vae-BF16"), partOfASet.getMessage());
        var several = assertThrows(Main.Failure.class, () -> Models.resolve("some/Multi-GGUF", hub));
        assertTrue(several.getMessage().contains("some/Multi-GGUF:m-Q8_0.gguf"), several.getMessage());
        assertFalse(several.getMessage().contains("mmproj"), several.getMessage());
        var none = assertThrows(Main.Failure.class, () -> Models.resolve("some/Multi-GGUF:IQ2", hub));
        assertTrue(none.getMessage().contains("fornax pull some/Multi-GGUF:IQ2"), none.getMessage());
        var revision = assertThrows(Main.Failure.class, () -> Models.resolve("some/Multi-GGUF@dev:Q8_0", hub));
        assertTrue(revision.getMessage().contains("drop @dev"), revision.getMessage());
        var unknown = assertThrows(Main.Failure.class, () -> Models.resolve("what", hub));
        assertTrue(unknown.getMessage().contains("qwen3-4b"), unknown.getMessage());
    }

    @Test
    void fetchTakesOneKindAndPullsWhatIsMissing() throws Exception {
        var hub = hub();
        var err = new ByteArrayOutputStream();
        var ps = new PrintStream(err, true, StandardCharsets.UTF_8);
        var whisper = Models.fetch("whisper-base.en", Kind.TRANSCRIPTION, "transcribe", true, hub, ps, false);
        assertEquals("ggml-base.en.bin", whisper.file("model").getFileName().toString());
        assertEquals("whisper-base.en", whisper.set().id());
        assertEquals("", err.toString(StandardCharsets.UTF_8), "nothing to download, so nothing said");

        Path file = Files.writeString(root.resolve("my.bin"), "x");   // a path, for a command that loads one file
        assertEquals(file, Models.fetch(file.toString(), Kind.TRANSCRIPTION, "transcribe", true, hub, ps, false).file("model"));
        assertEquals(Path.of("Qwen3-0.6B-Q8_0.gguf"), Models.fetch("Qwen/Qwen3-0.6B-GGUF:Q8_0", Kind.VISION, "see", true, hub, ps, false)
                .file("gguf").getFileName());

        // missing: pulled, which offline stops at the first file, after saying what it is fetching
        var offline = assertThrows(HubException.class, () -> Models.fetch("whisper-large-v3-turbo", Kind.TRANSCRIPTION, "transcribe", true, hub, ps, false));
        assertEquals(HubException.Reason.OFFLINE, offline.reason());
        assertTrue(err.toString(StandardCharsets.UTF_8).contains("[fornax] downloading whisper-large-v3-turbo: Whisper large-v3-turbo"), err.toString());
    }

    @Test
    void fetchRefusesTheWrongKindAndUnknownNamesBeforeDownloading() throws Exception {
        var hub = hub();
        var ps = new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8);
        var chat = assertThrows(Main.Failure.class, () -> Models.fetch("qwen3-4b", Kind.IMAGE, "image", false, hub, ps, false));
        assertEquals("qwen3-4b is a chat model; image takes an image model: flux2-klein, qwen-image, qwen-image-edit", chat.getMessage());
        var song = assertThrows(Main.Failure.class, () -> Models.fetch("flux2-klein", Kind.SONG, "song", false, hub, ps, false));
        assertTrue(song.getMessage().startsWith("flux2-klein is an image model; song takes a song model: ace-step-turbo"), song.getMessage());
        var stt = assertThrows(Main.Failure.class, () -> Models.fetch("kokoro", Kind.TRANSCRIPTION, "transcribe", true, hub, ps, false));
        assertTrue(stt.getMessage().contains("takes a speech-to-text model: whisper-large-v3-turbo, whisper-base.en"), stt.getMessage());
        var ref = assertThrows(Main.Failure.class, () -> Models.fetch("some/Multi-GGUF:Q8_0", Kind.IMAGE, "image", false, hub, ps, false));
        assertTrue(ref.getMessage().contains("not a reference"), ref.getMessage());
        var unknown = assertThrows(Main.Failure.class, () -> Models.fetch("what", Kind.SONG, "song", false, hub, ps, false));
        assertTrue(unknown.getMessage().startsWith("No such model: what") && unknown.getMessage().contains("ace-step-turbo"), unknown.getMessage());
        var unknownFile = assertThrows(Main.Failure.class, () -> Models.fetch("what.bin", Kind.TRANSCRIPTION, "transcribe", true, hub, ps, false));
        assertTrue(unknownFile.getMessage().startsWith("No such file or model: what.bin"), unknownFile.getMessage());
        var notWhisper = assertThrows(Main.Failure.class, () -> Models.fetch("Qwen/Qwen3-0.6B-GGUF:Q8_0", Kind.TRANSCRIPTION, "transcribe", true, hub, ps, false));
        assertTrue(notWhisper.getMessage().contains("not a whisper.cpp model"), notWhisper.getMessage());
    }

    @Test
    void aSplitModelMissingAPartIsIncomplete() throws Exception {
        var hub = hub();
        put(hub.cacheRoot(), "some/Split-GGUF", "s-Q4_K_M-00001-of-00002.gguf");   // part 2 never arrived
        var e = assertThrows(Main.Failure.class, () -> Models.resolve("some/Split-GGUF:Q4_K_M", hub));
        assertTrue(e.getMessage().contains("incomplete") && e.getMessage().contains("s-Q4_K_M-00002-of-00002.gguf"), e.getMessage());
        assertEquals(java.util.List.of(), Listing.otherLlms(hub.cacheRoot()).stream()
                .filter(x -> x.repo().equals("some/Split-GGUF")).toList());
    }

    @Test
    void listShowsTheCatalogueAndOtherModels() throws Exception {
        var out = new ByteArrayOutputStream();
        assertEquals(0, Listing.run(hub(), new PrintStream(out, true, StandardCharsets.UTF_8)));
        String s = out.toString(StandardCharsets.UTF_8);
        assertTrue(s.contains("Chat (fornax chat <alias>):\n  qwen3-0.6b"), s);
        assertTrue(s.contains("ace-step-turbo") && s.contains("partly downloaded"), s);   // the VAE alone
        assertTrue(s.contains("some/Multi-GGUF:m-Q8_0.gguf"), s);
        assertFalse(s.contains("Qwen/Qwen3-0.6B-GGUF:"), s);   // a catalogue model shows by its alias
        assertFalse(s.contains("mmproj"), s);
        assertTrue(s.contains("Not downloaded (fornax pull <alias>): ") && s.contains("flux2-klein"), s);
    }
}
