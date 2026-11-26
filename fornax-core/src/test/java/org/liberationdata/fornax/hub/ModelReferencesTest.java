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
import java.util.Optional;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/** How an engine's constructor turns a model reference into files, against a fake Hub. */
class ModelReferencesTest {
    static final ModelReferences.Engine STT = new ModelReferences.Engine("WhisperModel", "speech-to-text models",
            Set.of(Kind.TRANSCRIPTION), Set.of("model"), "a whisper.cpp model");
    static final ModelReferences.Engine STEMS = new ModelReferences.Engine("DemucsModel", "stem-separation models",
            Set.of(), Set.of("gguf"), "a stems.cpp GGUF");

    @TempDir Path dir;
    FakeHub fake;

    @BeforeEach void start() throws Exception {
        fake = new FakeHub();
        fake.add("ggerganov/whisper.cpp", "ggml-base.en.bin", 2_000, true);
        fake.add("ggerganov/whisper.cpp", "ggml-tiny.bin", 1_000, true);
        fake.add("stems/htdemucs-GGUF", "htdemucs-F32.gguf", 1_000, true);
        fake.add("stems/htdemucs-GGUF", "htdemucs_6s-F32.gguf", 1_000, true);
    }

    @AfterEach void stop() { fake.close(); }

    FornaxHub hub(List<ModelSet> catalogue) {
        return FornaxHub.builder().cacheRoot(dir.resolve("hub")).endpoint(fake.endpoint()).token(null)
                .modelFolders(ModelFolders.none()).offline(false).reserveBytes(0).retryDelay(Duration.ofMillis(10))
                .catalogue(catalogue).build();
    }

    @Test void aPathIsUsedAsGivenWithoutTheHub() throws Exception {
        Path file = Files.writeString(dir.resolve("model.bin"), "x");
        assertTrue(ModelReferences.isPath(file.toString()));
        assertEquals(Optional.empty(), ModelReferences.locate(file.toString(), STT, hub(List.of())));
        assertEquals(Optional.empty(), ModelReferences.locate("/not/there.bin", STT, hub(List.of())));   // a path, missing: the engine says so
        assertEquals(List.of(), fake.listings);
        assertEquals(0, fake.heads.get());
    }

    @Test void aReferenceIsDownloadedAndTheEnginesFileFound() {
        Map<String, Path> files = ModelReferences.locate("ggerganov/whisper.cpp:ggml-base.en.bin", STT, hub(List.of())).orElseThrow();
        assertEquals("ggml-base.en.bin", ModelReferences.file(files, STT).getFileName().toString());
        Map<String, Path> stems = ModelReferences.locate("stems/htdemucs-GGUF:htdemucs_6s-F32.gguf", STEMS, hub(List.of())).orElseThrow();
        assertEquals("htdemucs_6s-F32.gguf", ModelReferences.file(stems, STEMS).getFileName().toString());
    }

    @Test void anAliasOfTheEnginesKindIsDownloaded() {
        var base = ModelSet.of("whisper-base.en", Kind.TRANSCRIPTION, "base.en", "whisper.cpp",
                Part.of("model", "ggerganov/whisper.cpp", "ggml-base.en.bin"));
        Map<String, Path> files = ModelReferences.locate("whisper-base.en", STT, hub(List.of(base))).orElseThrow();
        assertEquals("ggml-base.en.bin", files.get("model").getFileName().toString());
    }

    @Test void anAliasOfAnotherKindIsRefusedBeforeAnyRequest() {
        var speech = ModelSet.of("kokoro", Kind.SPEECH, "Kokoro", "llama.cpp", Part.of("gguf", "stems/htdemucs-GGUF", "htdemucs-F32.gguf"));
        var e = assertThrows(IllegalArgumentException.class, () -> ModelReferences.locate("kokoro", STT, hub(List.of(speech))));
        assertEquals("kokoro is a speech model (llama.cpp); WhisperModel loads speech-to-text models", e.getMessage());
        // an engine with no catalogue kind refuses every alias
        assertThrows(IllegalArgumentException.class, () -> ModelReferences.locate("kokoro", STEMS, hub(List.of(speech))));
        assertEquals(List.of(), fake.listings);
        assertEquals(0, fake.heads.get());
    }

    @Test void aReferenceToTheWrongSortOfFileIsRefusedAfterTheListingOnly() {
        var e = assertThrows(IllegalArgumentException.class,
                () -> ModelReferences.locate("ggerganov/whisper.cpp:ggml-tiny.bin", STEMS, hub(List.of())));
        assertTrue(e.getMessage().startsWith("ggerganov/whisper.cpp:ggml-tiny.bin isn't a stems.cpp GGUF (it names [ggml-tiny.bin]"),
                e.getMessage());
        assertThrows(IllegalArgumentException.class, () -> ModelReferences.locate("stems/htdemucs-GGUF:htdemucs-F32.gguf", STT, hub(List.of())));
        assertEquals(2, fake.listings.size());
        assertEquals(0, fake.heads.get());   // nothing downloaded
        assertTrue(fake.gets.isEmpty());
    }

    @Test void aUrlIsRefusedWithTheReferenceToUse() {
        var e = assertThrows(IllegalArgumentException.class, () -> ModelReferences.locate(
                "https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.en.bin", STT, hub(List.of())));
        assertTrue(e.getMessage().contains("ggerganov/whisper.cpp:ggml-base.en.bin"), e.getMessage());
    }

    @Test void aMissingFileFailsAsTheHubSays() {
        var e = assertThrows(java.io.UncheckedIOException.class,
                () -> ModelReferences.locate("ggerganov/whisper.cpp:ggml-huge.bin", STT, hub(List.of())));
        assertInstanceOf(HubException.class, e.getCause());
        assertTrue(e.getMessage().startsWith("ggerganov/whisper.cpp:ggml-huge.bin: "), e.getMessage());
    }
}
