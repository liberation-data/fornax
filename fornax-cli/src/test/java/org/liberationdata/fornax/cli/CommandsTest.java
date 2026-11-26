package org.liberationdata.fornax.cli;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.hub.ModelCatalog;
import org.liberationdata.fornax.whisper.WhisperModel.Alignment;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/** image, song, speak, transcribe and see up to the point where a model would load: arguments, defaults, refusals. */
class CommandsTest {
    @TempDir Path tmp;
    ByteArrayOutputStream out = new ByteArrayOutputStream(), err = new ByteArrayOutputStream();

    int run(String... argv) {
        out.reset();
        err.reset();
        return Main.run(argv, new PrintStream(out, true, StandardCharsets.UTF_8), new PrintStream(err, true, StandardCharsets.UTF_8));
    }

    static Args args(Set<String> options, String... argv) { return Args.parse(List.of(argv), options, Set.of()); }

    @Test
    void eachCommandHasHelp() {
        for (var c : List.of(List.of("image", "--width"), List.of("song", "--lyrics"), List.of("speak", "--list-voices"),
                List.of("transcribe", "--words"), List.of("see", "--mmproj"))) {
            assertEquals(0, run(c.get(0), "--help"), c.get(0));
            assertTrue(out.toString().startsWith("usage: fornax " + c.get(0)), out.toString());
            assertTrue(out.toString().contains(c.get(1)), out.toString());
            assertEquals(0, run("help", c.get(0)));
            assertTrue(out.toString().startsWith("usage: fornax " + c.get(0)), out.toString());
        }
        assertEquals(0, run("--help"));
        for (String c : List.of("image <model>", "song <model>", "speak", "transcribe <audio.wav>", "see <model> <image>"))
            assertTrue(out.toString().contains("  " + c), out.toString());
        assertEquals(0, run("image", "--help"));
        assertTrue(out.toString().contains("flux2-klein, qwen-image, qwen-image-edit"), out.toString());
        assertEquals(0, run("song", "--help"));
        assertTrue(out.toString().contains("ace-step-turbo") && out.toString().contains("yue2"), out.toString());
    }

    @Test
    void mistakesAreUsageErrors() throws Exception {
        assertEquals(2, run("image"));
        assertTrue(err.toString().contains("which model? an image model: flux2-klein"), err.toString());
        assertEquals(2, run("image", "flux2-klein"));
        assertTrue(err.toString().contains("Give a prompt"), err.toString());
        assertEquals(2, run("image", "flux2-klein", "a fox", "--width", "1000"));
        assertTrue(err.toString().contains("--width must be a multiple of 16, not 1000"), err.toString());
        assertEquals(2, run("image", "flux2-klein", "a fox", "--steps", "0"));
        assertEquals(2, run("image", "flux2-klein", "a fox", "--cfg", "lots"));
        assertEquals(2, run("image", "flux2-klein", "a fox", "--out", tmp.resolve("a.jpg").toString()));
        assertTrue(err.toString().contains("this writes a .png file"), err.toString());
        assertEquals(2, run("image", "qwen-image-edit", "make it night"));
        assertTrue(err.toString().contains("give it with --ref FILE"), err.toString());
        assertEquals(2, run("image", "flux2-klein", "a fox", "--nope"));
        assertEquals(2, run("song"));
        assertEquals(2, run("song", "ace-step-turbo"));
        assertTrue(err.toString().contains("Describe it"), err.toString());
        assertEquals(2, run("song", "ace-step-turbo", "folk", "--seconds", "0"));
        assertEquals(2, run("song", "ace-step-turbo", "folk", "--out", tmp.resolve("a.mp3").toString()));
        assertEquals(2, run("speak", "--list-voices", "hello"));
        assertEquals(2, run("speak", "hello", "--out", tmp.resolve("a.mp3").toString()));
        assertEquals(2, run("transcribe"));
        assertEquals(2, run("transcribe", "a.wav", "b.wav"));
        assertEquals(2, run("see"));
        assertEquals(2, run("see", "qwen3-vl-4b"));
        assertTrue(err.toString().contains("which image?"), err.toString());
    }

    @Test
    void theWrongKindOfModelIsRefusedWithTheRightOnes() throws Exception {
        assertEquals(1, run("image", "qwen3-4b", "a fox", "--out", tmp.resolve("a.png").toString()));
        assertEquals("fornax image: qwen3-4b is a chat model; image takes an image model: flux2-klein, qwen-image, qwen-image-edit\n",
                err.toString());
        assertEquals(1, run("song", "flux2-klein", "folk", "--out", tmp.toString()));
        assertTrue(err.toString().contains("flux2-klein is an image model; song takes a song model: ace-step-turbo"), err.toString());
        assertEquals(1, run("speak", "--model", "qwen3-4b", "hello", "--out", tmp.toString()));
        assertTrue(err.toString().contains("qwen3-4b is a chat model; speak takes a speech model: kokoro"), err.toString());
        Path wav = tmp.resolve("a.wav");
        Audio.ofMono(new float[1600], 16000).writeWav(wav.toFile());
        assertEquals(1, run("transcribe", wav.toString(), "--model", "kokoro"));
        assertTrue(err.toString().contains("kokoro is a speech model; transcribe takes a speech-to-text model: whisper-large-v3-turbo"), err.toString());
        Path png = tmp.resolve("a.png");
        ImageIO.write(new BufferedImage(16, 16, BufferedImage.TYPE_INT_RGB), "png", png.toFile());
        assertEquals(1, run("see", "qwen3-4b", png.toString()));
        assertTrue(err.toString().contains("qwen3-4b is a chat model; see takes a vision model: qwen3-vl-4b"), err.toString());
        assertEquals(1, run("image", "nope", "a fox", "--out", tmp.toString()));
        assertTrue(err.toString().contains("No such model: nope"), err.toString());
        assertEquals(1, run("song", "acme/Song-GGUF", "folk", "--out", tmp.toString()));
        assertTrue(err.toString().contains("not a reference"), err.toString());
    }

    @Test
    void missingInputFilesFailBeforeAnyModel() {
        assertEquals(1, run("transcribe", tmp.resolve("none.wav").toString()));
        assertTrue(err.toString().contains("no such file"), err.toString());
        assertEquals(1, run("see", "qwen3-vl-4b", tmp.resolve("none.png").toString()));
        assertTrue(err.toString().contains("no such image"), err.toString());
        assertEquals(1, run("image", "qwen-image-edit", "night", "--ref", tmp.resolve("none.png").toString()));
        assertEquals(1, run("song", "ace-step", "folk", "--lyrics", tmp.resolve("none.txt").toString()));
        assertTrue(err.toString().contains("no such lyrics file"), err.toString());
    }

    @Test
    void notAWavSaysHowToConvert() throws Exception {
        Path mp3 = Files.writeString(tmp.resolve("a.mp3"), "ID3 not really");
        assertEquals(1, run("transcribe", mp3.toString()));
        assertTrue(err.toString().contains("ffmpeg"), err.toString());
    }

    @Test
    void imageDefaultsAreTheModels() {
        var klein = Image.job(args(Image.OPTIONS), ModelCatalog.require("flux2-klein"), "a fox", null);
        assertEquals(List.of(1024, 1024, 20), List.of(klein.width(), klein.height(), klein.steps()));
        assertEquals(4.0, klein.cfg());
        assertTrue(klein.seed() >= 0);
        assertEquals(2.5, Image.job(args(Image.OPTIONS), ModelCatalog.require("qwen-image"), "a fox", null).cfg());
        var given = Image.job(args(Image.OPTIONS, "--width", "512", "--height=256", "--steps", "8", "--cfg", "1.5", "--seed", "42"),
                ModelCatalog.require("flux2-klein"), "a fox", null);
        assertEquals(new Image.Job("a fox", 512, 256, 8, 1.5, 42), given);
        var ref = Image.job(args(Image.OPTIONS), ModelCatalog.require("qwen-image-edit"), "night", new BufferedImage(1000, 600, BufferedImage.TYPE_INT_RGB));
        assertEquals(List.of(992, 592), List.of(ref.width(), ref.height()));   // the reference's size, to a multiple of 16
    }

    @Test
    void songRequestsAsTheEnginesTakeThem() {
        String turbo = Song.request(ModelCatalog.require("ace-step-turbo"), "folk \"live\"", null, 10, 7);
        assertTrue(turbo.contains("\"caption\": \"folk \\\"live\\\"\""), turbo);
        assertTrue(turbo.contains("\"lyrics\": \"[Instrumental]\""), turbo);   // no lyric LM to write any
        assertTrue(turbo.contains("\"duration\": 10") && turbo.contains("\"seed\": 7") && turbo.contains("\"inference_steps\": 8"), turbo);
        assertTrue(Song.request(ModelCatalog.require("ace-step"), "folk", null, 30, 1).contains("\"lyrics\": \"\""));   // the LM writes them
        assertTrue(Song.request(ModelCatalog.require("ace-step-turbo"), "folk", "[Verse]\nla", 30, 1).contains("\"lyrics\": \"[Verse]\\nla\""));
        String yue = Song.request(ModelCatalog.require("yue2"), "folk", null, 20, 3);
        assertTrue(yue.contains("\"style\": \"folk\"") && yue.contains("\"lyrics\": \"\"") && yue.contains("\"duration\": 20"), yue);
        assertFalse(yue.contains("caption"), yue);
    }

    @Test
    void aVoiceTheModelHas() {
        assertEquals("af_heart", Speak.voice(null, List.of("af_heart", "am_michael"), "kokoro"));
        assertEquals("am_michael", Speak.voice("am_michael", List.of("af_heart", "am_michael"), "kokoro"));
        assertNull(Speak.voice(null, List.of(), "soprano"));
        var e = assertThrows(Main.Failure.class, () -> Speak.voice("bob", List.of("af_heart", "am_michael"), "kokoro"));
        assertEquals("kokoro has no voice bob; its voices: af_heart am_michael", e.getMessage());
    }

    @Test
    void whisperFilesGetTheirAlignmentHeads() {
        assertEquals(Alignment.BASE_EN, Transcribe.alignment(Path.of("/cache/snapshots/abc/ggml-base.en.bin")));
        assertEquals(Alignment.LARGE_V3_TURBO, Transcribe.alignment(Path.of("ggml-large-v3-turbo-q5_0.bin")));
        assertEquals(Alignment.NONE, Transcribe.alignment(Path.of("my-whisper.bin")));
    }

    @Test
    void outputGoesWhereItIsToldAndNeverOverDefaults() throws Exception {
        assertEquals(tmp.resolve("image.png"), OutFile.choose(null, tmp, "image.png", ".png"));
        Files.writeString(tmp.resolve("image.png"), "earlier");
        assertEquals(tmp.resolve("image-2.png"), OutFile.choose(null, tmp, "image.png", ".png"));
        Files.writeString(tmp.resolve("image-2.png"), "earlier");
        assertEquals(tmp.resolve("image-3.png"), OutFile.choose(null, tmp, "image.png", ".png"));
        Path named = tmp.resolve("image.png");   // named: written over
        assertEquals(named, OutFile.choose(named.toString(), tmp, "image.png", ".png"));
        Path deep = tmp.resolve("a/b/song.WAV");   // its folders made; any case of the extension
        assertEquals(deep, OutFile.choose(deep.toString(), tmp, "song.wav", ".wav"));
        assertTrue(Files.isDirectory(deep.getParent()));
        assertEquals(tmp.resolve("image-3.png"), OutFile.choose(tmp.toString(), Path.of("elsewhere"), "image.png", ".png"));   // a folder
        var e = assertThrows(Args.UsageException.class, () -> OutFile.choose("speech.mp3", tmp, "speech.wav", ".wav"));
        assertEquals("--out speech.mp3: this writes a .wav file", e.getMessage());
    }
}
