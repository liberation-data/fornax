package org.liberationdata.fornax.snippets;

// --8<-- [start:acestep]
import org.liberationdata.fornax.acestep.AceStepModel;
// --8<-- [end:acestep]
// --8<-- [start:yue2]
import org.liberationdata.fornax.yue2.Yue2Model;
// --8<-- [end:yue2]

// --8<-- [start:acestep]
// --8<-- [start:yue2]
import java.io.File;
// --8<-- [end:acestep]
import java.nio.file.Files;
import java.nio.file.Path;
// --8<-- [end:yue2]

/** The snippets of website/src/content/docs/docs/guides/songs.md: compiled, never run. */
class Songs {
    void aceStep() throws Exception {
        // --8<-- [start:acestep]
        String request = """
                {"caption": "modern country rock, male baritone vocal with a southern twang, telecaster, driving drums",
                 "lyrics": "[Verse 1]\\nRusty rolled in on a Tuesday night\\n\\n[Chorus]\\nHe's a machine with a honky-tonk heart\\n",
                 "bpm": 96, "duration": 60, "keyscale": "G major", "timesignature": "4", "vocal_language": "en",
                 "seed": 11, "lm_seed": 11, "inference_steps": 8, "guidance_scale": 1.0, "shift": 3.0}""";

        try (var songs = new AceStepModel("ace-step")) {     // the four files, downloaded on first use
            AceStepModel.Result r = songs.generate(request);
            r.audio().writeWav(new File("song.wav"));
            System.out.println(r.resolvedRequestJson());      // the request after the LM filled in what it decided
        }
        // --8<-- [end:acestep]
    }

    @SuppressWarnings("try")   // the block shows how to open the model; its body is "..."
    void aceStepFiles() {
        // --8<-- [start:acestep-files]
        String repo = "Serveurperso/ACE-Step-1.5-GGUF";
        try (var songs = new AceStepModel(repo + ":acestep-5Hz-lm-1.7B-Q8_0.gguf", repo + ":Qwen3-Embedding-0.6B-Q8_0.gguf",
                repo + ":acestep-v15-turbo-Q8_0.gguf", repo + ":vae-BF16.gguf", true)) {   // lm, text encoder, DiT, VAE
            // ...
        }
        // --8<-- [end:acestep-files]
    }

    void yue2() throws Exception {
        // --8<-- [start:yue2]
        String request = """
                {"style": "modern country rock, male baritone lead vocal, overdriven electric guitars, mid-tempo stomp",
                 "lyrics": "[Verse 1]\\nRusty rolled in on a Tuesday night\\n\\n[Chorus]\\nHe's a machine with a honky-tonk heart\\n",
                 "duration": 60, "seed": 1, "lm_seed": 1}""";

        try (var yue = new Yue2Model("yue2", true, 0)) {     // the model and its VAE, downloaded on first use
            Yue2Model.Result r = yue.generate(request);
            r.audio().writeWav(new File("yue2-song.wav"));
            if (r.abcScore() != null) Files.writeString(Path.of("yue2-song.abc"), r.abcScore());
        }
        // --8<-- [end:yue2]
    }
}
