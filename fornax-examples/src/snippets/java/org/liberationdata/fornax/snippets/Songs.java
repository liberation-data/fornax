package org.liberationdata.fornax.snippets;

// --8<-- [start:acestep]
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.acestep.AceStepModel;

import java.io.File;
// --8<-- [end:acestep]
// --8<-- [start:yue2]
import org.liberationdata.fornax.yue2.Yue2Model;

import java.nio.file.Files;
import java.nio.file.Path;
// --8<-- [end:yue2]

/** The snippets of website/src/content/docs/docs/guides/songs.md: compiled, never run. */
class Songs {
    void aceStep() throws Exception {
        // --8<-- [start:acestep]
        String repo = "Serveurperso/ACE-Step-1.5-GGUF";
        String lm = HfCache.require(repo, "acestep-5Hz-lm-4B-Q8_0.gguf").toString();
        String textEncoder = HfCache.require(repo, "Qwen3-Embedding-0.6B-Q8_0.gguf").toString();
        String dit = HfCache.require(repo, "acestep-v15-turbo-Q8_0.gguf").toString();
        String vae = HfCache.require(repo, "vae-BF16.gguf").toString();

        String request = """
                {"caption": "modern country rock, male baritone vocal with a southern twang, telecaster, driving drums",
                 "lyrics": "[Verse 1]\\nRusty rolled in on a Tuesday night\\n\\n[Chorus]\\nHe's a machine with a honky-tonk heart\\n",
                 "bpm": 96, "duration": 60, "keyscale": "G major", "timesignature": "4", "vocal_language": "en",
                 "seed": 11, "lm_seed": 11, "inference_steps": 8, "guidance_scale": 1.0, "shift": 3.0}""";

        try (var songs = new AceStepModel(lm, textEncoder, dit, vae, true)) {
            AceStepModel.Result r = songs.generate(request);
            r.audio().writeWav(new File("song.wav"));
            System.out.println(r.resolvedRequestJson());      // the request after the LM filled in what it decided
        }
        // --8<-- [end:acestep]
    }

    void yue2() throws Exception {
        // --8<-- [start:yue2]
        String repo = "Serveurperso/YuE2-GGUF";
        String request = """
                {"style": "modern country rock, male baritone lead vocal, overdriven electric guitars, mid-tempo stomp",
                 "lyrics": "[Verse 1]\\nRusty rolled in on a Tuesday night\\n\\n[Chorus]\\nHe's a machine with a honky-tonk heart\\n",
                 "duration": 60, "seed": 1, "lm_seed": 1}""";

        try (var yue = new Yue2Model(HfCache.require(repo, "YuE2-3B-Q8_0.gguf").toString(),
                HfCache.require(repo, "YuE2-Vae-F32.gguf").toString(), true, 0)) {
            Yue2Model.Result r = yue.generate(request);
            r.audio().writeWav(new File("yue2-song.wav"));
            if (r.abcScore() != null) Files.writeString(Path.of("yue2-song.abc"), r.abcScore());
        }
        // --8<-- [end:yue2]
    }
}
