package org.liberationdata.fornax.snippets;

// --8<-- [start:separate]
import org.liberationdata.fornax.Audio;
import org.liberationdata.fornax.demucs.DemucsModel;
import org.liberationdata.fornax.demucs.Stems;

import java.io.File;
// --8<-- [end:separate]

/** The snippets of website/src/content/docs/docs/guides/stem-separation.md: compiled, never run. */
class StemSeparation {
    void separate() throws Exception {
        // --8<-- [start:separate]
        try (var demucs = new DemucsModel(DemucsModel.Weights.HTDEMUCS)) {
            Stems s = demucs.separate(Audio.readWav(new File("song.wav")));
            s.vocals().writeWav(new File("song-vocals.wav"));
            s.backing().writeWav(new File("song-backing.wav"));
            Audio drums = s.get("drums");
        }
        // --8<-- [end:separate]
    }

    void options() {
        // --8<-- [start:options]
        var options = DemucsModel.Options.DEFAULT
                .withShifts(2)       // average 2 randomly time-shifted passes (demucs --shifts): a little cleaner, 2x the time
                .withOverlap(0.25f)  // between segments; below 0 (the default) is the model's own, 0.25
                .withSeed(7);        // picks the shifts
        // --8<-- [end:options]
    }

    void progress(DemucsModel demucs, Audio audio) {
        // --8<-- [start:progress]
        Stems s = demucs.separate(audio, DemucsModel.Options.DEFAULT,
                (done, total) -> System.out.printf("%d/%d%n", done, total));
        // --8<-- [end:progress]
    }
}
