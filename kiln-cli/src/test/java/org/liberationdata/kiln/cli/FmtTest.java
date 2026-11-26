package org.liberationdata.kiln.cli;

import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.hub.ModelCatalog;
import org.liberationdata.kiln.hub.PullProgress;

import static org.junit.jupiter.api.Assertions.*;

class FmtTest {
    @Test
    void sizesAndTimes() {
        assertEquals("270 B", Fmt.bytes(270));
        assertEquals("6 kB", Fmt.bytes(6236));
        assertEquals("639 MB", Fmt.bytes(639446688));
        assertEquals("2.5 GB", Fmt.bytes(2_497_280_000L));
        assertEquals("12.5 GB", Fmt.bytes(12_500_000_000L));
        assertEquals("4.3s", Fmt.seconds(4.33));
        assertEquals("2:05", Fmt.seconds(125));
        assertEquals("1:00:01", Fmt.seconds(3601));
    }

    @Test
    void progressLine() {
        var set = ModelCatalog.require("flux2-klein");
        var part = set.parts().get(1);
        // the part's bytes and the rate; the time left is for the whole pull (here 1.5 GB more at 10 MB/s)
        var p = new PullProgress(set, part, 1, 3, PullProgress.Phase.DOWNLOADING, 500_000_000, 1_000_000_000,
                1_500_000_000, 3_000_000_000L, 10_000_000);
        assertEquals("  [2/3] flux2-vae.safetensors   50%  500 MB / 1.0 GB  10.0 MB/s  2:30 left", PullLine.line(p, false));
        assertEquals("  [2/3] flux2-vae.safetensors   50%  500 MB / 1.0 GB  10.0 MB/s", PullLine.line(p, true));
    }
}
