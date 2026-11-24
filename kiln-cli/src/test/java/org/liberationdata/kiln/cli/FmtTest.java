package org.liberationdata.kiln.cli;

import org.junit.jupiter.api.Test;

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
        assertEquals("  model.gguf   50%  500 MB / 1.0 GB  10.0 MB/s  50.0s left",
                Progress.line("model.gguf", 500_000_000, 1_000_000_000, 10_000_000, false));
        assertEquals("  model.gguf  100%  1.0 GB / 1.0 GB  10.0 MB/s",
                Progress.line("model.gguf", 1_000_000_000, 1_000_000_000, 10_000_000, true));
    }
}
