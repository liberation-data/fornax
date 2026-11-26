package org.liberationdata.kiln.cli;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

class MainTest {
    ByteArrayOutputStream out = new ByteArrayOutputStream(), err = new ByteArrayOutputStream();

    int run(String... argv) {
        return Main.run(argv, new PrintStream(out, true, StandardCharsets.UTF_8), new PrintStream(err, true, StandardCharsets.UTF_8));
    }

    @Test
    void usageAndHelp() {
        assertEquals(2, run());
        assertTrue(err.toString().contains("flux2-klein"), err.toString());
        assertEquals(0, run("--help"));
        assertEquals(0, run("help", "pull"));
        assertTrue(out.toString().contains("usage: kiln pull"));
        assertEquals(0, run("chat", "--help"));
        assertTrue(out.toString().contains("--max-tokens"));
    }

    @Test
    void mistakesExitWithTwo() {
        assertEquals(2, run("frobnicate"));
        assertEquals(2, run("chat"));
        assertTrue(err.toString().contains("which model?"), err.toString());
        assertEquals(2, run("test", "qwen3-0.6b", "--bogus"));
        assertEquals(2, run("list", "extra"));
        assertEquals(2, run("pull", "not-a-repo"));
    }
}
