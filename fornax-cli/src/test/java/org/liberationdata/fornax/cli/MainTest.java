package org.liberationdata.fornax.cli;

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
        assertTrue(out.toString().contains("usage: fornax pull"));
        assertEquals(0, run("chat", "--help"));
        assertTrue(out.toString().contains("--max-tokens"));
    }

    @Test
    void theCommandSetsTheHttpServersSlowClientLimitsUnlessGiven() {
        String req = "sun.net.httpserver.maxReqTime", conns = "jdk.httpserver.maxConnections", rsp = "sun.net.httpserver.maxRspTime";
        String req0 = System.getProperty(req), conns0 = System.getProperty(conns);
        try {
            System.clearProperty(req);
            System.setProperty(conns, "9");   // as -D in FORNAX_OPTS
            Server.httpLimits();
            assertEquals("120", System.getProperty(req));
            assertEquals("9", System.getProperty(conns));
            assertNull(System.getProperty(rsp), "no limit on answering: it would cut off a long stream");
        } finally {
            if (req0 == null) System.clearProperty(req); else System.setProperty(req, req0);
            if (conns0 == null) System.clearProperty(conns); else System.setProperty(conns, conns0);
        }
    }

    @Test
    void mistakesExitWithTwo() {
        assertEquals(2, run("frobnicate"));
        assertEquals(2, run("chat"));
        assertTrue(err.toString().contains("which model?"), err.toString());
        assertEquals(2, run("test", "qwen3-0.6b", "--bogus"));
        assertEquals(2, run("list", "extra"));
        assertEquals(2, run("pull", "not-a-repo"));
        assertEquals(2, run("rm"));
    }

    @Test
    void rmTakesOnlyAnAlias() {
        assertEquals(1, run("rm", "Qwen/Qwen3-0.6B-GGUF"));   // a repo is left to `hf cache rm`, never deleted here
        assertTrue(err.toString().contains("hf cache rm"), err.toString());
    }
}
