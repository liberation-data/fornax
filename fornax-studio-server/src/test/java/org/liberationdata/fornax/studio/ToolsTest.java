package org.liberationdata.fornax.studio;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ToolsTest {
    @Test void calculates() {
        assertEquals("32", Tools.call("calculate", Map.of("expression", "(17 + 4) * 3 / 2 + 0.5")));
        assertEquals("512", Tools.call("calculate", Map.of("expression", "2^3^2")));
        assertEquals("-1", Tools.call("calculate", Map.of("expression", "-(3 - 2)")));
        assertEquals("0.1", Tools.call("calculate", Map.of("expression", "1/10")));
    }

    @Test void answersFailuresAsText() {
        assertTrue(Tools.call("calculate", Map.of("expression", "1/0")).startsWith("error"));
        assertTrue(Tools.call("calculate", Map.of("expression", "2 +")).startsWith("error"));
        assertTrue(Tools.call("calculate", Map.of("expression", "System.exit(1)")).startsWith("error"));
        assertTrue(Tools.call("get_current_time", Map.of("timezone", "Mars/Olympus")).startsWith("error"));
        assertTrue(Tools.call("rm", Map.of()).startsWith("error"));
    }

    @Test void tellsTheTime() {
        assertTrue(Tools.call("get_current_time", Map.of("timezone", "UTC")).endsWith("UTC"));
    }
}
