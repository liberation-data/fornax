package org.liberationdata.fornax.cli;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class ArgsTest {
    static Args parse(String... argv) { return Args.parse(List.of(argv), Set.of("max-tokens", "system"), Set.of("yes")); }

    @Test
    void positionalsOptionsAndFlags() {
        Args a = parse("qwen3-4b", "--max-tokens", "12", "hello", "--system=be brief", "--yes", "world");
        assertEquals(List.of("qwen3-4b", "hello", "world"), a.positional());
        assertEquals(12, a.intOption("max-tokens", 0));
        assertEquals("be brief", a.option("system", null));
        assertTrue(a.has("yes"));
        assertEquals(List.of("hello", "world"), a.rest(1));
        assertNull(a.positional(5));
    }

    @Test
    void doubleDashEndsOptionsAndHelpIsAlwaysAccepted() {
        Args a = parse("m", "--", "--max-tokens", "-h");
        assertEquals(List.of("m", "--max-tokens", "-h"), a.positional());
        assertTrue(parse("-h").has("help"));
        assertTrue(parse("x", "--help").has("help"));
    }

    @Test
    void singleDashAndNegativeLookingValuesArePositional() {
        assertEquals(List.of("-", "-3"), parse("-", "-3").positional());
        assertEquals(-1, parse("--max-tokens", "-1").intOption("max-tokens", 0));
    }

    @Test
    void mistakesAreUsageErrors() {
        assertThrows(Args.UsageException.class, () -> parse("--nope"));
        assertThrows(Args.UsageException.class, () -> parse("--max-tokens"));
        assertThrows(Args.UsageException.class, () -> parse("--yes=1"));
        assertThrows(Args.UsageException.class, () -> parse("--max-tokens", "lots").intOption("max-tokens", 0));
        assertThrows(Args.UsageException.class, () -> Args.parse(List.of("--temperature", "hot"), Set.of("temperature"), Set.of())
                .floatOption("temperature", 0));
    }

    @Test
    void shortNamesAndRepeatedOptions() {
        Args a = Args.parse(List.of("-m", "qwen3-4b", "--model", "flux2-klein", "--model=kokoro", "-x"), Set.of("model"), Set.of(),
                java.util.Map.of("m", "model"));
        assertEquals(List.of("qwen3-4b", "flux2-klein", "kokoro"), a.all("model"));
        assertEquals("kokoro", a.option("model", null), "the last one");
        assertEquals(List.of("-x"), a.positional(), "a one-letter name a command doesn't give is positional");
        assertEquals(List.of(), a.all("other"));
        assertEquals(List.of("-m", "x"), parse("-m", "x").positional(), "and without short names, -m is positional");
    }
}
