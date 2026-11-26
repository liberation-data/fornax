package org.liberationdata.fornax.cli;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ThinkFilterTest {
    /** Feeds the pieces through a filter and returns what it let out. */
    static String run(boolean show, List<String> pieces) { return run(show, false, pieces); }

    static String run(boolean show, boolean startsInside, List<String> pieces) {
        StringBuilder shown = new StringBuilder();
        var f = new ThinkFilter(shown::append, show, startsInside);
        pieces.forEach(f);
        f.finish();
        return shown.toString();
    }

    @Test
    void hidesThinkBlocksAndTheBlankLinesAfter() {
        assertEquals("Hello!", run(false, List.of("<think>\nThe user greets me.\n</think>\n\n", "Hello!")));
        assertEquals("Hi", run(false, List.of("<think>\n\n</think>\n\nHi")));   // Qwen3's empty block
        assertEquals("no tags at all", run(false, List.of("no tags ", "at all")));
    }

    @Test
    void handlesTagsSplitAcrossPieces() {
        assertEquals("Answer: 4", run(false, List.of("<th", "ink>2+2</thi", "nk>", "\n\nAnswer", ": 4")));
        assertEquals("a < b, and <thin air", run(false, List.of("a < b, and <thin", " air")));   // not a tag after all
    }

    @Test
    void aReplyThatIsOnlyReasoningShowsNothing() {
        var f = new ThinkFilter(s -> fail("shown: " + s), false);
        for (char c : "<think>\nstill thinking when max-tokens ran out".toCharArray()) f.accept(String.valueOf(c));
        f.finish();
        assertEquals("", f.reply());
    }

    @Test
    void showsThemWithThink() {
        String in = "<think>\nhmm\n</think>\n\nHello";
        assertEquals(in, run(true, List.of(in.substring(0, 5), in.substring(5))));
    }

    @Test
    void theReplyLeavesTheReasoningOut() {
        var f = new ThinkFilter(s -> {}, true);
        f.accept("<think>reasoning</think>\n\nThe answer.");
        f.finish();
        assertEquals("The answer.", f.reply());
    }

    // Qwen3 Thinking and DeepSeek-R1 distills: the <think> is in the prompt, so the reply has only </think>

    @Test
    void aReplyThatOpensInsideHidesWhatCameBeforeTheClose() {
        var pieces = List.of("Okay, the user wants 2+2.\n</thi", "nk>\n\n", "4");
        assertEquals("4", run(false, pieces));
        assertEquals("<think>Okay, the user wants 2+2.\n</think>\n\n4", run(true, pieces));
        assertEquals("4", run(false, List.of("2+2 is easy.</think>\n\n4")));   // short: held back until it's clear
        var f = new ThinkFilter(s -> {}, true);
        f.accept("Hmm, sums.</think>\n\nThe answer.");
        f.finish();
        assertEquals("The answer.", f.reply());
        assertTrue(f.openedInside());
    }

    @Test
    void longReasoningThatOpensInsideStaysHiddenUntilTheClose() {
        String reasoning = "Okay, the user is asking about sums. " + "Let me check the arithmetic again. ".repeat(20);
        var shown = new StringBuilder();
        var f = new ThinkFilter(shown::append, false);
        for (char c : reasoning.toCharArray()) f.accept(String.valueOf(c));
        assertEquals("", shown.toString());
        f.accept("</think>\n\n4");
        f.finish();
        assertEquals("4", shown.toString());
    }

    @Test
    void aReplyWithNoTagsStreamsOnceItIsPastTheStart() {
        var shown = new StringBuilder();
        var f = new ThinkFilter(shown::append, false);
        String text = "Paris is the capital of France. ".repeat(10);
        for (char c : text.toCharArray()) f.accept(String.valueOf(c));
        assertFalse(shown.isEmpty(), "held back the whole reply");
        f.finish();
        assertEquals(text, shown.toString());
        assertFalse(f.openedInside());
    }

    @Test
    void aCloseAfterTheStartWasLetOutStillEndsTheReasoning() {
        String reasoning = "Two and two, as numbers, add to four; ".repeat(10);
        var f = new ThinkFilter(s -> {}, false);
        assertEquals(reasoning + "\n\n4", run(false, List.of(reasoning, "</th", "ink>", "\n\n4")));
        f.accept(reasoning);
        f.accept("</think>\n\n4");
        f.finish();
        assertEquals("4", f.reply());
        assertTrue(f.openedInside());
    }

    @Test
    void aLaterReplyOfTheChatCanStartInside() {
        assertEquals("4", run(false, true, List.of("Two and two.\n</think>\n\n4")));
        assertEquals("<think>Two and two.\n</think>\n\n4", run(true, true, List.of("Two and two.\n</think>\n\n4")));
    }
}
