package org.liberationdata.kiln.cli;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ThinkFilterTest {
    /** Feeds the pieces through a filter and returns what it let out. */
    static String run(boolean show, List<String> pieces) {
        StringBuilder shown = new StringBuilder();
        var f = new ThinkFilter(shown::append, show);
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
}
