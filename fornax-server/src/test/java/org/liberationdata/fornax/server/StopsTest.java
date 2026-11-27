package org.liberationdata.fornax.server;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class StopsTest {
    final List<String> out = new ArrayList<>();

    @Test void withoutStopsEveryPieceGoesStraightThrough() {
        var s = new Stops(List.of(), out::add);
        assertFalse(s.accept("a"));
        assertFalse(s.accept("bc"));
        assertEquals(List.of("a", "bc"), out);
        s.finish();
        assertEquals("abc", s.text());
    }

    @Test void aStopSplitAcrossPiecesIsCaughtAndNothingOfItIsSent() {
        var s = new Stops(List.of("END"), out::add);
        assertFalse(s.accept("hello E"));
        assertFalse(s.accept("N"));
        assertTrue(s.accept("D and more"));
        assertTrue(s.stopped());
        assertEquals("hello ", s.text());
        assertTrue(s.accept("later"), "nothing after a stop");
        assertEquals("hello ", String.join("", out));
    }

    @Test void heldTextGoesOutAtTheEnd() {
        var s = new Stops(List.of("END"), out::add);
        s.accept("almost E");
        s.accept("N");
        assertEquals("almost ", String.join("", out), "what could start a stop is held");
        s.finish();
        assertEquals("almost EN", s.text());
        assertFalse(s.stopped());
    }

    @Test void theEarliestOfSeveralStopsWins() {
        var s = new Stops(List.of("bb", "a"), out::add);
        assertTrue(s.accept("xxbbxa"));
        assertEquals("xx", s.text());
    }

    @Test void aSurrogatePairIsNeverSplit() {
        var s = new Stops(List.of("zz"), out::add);
        s.accept("a😀");   // an emoji, as two chars
        for (String piece : out) assertFalse(Character.isHighSurrogate(piece.charAt(piece.length() - 1)), piece);
        s.finish();
        assertEquals("a😀", s.text());
    }
}
