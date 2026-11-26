package org.liberationdata.fornax.studio;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class JsonTest {
    @Test void readsWhatItWrites() {
        var v = Json.obj("s", "a \"q\"\n\t\u0001é", "n", 3L, "d", 1.5, "b", true, "z", null, "l", List.of(1L, "x", Map.of()));
        assertEquals(v, Json.parse(Json.write(v)));
    }

    @Test void readsEscapesAndNumbers() {
        var m = Json.object("{\"a\": \"\\u0041\\/\", \"n\": -12, \"e\": 1e3, \"big\": 12345678901234567890}");
        assertEquals("A/", m.get("a"));
        assertEquals(-12L, m.get("n"));
        assertEquals(1000.0, m.get("e"));
        assertInstanceOf(Double.class, m.get("big"));
    }

    @Test void rejectsBadJson() {
        assertThrows(IllegalArgumentException.class, () -> Json.parse("{\"a\": }"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("[1, 2"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("{} x"));
        assertThrows(IllegalArgumentException.class, () -> Json.object("[]"));
    }

    @Test void writesRawJsonAsItIs() {
        assertEquals("{\"r\":{\"x\":1}}", Json.write(Json.obj("r", new Json.Raw("{\"x\":1}"))));
    }
}
