package org.liberationdata.kiln.cli;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class JsonTest {
    @Test
    void parsesNestedValues() {
        Map<String, Object> m = Json.object("""
                {"sha": "abc", "gated": false, "n": 12, "x": -1.5e2, "none": null,
                 "siblings": [{"rfilename": "a\\/b \\"q\\" \\u00e9.gguf", "lfs": {"size": 639446688}}], "empty": {}, "e": []}""");
        assertEquals("abc", m.get("sha"));
        assertEquals(Boolean.FALSE, m.get("gated"));
        assertEquals(12L, m.get("n"));
        assertEquals(-150.0, m.get("x"));
        assertTrue(m.containsKey("none") && m.get("none") == null);
        var sib = (Map<?, ?>) ((List<?>) m.get("siblings")).get(0);
        assertEquals("a/b \"q\" é.gguf", sib.get("rfilename"));
        assertEquals(639446688L, ((Map<?, ?>) sib.get("lfs")).get("size"));
        assertEquals(Map.of(), m.get("empty"));
        assertEquals(List.of(), m.get("e"));
    }

    @Test
    void quoteRoundTrips() {
        String s = "line 1\nline \"2\"\t\\ \u0001 é";
        assertEquals(s, Json.parse(Json.quote(s)));
    }

    @Test
    void rejectsBrokenJson() {
        assertThrows(IllegalArgumentException.class, () -> Json.parse("{\"a\": 1"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("[1,]x"));
        assertThrows(IllegalArgumentException.class, () -> Json.object("[1]"));
    }
}
