package org.liberationdata.fornax;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class JsonTest {
    @Test
    void parsesAChatJsonResponse() {
        Object r = Json.parse("""
            {"message": {"role": "assistant", "content": "{\\"total\\": 12.5}", "tool_calls": []},
             "finish_reason": "stop", "usage": {"prompt_tokens": 85, "completion_tokens": 9}, "ok": true, "none": null}""");
        assertEquals("{\"total\": 12.5}", Json.get(r, "message", "content"));
        assertEquals("stop", Json.get(r, "finish_reason"));
        assertEquals(85L, Json.get(r, "usage", "prompt_tokens"));
        assertEquals(List.of(), Json.get(r, "message", "tool_calls"));
        assertEquals(true, Json.get(r, "ok"));
        assertNull(Json.get(r, "none"));
        assertNull(Json.get(r, "missing", "deeper"));
    }

    @Test
    void parsesNumbersArraysAndEscapes() {
        var m = (Map<?, ?>) Json.parse("{\"a\": [1, -2.5, 3e2], \"s\": \"tab\\tquote\\\" \\u00e9 slash\\/\"}");
        assertEquals(List.of(1L, -2.5, 300.0), m.get("a"));
        assertEquals("tab\tquote\" é slash/", m.get("s"));
        assertEquals(-2.5, Json.get(m, "a", 1));
        assertNull(Json.get(m, "a", 7));
        assertEquals(1.2345678901234567e19, Json.parse("12345678901234567890"));  // past a long: a double, not an error
        assertThrows(IllegalArgumentException.class, () -> Json.parse("1-2"));
    }

    @Test
    void quoteRoundTrips() {
        String s = "line\nbreak \"quoted\" back\\slash \u0001 café";
        assertEquals(s, Json.parse(Json.quote(s)));
    }

    @Test
    void rejectsMalformedJson() {
        assertThrows(IllegalArgumentException.class, () -> Json.parse("{\"a\": }"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("[1, 2"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("{} extra"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("\"unterminated"));
    }
}
