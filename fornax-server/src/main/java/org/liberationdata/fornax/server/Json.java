package org.liberationdata.fornax.server;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * JSON for the server's requests and replies, so it needs nothing beyond the JDK: {@link org.liberationdata.fornax.Json}
 * reads, and this writes. Objects read as {@code Map<String, Object>} (in order), arrays as {@code List<Object>},
 * numbers as {@code Long} when whole and {@code Double} otherwise.
 */
final class Json {
    private Json() {}

    /** A request body as a JSON object, or a 400 saying why it isn't one. */
    @SuppressWarnings("unchecked")
    static Map<String, Object> object(String s) {
        Object v;
        try { v = org.liberationdata.fornax.Json.parse(s); }
        catch (IllegalArgumentException e) { throw ApiError.badRequest("the body isn't JSON: " + e.getMessage(), null); }
        if (!(v instanceof Map)) throw ApiError.badRequest("the body must be a JSON object", null);
        return (Map<String, Object>) v;
    }

    static String write(Object v) {
        var b = new StringBuilder();
        write(b, v);
        return b.toString();
    }

    /** An ordered map from alternating keys and values, for writing replies: {@code Json.obj("a", 1, "b", "x")}. */
    static Map<String, Object> obj(Object... kv) {
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i + 1 < kv.length; i += 2) m.put((String) kv[i], kv[i + 1]);
        return m;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> map(Object v) { return v instanceof Map<?, ?> m ? (Map<String, Object>) m : null; }

    @SuppressWarnings("unchecked")
    static List<Object> list(Object v) { return v instanceof List<?> l ? (List<Object>) l : null; }

    private static void write(StringBuilder b, Object v) {
        switch (v) {
            case null -> b.append("null");
            case String s -> b.append(org.liberationdata.fornax.Json.quote(s));
            case Boolean x -> b.append(x);
            case Double d when d.isNaN() || d.isInfinite() -> b.append("null");
            case Float f when f.isNaN() || f.isInfinite() -> b.append("null");
            case Number n -> b.append(n);
            case float[] fs -> {
                b.append('[');
                for (int i = 0; i < fs.length; i++) {
                    if (i > 0) b.append(',');
                    if (Float.isFinite(fs[i])) b.append(fs[i]); else b.append("null");
                }
                b.append(']');
            }
            case Map<?, ?> m -> {
                b.append('{');
                boolean first = true;
                for (var e : m.entrySet()) {
                    if (!first) b.append(',');
                    first = false;
                    b.append(org.liberationdata.fornax.Json.quote(String.valueOf(e.getKey()))).append(':');
                    write(b, e.getValue());
                }
                b.append('}');
            }
            case Iterable<?> it -> {
                b.append('[');
                boolean first = true;
                for (Object o : it) {
                    if (!first) b.append(',');
                    first = false;
                    write(b, o);
                }
                b.append(']');
            }
            default -> b.append(org.liberationdata.fornax.Json.quote(v.toString()));
        }
    }
}
