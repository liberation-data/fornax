package org.liberationdata.kiln.studio;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Just enough JSON for the studio's requests and replies, so the server needs nothing beyond the JDK.
 * Objects read as {@code Map<String, Object>} (in order), arrays as {@code List<Object>}, numbers as {@code Long}
 * when whole and {@code Double} otherwise.
 */
final class Json {
    private Json() {}

    static Object parse(String s) {
        var p = new Parser(s);
        p.ws();
        Object v = p.value();
        p.ws();
        if (p.i != s.length()) throw p.fail("trailing characters");
        return v;
    }

    @SuppressWarnings("unchecked")
    static Map<String, Object> object(String s) {
        Object v = parse(s);
        if (!(v instanceof Map)) throw new IllegalArgumentException("expected a JSON object");
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

    // ---- reading values out of a parsed request ----

    static String str(Map<String, Object> m, String k, String dflt) {
        Object v = m.get(k);
        return v == null ? dflt : v.toString();
    }

    static long num(Map<String, Object> m, String k, long dflt) {
        return m.get(k) instanceof Number n ? n.longValue() : dflt;
    }

    static double dbl(Map<String, Object> m, String k, double dflt) {
        return m.get(k) instanceof Number n ? n.doubleValue() : dflt;
    }

    static boolean bool(Map<String, Object> m, String k, boolean dflt) {
        return m.get(k) instanceof Boolean b ? b : dflt;
    }

    @SuppressWarnings("unchecked")
    static List<Object> list(Map<String, Object> m, String k) {
        return m.get(k) instanceof List<?> l ? (List<Object>) l : List.of();
    }

    // ---- writing ----

    private static void write(StringBuilder b, Object v) {
        switch (v) {
            case null -> b.append("null");
            case String s -> quote(b, s);
            case Boolean x -> b.append(x);
            case Double d when d.isNaN() || d.isInfinite() -> b.append("null");
            case Float f when f.isNaN() || f.isInfinite() -> b.append("null");
            case Number n -> b.append(n);
            case Map<?, ?> m -> {
                b.append('{');
                boolean first = true;
                for (var e : m.entrySet()) {
                    if (!first) b.append(',');
                    first = false;
                    quote(b, String.valueOf(e.getKey()));
                    b.append(':');
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
            case Raw r -> b.append(r.json());
            default -> quote(b, v.toString());
        }
    }

    /** JSON already written (for example by an engine), put into a reply as it is. */
    record Raw(String json) {}

    private static void quote(StringBuilder b, String s) {
        b.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> b.append("\\\"");
                case '\\' -> b.append("\\\\");
                case '\n' -> b.append("\\n");
                case '\r' -> b.append("\\r");
                case '\t' -> b.append("\\t");
                default -> {
                    if (c < 0x20) b.append(String.format("\\u%04x", (int) c));
                    else b.append(c);
                }
            }
        }
        b.append('"');
    }

    private static final class Parser {
        final String s;
        int i;

        Parser(String s) { this.s = s; }

        IllegalArgumentException fail(String why) { return new IllegalArgumentException("bad JSON at " + i + ": " + why); }

        void ws() { while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++; }

        Object value() {
            if (i >= s.length()) throw fail("unexpected end");
            char c = s.charAt(i);
            return switch (c) {
                case '{' -> object();
                case '[' -> array();
                case '"' -> string();
                case 't' -> literal("true", Boolean.TRUE);
                case 'f' -> literal("false", Boolean.FALSE);
                case 'n' -> literal("null", null);
                default -> number();
            };
        }

        Object literal(String word, Object v) {
            if (!s.startsWith(word, i)) throw fail("expected " + word);
            i += word.length();
            return v;
        }

        Map<String, Object> object() {
            Map<String, Object> m = new LinkedHashMap<>();
            i++;
            ws();
            if (peek('}')) { i++; return m; }
            while (true) {
                ws();
                if (!peek('"')) throw fail("expected a key");
                String k = string();
                ws();
                expect(':');
                ws();
                m.put(k, value());
                ws();
                if (peek(',')) { i++; continue; }
                expect('}');
                return m;
            }
        }

        List<Object> array() {
            List<Object> l = new ArrayList<>();
            i++;
            ws();
            if (peek(']')) { i++; return l; }
            while (true) {
                ws();
                l.add(value());
                ws();
                if (peek(',')) { i++; continue; }
                expect(']');
                return l;
            }
        }

        String string() {
            expect('"');
            var b = new StringBuilder();
            while (true) {
                if (i >= s.length()) throw fail("unterminated string");
                char c = s.charAt(i++);
                if (c == '"') return b.toString();
                if (c != '\\') { b.append(c); continue; }
                if (i >= s.length()) throw fail("unterminated escape");
                char e = s.charAt(i++);
                switch (e) {
                    case 'n' -> b.append('\n');
                    case 't' -> b.append('\t');
                    case 'r' -> b.append('\r');
                    case 'b' -> b.append('\b');
                    case 'f' -> b.append('\f');
                    case 'u' -> {
                        if (i + 4 > s.length()) throw fail("bad unicode escape");
                        b.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                    }
                    default -> b.append(e);
                }
            }
        }

        Object number() {
            int start = i;
            if (peek('-')) i++;
            while (i < s.length() && "0123456789.eE+-".indexOf(s.charAt(i)) >= 0) i++;
            String n = s.substring(start, i);
            if (n.isEmpty()) throw fail("unexpected character '" + s.charAt(i) + "'");
            if (n.contains(".") || n.contains("e") || n.contains("E")) return Double.parseDouble(n);
            try { return Long.parseLong(n); } catch (NumberFormatException e) { return Double.parseDouble(n); }
        }

        boolean peek(char c) { return i < s.length() && s.charAt(i) == c; }

        void expect(char c) {
            if (!peek(c)) throw fail("expected '" + c + "'");
            i++;
        }
    }
}
