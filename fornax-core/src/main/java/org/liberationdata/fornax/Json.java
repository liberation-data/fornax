package org.liberationdata.fornax;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Just enough JSON for building requests, reading the replies of {@code LlamaModel.chatJson} and the Hub's API, so Fornax needs no
 * JSON library. {@link #parse} gives {@code Map<String, Object>} (in order), {@code List<Object>}, {@code String},
 * {@code Double} or {@code Long} (an integer too big for a long is a Double), {@code Boolean} and {@code null}.
 */
public final class Json {
    private Json() {}

    /** s as a JSON string literal, quotes included. */
    public static String quote(String s) {
        StringBuilder b = new StringBuilder(s.length() + 2).append('"');
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
        return b.append('"').toString();
    }

    /** Parses a JSON document. Throws IllegalArgumentException on malformed input. */
    public static Object parse(String json) {
        Parser p = new Parser(json);
        Object v = p.value();
        p.ws();
        if (p.i != json.length()) throw p.error("trailing characters");
        return v;
    }

    /** The value at a path of object keys and list indexes, or null if any step is missing: get(doc, "message", "content"). */
    public static Object get(Object doc, Object... path) {
        Object v = doc;
        for (Object k : path) {
            if (k instanceof String key && v instanceof Map<?, ?> m) v = m.get(key);
            else if (k instanceof Integer idx && v instanceof List<?> l && idx >= 0 && idx < l.size()) v = l.get(idx);
            else return null;
        }
        return v;
    }

    private static final class Parser {
        final String s;
        int i;

        Parser(String s) { this.s = s; }

        IllegalArgumentException error(String what) {
            return new IllegalArgumentException("bad JSON at " + i + ": " + what);
        }

        void ws() { while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++; }

        char peek() {
            ws();
            if (i >= s.length()) throw error("unexpected end");
            return s.charAt(i);
        }

        void expect(char c) {
            if (peek() != c) throw error("expected '" + c + "'");
            i++;
        }

        Object value() {
            char c = peek();
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
            if (!s.startsWith(word, i)) throw error("expected " + word);
            i += word.length();
            return v;
        }

        Map<String, Object> object() {
            expect('{');
            Map<String, Object> m = new LinkedHashMap<>();
            if (peek() == '}') { i++; return m; }
            while (true) {
                if (peek() != '"') throw error("expected a key");
                String k = string();
                expect(':');
                m.put(k, value());
                char c = peek();
                i++;
                if (c == '}') return m;
                if (c != ',') throw error("expected ',' or '}'");
            }
        }

        List<Object> array() {
            expect('[');
            List<Object> l = new ArrayList<>();
            if (peek() == ']') { i++; return l; }
            while (true) {
                l.add(value());
                char c = peek();
                i++;
                if (c == ']') return l;
                if (c != ',') throw error("expected ',' or ']'");
            }
        }

        String string() {
            expect('"');
            StringBuilder b = new StringBuilder();
            while (true) {
                if (i >= s.length()) throw error("unterminated string");
                char c = s.charAt(i++);
                if (c == '"') return b.toString();
                if (c != '\\') { b.append(c); continue; }
                if (i >= s.length()) throw error("unterminated escape");
                char e = s.charAt(i++);
                switch (e) {
                    case '"', '\\', '/' -> b.append(e);
                    case 'b' -> b.append('\b');
                    case 'f' -> b.append('\f');
                    case 'n' -> b.append('\n');
                    case 'r' -> b.append('\r');
                    case 't' -> b.append('\t');
                    case 'u' -> {
                        if (i + 4 > s.length()) throw error("short \\u escape");
                        b.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                    }
                    default -> throw error("bad escape \\" + e);
                }
            }
        }

        Object number() {
            int start = i;
            while (i < s.length() && "+-0123456789.eE".indexOf(s.charAt(i)) >= 0) i++;
            String n = s.substring(start, i);
            if (n.isEmpty()) throw error("unexpected '" + s.charAt(start) + "'");
            try {
                if (n.contains(".") || n.contains("e") || n.contains("E")) return Double.parseDouble(n);
                try { return Long.parseLong(n); }
                catch (NumberFormatException tooBig) { return Double.parseDouble(n); }  // valid JSON, past a long
            } catch (NumberFormatException ex) {
                throw error("bad number " + n);
            }
        }
    }
}
