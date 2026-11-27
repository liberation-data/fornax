package org.liberationdata.fornax.llama;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * What a JSON schema costs as a grammar. llama.cpp turns a schema (a {@code json_schema}, a tool's parameters) into a
 * grammar that chains one rule per optional property of an object, and refuses a grammar whose rules chain more than
 * {@value #MAX_OPTIONAL_PROPERTIES} deep. Its converter also recurses once per optional property, in time that grows
 * with their cube: a few thousand take minutes, and enough of them overflow the native stack, which crashes the
 * process. So an object with more optional properties than llama.cpp could take is refused before it gets there.
 */
public final class Schemas {
    /** The most optional properties one object can have: llama.cpp's grammar recursion limit. */
    public static final int MAX_OPTIONAL_PROPERTIES = 256;

    private Schemas() {}

    /**
     * The most optional properties of any object schema in doc (a schema, or a request that holds schemas), as
     * llama.cpp counts them: the properties not in its {@code required}, and one more for {@code additionalProperties}
     * that isn't false. An {@code allOf} is one object too: llama.cpp merges its parts' properties, the ones of its
     * {@code anyOf} alternatives optional, the rest required. 0 when there are none. Walks doc without recursion, so a
     * deeply nested one can't overflow.
     */
    public static int mostOptionalProperties(Object doc) {
        int most = 0;
        Deque<Node> todo = new ArrayDeque<>();
        todo.push(new Node(doc, null));
        while (!todo.isEmpty()) {
            Node n = todo.pop();
            if (n.value instanceof Map<?, ?> m) {
                if (m.get("properties") instanceof Map<?, ?> props) {
                    int optional = props.size();
                    if (m.get("required") instanceof List<?> required)
                        optional -= (int) required.stream().distinct().filter(props::containsKey).count();
                    Object extra = m.get("additionalProperties");
                    if (extra != null && !Boolean.FALSE.equals(extra)) optional++;
                    most = Math.max(most, optional);
                }
                if (m.get("allOf") instanceof List<?> parts) most = Math.max(most, mergedOptional(parts, n));
                m.values().forEach(v -> todo.push(new Node(v, n)));
            } else if (n.value instanceof Collection<?> c) {
                c.forEach(v -> todo.push(new Node(v, n)));
            }
        }
        return most;
    }

    /** A value in the walk, and the one that holds it: a $ref resolves against the nearest that has its path. */
    private record Node(Object value, Node parent) {}

    /**
     * The optional properties of the object llama.cpp merges an allOf into: each property of an alternative of a part's
     * anyOf (or oneOf), duplicates and all, unless a part's own properties name it, which are required.
     */
    private static int mergedOptional(List<?> parts, Node at) {
        Set<Object> required = new HashSet<>();
        List<Object> optional = new ArrayList<>();
        for (Object part : parts) {
            Object alts = part instanceof Map<?, ?> m ? (m.containsKey("anyOf") ? m.get("anyOf") : m.get("oneOf")) : null;
            if (alts instanceof List<?> l) {
                for (Object alt : l) optional.addAll(propertyNames(alt, at));
            } else {
                required.addAll(propertyNames(part, at));
            }
        }
        return (int) optional.stream().filter(name -> !required.contains(name)).count();
    }

    /** The property names of schema, following its $refs (as far as llama.cpp's 128). */
    private static Collection<?> propertyNames(Object schema, Node at) {
        for (int hops = 0; hops < 128 && schema instanceof Map<?, ?> m && m.get("$ref") instanceof String ref; hops++)
            schema = resolve(ref, at);
        return schema instanceof Map<?, ?> m && m.get("properties") instanceof Map<?, ?> props ? props.keySet() : List.of();
    }

    /** A "#/a/b" $ref's target in the nearest value around at that has that path, or null. */
    private static Object resolve(String ref, Node at) {
        if (!ref.startsWith("#/")) return null;
        String[] path = ref.substring(2).split("/", -1);
        for (Node n = at; n != null; n = n.parent) {
            Object v = n.value;
            for (String step : path) {
                if (v instanceof Map<?, ?> m) v = m.get(step);
                else if (v instanceof List<?> l && step.matches("\\d{1,9}") && Integer.parseInt(step) < l.size()) v = l.get(Integer.parseInt(step));
                else v = null;
                if (v == null) break;
            }
            if (v != null) return v;
        }
        return null;
    }

    /**
     * Throws an IllegalArgumentException, saying why, when an object schema in doc has more optional properties than
     * llama.cpp's grammar can take. what names doc in the message ("response_format", say).
     */
    public static void requireBuildable(Object doc, String what) {
        int n = mostOptionalProperties(doc);
        if (n > MAX_OPTIONAL_PROPERTIES)
            throw new IllegalArgumentException(what + " has an object with " + n + " optional properties; a grammar takes at most "
                    + MAX_OPTIONAL_PROPERTIES + " (llama.cpp chains a rule for each): list more of them in \"required\", or split the object");
    }

    /**
     * How deep a request's JSON can nest. llama.cpp refuses a schema nested more than 128 levels itself, and cleanly;
     * this is well above that, so it refuses no schema llama.cpp would take, and low enough that parsing the request,
     * here or in the engine, can't overflow a stack.
     */
    public static final int MAX_DEPTH = 256;

    /**
     * requireBuildable on a request's JSON text, parsing it only if it could hold an object schema: if it names
     * "properties", or escapes a character, as {@code "propert\\u0069es"} could. First, without parsing, it refuses
     * JSON nested more than {@link #MAX_DEPTH} levels.
     */
    static void requireBuildableJson(String json, String what) {
        if (depth(json) > MAX_DEPTH)
            throw new IllegalArgumentException(what + " is nested more than " + MAX_DEPTH + " levels deep");
        if (json.contains("\"properties\"") || json.contains("\\u")) requireBuildable(Json.parse(json), what);
    }

    /** How deep json's arrays and objects nest, brackets in strings aside. */
    static int depth(String json) {
        int depth = 0, most = 0;
        boolean inString = false;
        for (int i = 0; i < json.length(); i++) {
            char c = json.charAt(i);
            if (inString) {
                if (c == '\\') i++;
                else if (c == '"') inString = false;
            } else if (c == '"') inString = true;
            else if (c == '[' || c == '{') most = Math.max(most, ++depth);
            else if (c == ']' || c == '}') depth--;
        }
        return most;
    }
}
