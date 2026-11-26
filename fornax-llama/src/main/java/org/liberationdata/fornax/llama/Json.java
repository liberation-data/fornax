package org.liberationdata.fornax.llama;

/**
 * {@link org.liberationdata.fornax.Json}, where this class now lives, under the name {@link LlamaModel#chatJson}'s
 * guides use. New code can use either.
 */
public final class Json {
    private Json() {}

    /** s as a JSON string literal, quotes included. */
    public static String quote(String s) { return org.liberationdata.fornax.Json.quote(s); }

    /** Parses a JSON document. Throws IllegalArgumentException on malformed input. See {@link org.liberationdata.fornax.Json}. */
    public static Object parse(String json) { return org.liberationdata.fornax.Json.parse(json); }

    /** The value at a path of object keys and list indexes, or null if any step is missing: get(doc, "message", "content"). */
    public static Object get(Object doc, Object... path) { return org.liberationdata.fornax.Json.get(doc, path); }
}
