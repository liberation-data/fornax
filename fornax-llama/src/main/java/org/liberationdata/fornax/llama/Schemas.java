package org.liberationdata.fornax.llama;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Deque;
import java.util.List;
import java.util.Map;

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
     * that isn't false. 0 when there are none. Walks doc without recursion, so a deeply nested one can't overflow.
     */
    public static int mostOptionalProperties(Object doc) {
        int most = 0;
        Deque<Object> todo = new ArrayDeque<>();
        todo.push(doc);
        while (!todo.isEmpty()) {
            Object o = todo.pop();
            if (o instanceof Map<?, ?> m) {
                if (m.get("properties") instanceof Map<?, ?> props) {
                    int optional = props.size();
                    if (m.get("required") instanceof List<?> required)
                        optional -= (int) required.stream().distinct().filter(props::containsKey).count();
                    Object extra = m.get("additionalProperties");
                    if (extra != null && !Boolean.FALSE.equals(extra)) optional++;
                    most = Math.max(most, optional);
                }
                m.values().forEach(todo::push);
            } else if (o instanceof Collection<?> c) {
                c.forEach(todo::push);
            }
        }
        return most;
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
}
