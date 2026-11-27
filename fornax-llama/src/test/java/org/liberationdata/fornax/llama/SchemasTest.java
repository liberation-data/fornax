package org.liberationdata.fornax.llama;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Json;

import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

class SchemasTest {
    /** An object schema with n string properties, the first `required` of them required. */
    static String object(int n, int required) {
        String props = IntStream.range(0, n).mapToObj(i -> "\"p" + i + "\": {\"type\": \"string\"}").collect(Collectors.joining(", "));
        String req = IntStream.range(0, required).mapToObj(i -> "\"p" + i + "\"").collect(Collectors.joining(", "));
        return "{\"type\": \"object\", \"properties\": {" + props + "}, \"required\": [" + req + "]}";
    }

    @Test void countsTheOptionalPropertiesOfTheWorstObjectAnywhere() {
        assertEquals(0, Schemas.mostOptionalProperties(Json.parse("{\"type\": \"string\"}")));
        assertEquals(3, Schemas.mostOptionalProperties(Json.parse(object(5, 2))));
        assertEquals(0, Schemas.mostOptionalProperties(Json.parse(object(5, 5))));
        // nested, in a list, in a request: the worst one counts
        String request = "{\"messages\": [], \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"f\", \"parameters\": "
                + "{\"type\": \"object\", \"properties\": {\"inner\": " + object(40, 0) + "}}}}], \"json_schema\": " + object(7, 0) + "}";
        assertEquals(40, Schemas.mostOptionalProperties(Json.parse(request)));
    }

    @Test void additionalPropertiesCountsAsOneMoreUnlessFalse() {
        String two = "{\"type\": \"object\", \"properties\": {\"a\": {}, \"b\": {}}";
        assertEquals(3, Schemas.mostOptionalProperties(Json.parse(two + ", \"additionalProperties\": true}")));
        assertEquals(3, Schemas.mostOptionalProperties(Json.parse(two + ", \"additionalProperties\": {\"type\": \"number\"}}")));
        assertEquals(2, Schemas.mostOptionalProperties(Json.parse(two + ", \"additionalProperties\": false}")));
    }

    @Test void aRequiredNameThatIsntAPropertyOrIsListedTwiceDoesntCount() {
        String s = "{\"type\": \"object\", \"properties\": {\"a\": {}, \"b\": {}}, \"required\": [\"a\", \"a\", \"zzz\"]}";
        assertEquals(1, Schemas.mostOptionalProperties(Json.parse(s)));
    }

    @Test void refusesPastTheLimitAndSaysWhy() {
        Schemas.requireBuildable(Json.parse(object(Schemas.MAX_OPTIONAL_PROPERTIES, 0)), "x");
        Schemas.requireBuildable(Json.parse(object(5000, 5000)), "x");   // required properties don't chain
        var e = assertThrows(IllegalArgumentException.class,
                () -> Schemas.requireBuildable(Json.parse(object(Schemas.MAX_OPTIONAL_PROPERTIES + 1, 0)), "response_format"));
        assertTrue(e.getMessage().startsWith("response_format has an object with 257 optional properties"), e.getMessage());
        assertTrue(e.getMessage().contains("\"required\""), e.getMessage());
    }

    @Test void aDeeplyNestedDocumentDoesntOverflowTheWalk() {
        Object deep = Json.parse(object(3, 0));
        for (int i = 0; i < 100_000; i++) deep = java.util.Map.of("items", deep);
        assertEquals(3, Schemas.mostOptionalProperties(deep));
    }
}
