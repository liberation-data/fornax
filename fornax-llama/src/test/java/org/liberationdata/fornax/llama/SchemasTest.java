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

    /** llama.cpp merges an allOf into one object: its anyOf alternatives' properties optional, its parts' required. */
    @Test void anAllOfIsOneObjectOfItsPartsProperties() {
        String two = "{\"allOf\": [{\"anyOf\": [" + object(200, 0) + ", " + object(200, 0) + "]}]}";
        assertEquals(400, Schemas.mostOptionalProperties(Json.parse(two)));
        // through a $ref, against the schema that holds it, though the request around it doesn't have the path
        String refs = "{\"messages\": [], \"json_schema\": {\"$defs\": {\"big\": " + object(200, 0) + "}, \"allOf\": [{\"anyOf\": "
                + "[{\"$ref\": \"#/$defs/big\"}, {\"$ref\": \"#/$defs/big\"}, {\"$ref\": \"#/$defs/big\"}]}]}}";
        assertEquals(600, Schemas.mostOptionalProperties(Json.parse(refs)));
        // a part's own properties are required, so they don't chain, and make an alternative's of that name required
        String part = "{\"allOf\": [" + object(200, 0) + ", {\"anyOf\": [" + object(200, 0) + ", " + object(200, 0) + "]}]}";
        assertEquals(200, Schemas.mostOptionalProperties(Json.parse(part)));   // each object on its own still counts
    }

    /** The text check skips parsing a request with no schema, but a key can be written with an escape. */
    @Test void anEscapedPropertiesKeyIsStillChecked() {
        String escaped = object(300, 0).replace("\"properties\"", "\"propert\\u0069es\"");
        assertEquals("properties", ((java.util.Map<?, ?>) Json.parse(escaped)).keySet().toArray()[1]);
        assertThrows(IllegalArgumentException.class, () -> Schemas.requireBuildableJson(escaped, "x"));
        Schemas.requireBuildableJson("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}]}", "x");
    }

    @Test void aDeeplyNestedDocumentDoesntOverflowTheWalk() {
        Object deep = Json.parse(object(3, 0));
        for (int i = 0; i < 100_000; i++) deep = java.util.Map.of("items", deep);
        assertEquals(3, Schemas.mostOptionalProperties(deep));
    }

    /** A schema whose object properties nest levels deep: {"type": "object", "properties": {"a": {...}}}. */
    static String nested(int levels) {
        String s = "{\"type\": \"string\"}";
        for (int i = 0; i < levels; i++) s = "{\"type\": \"object\", \"properties\": {\"a\": " + s + "}}";
        return s;
    }

    @Test void depthCountsBracketsButNotThoseInStrings() {
        assertEquals(0, Schemas.depth("\"x\""));
        assertEquals(2, Schemas.depth("{\"a\": [1, 2]}"));
        assertEquals(1, Schemas.depth("{\"a\": \"[[{{\\\"]]\"}"));   // brackets, and an escaped quote, in a string
    }

    @Test void aRequestNestedTooDeeplyIsRefusedWithoutParsingIt() {
        Schemas.requireBuildableJson("{\"json_schema\": " + nested(100) + "}", "x");   // 201 levels: llama.cpp's to judge
        String deep = "[".repeat(100_000) + "]".repeat(100_000);   // would overflow a recursive parser
        var e = assertThrows(IllegalArgumentException.class, () -> Schemas.requireBuildableJson(deep, "the request"));
        assertEquals("the request is nested more than " + Schemas.MAX_DEPTH + " levels deep", e.getMessage());
    }
}
