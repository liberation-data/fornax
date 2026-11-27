package org.liberationdata.fornax.server;

import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.util.Base64;
import java.util.List;
import java.util.zip.CRC32;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.liberationdata.fornax.Json.get;

/** An OpenAI chat request, as the request chatJson takes. */
class ChatRequestTest {
    static final String MSG = "\"messages\": [{\"role\": \"user\", \"content\": \"Hi\"}]";

    static ChatRequest of(String body) { return ChatRequest.of(Json.object(body), "default-model"); }

    static Object engine(String body) { return org.liberationdata.fornax.Json.parse(of(body).engineJson()); }

    @Test void messagesAndSamplingPassThrough() {
        Object e = engine("{" + MSG + ", \"temperature\": 0, \"top_p\": 0.9, \"top_k\": 20, \"min_p\": 0.1, \"seed\": 42, \"max_tokens\": 64}");
        assertEquals("user", get(e, "messages", 0, "role"));
        assertEquals(0L, get(e, "temperature"));
        assertEquals(0.9, get(e, "top_p"));
        assertEquals(20L, get(e, "top_k"));
        assertEquals(0.1, get(e, "min_p"));
        assertEquals(42L, get(e, "seed"));
        assertEquals(64L, get(e, "max_tokens"));
        assertNull(get(e, "json_schema"));
        assertEquals("default-model", of("{" + MSG + "}").model());
        assertEquals("qwen3-4b", of("{" + MSG + ", \"model\": \" qwen3-4b \"}").model());
    }

    @Test void maxCompletionTokensIsTheNewName() {
        assertEquals(9L, get(engine("{" + MSG + ", \"max_tokens\": 5, \"max_completion_tokens\": 9}"), "max_tokens"));
        assertNull(get(engine("{" + MSG + "}"), "max_tokens"), "until the model stops");
    }

    @Test void responseFormatIsTheEnginesJsonSchema() {
        String schema = "{\"type\": \"object\", \"properties\": {\"total\": {\"type\": \"number\"}}}";
        Object e = engine("{" + MSG + ", \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {\"name\": \"x\", \"strict\": true, \"schema\": " + schema + "}}}");
        assertEquals("number", get(e, "json_schema", "properties", "total", "type"));
        assertEquals(Map.of("type", "object"), get(engine("{" + MSG + ", \"response_format\": {\"type\": \"json_object\"}}"), "json_schema"));
        assertNull(get(engine("{" + MSG + ", \"response_format\": {\"type\": \"text\"}}"), "json_schema"));
        assertThrows(ApiError.class, () -> of("{" + MSG + ", \"response_format\": {\"type\": \"json_schema\", \"json_schema\": {}}}"));
    }

    @Test void aToolChoiceNamingOneFunctionRequiresThatOne() {
        String tools = "[{\"type\": \"function\", \"function\": {\"name\": \"a\"}}, {\"type\": \"function\", \"function\": {\"name\": \"b\"}}]";
        Object e = engine("{" + MSG + ", \"tools\": " + tools + ", \"tool_choice\": {\"type\": \"function\", \"function\": {\"name\": \"b\"}}}");
        assertEquals("required", get(e, "tool_choice"));
        assertEquals(1, ((List<?>) get(e, "tools")).size());
        assertEquals("b", get(e, "tools", 0, "function", "name"));
        assertEquals("auto", get(engine("{" + MSG + ", \"tools\": " + tools + ", \"tool_choice\": \"auto\"}"), "tool_choice"));
        assertEquals(false, get(engine("{" + MSG + ", \"tools\": " + tools + ", \"parallel_tool_calls\": false}"), "parallel_tool_calls"));
        assertThrows(ApiError.class, () -> of("{" + MSG + ", \"tools\": " + tools + ", \"tool_choice\": {\"type\": \"function\", \"function\": {\"name\": \"c\"}}}"));
        assertThrows(ApiError.class, () -> of("{" + MSG + ", \"tool_choice\": \"sometimes\"}"));
        assertNull(get(engine("{" + MSG + ", \"tools\": []}"), "tools"));
    }

    @Test void thinkingCanBeTurnedOff() {
        assertEquals(false, get(engine("{" + MSG + ", \"chat_template_kwargs\": {\"enable_thinking\": false}}"), "enable_thinking"));
        assertEquals(false, get(engine("{" + MSG + ", \"enable_thinking\": false}"), "enable_thinking"));
        assertNull(get(engine("{" + MSG + "}"), "enable_thinking"));
    }

    @Test void streamingStopsAndImages() {
        var r = of("{" + MSG + ", \"stream\": true, \"stream_options\": {\"include_usage\": true}, \"stop\": \"\\n\\n\"}");
        assertTrue(r.stream());
        assertTrue(r.includeUsage());
        assertEquals(List.of("\n\n"), r.stops());
        assertEquals(List.of("a", "b"), of("{" + MSG + ", \"stop\": [\"a\", \"b\"]}").stops());
        assertEquals(List.of(), of("{" + MSG + ", \"stop\": null}").stops());
        assertFalse(of("{" + MSG + "}").images());
        assertTrue(of("{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"?\"}, "
                + "{\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64," + png(2, 2) + "\"}}]}]}").images());
        assertThrows(ApiError.class, () -> of("{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"input_audio\"}]}]}"));
    }

    @Test void mistakes() {
        for (String bad : List.of("{}", "{\"messages\": {}}", "{" + MSG + ", \"n\": 3}", "{" + MSG + ", \"temperature\": -1}",
                "{" + MSG + ", \"top_p\": \"high\"}", "{" + MSG + ", \"seed\": 1.5}", "{" + MSG + ", \"max_tokens\": -5}",
                "{" + MSG + ", \"stop\": [1]}", "{" + MSG + ", \"model\": 4}")) {
            ApiError e = assertThrows(ApiError.class, () -> of(bad), bad);
            assertEquals(400, e.status, bad);
        }
        assertDoesNotThrow(() -> of("{" + MSG + ", \"n\": 1}"));
        ApiError none = assertThrows(ApiError.class, () -> ChatRequest.of(Json.object("{" + MSG + "}"), null));
        assertEquals("model", none.param);
    }

    /** A real PNG this size, as base64. */
    static String png(int w, int h) {
        var out = new ByteArrayOutputStream();
        try { ImageIO.write(new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB), "png", out); }
        catch (java.io.IOException e) { throw new java.io.UncheckedIOException(e); }
        return Base64.getEncoder().encodeToString(out.toByteArray());
    }

    /** A PNG's signature and header alone, saying it is this size: a few bytes that would decode to w x h pixels. */
    static String pngHeader(int w, int h) {
        ByteBuffer ihdr = ByteBuffer.allocate(17).put("IHDR".getBytes()).putInt(w).putInt(h).put(new byte[]{8, 2, 0, 0, 0});
        CRC32 crc = new CRC32();
        crc.update(ihdr.array());
        ByteBuffer b = ByteBuffer.allocate(8 + 4 + 17 + 4).put(new byte[]{(byte) 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
                .putInt(13).put(ihdr.array()).putInt((int) crc.getValue());
        return Base64.getEncoder().encodeToString(b.array());
    }

    static String withImages(String... base64) {
        var parts = new StringBuilder("{\"type\": \"text\", \"text\": \"?\"}");
        for (String b : base64) parts.append(", {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,").append(b).append("\"}}");
        return "{\"messages\": [{\"role\": \"user\", \"content\": [" + parts + "]}]}";
    }

    @Test void imagesAreCheckedBeforeTheEngineDecodesThem() {
        assertTrue(of(withImages(png(4, 3))).images());
        assertTrue(of(withImages(pngHeader(4096, 4096))).images(), "the header is enough");
        String mime = png(4, 3).replaceAll("(.{8})", "$1\\\\n");   // line breaks, as a MIME encoder writes it
        assertTrue(of(withImages(mime)).images());
        for (String bad : List.of("%%%%", Base64.getEncoder().encodeToString("not an image".getBytes()))) {
            ApiError e = assertThrows(ApiError.class, () -> of(withImages(bad)), bad);
            assertEquals(400, e.status);
            assertEquals("messages", e.param);
        }
        ApiError huge = assertThrows(ApiError.class, () -> of(withImages(pngHeader(9000, 9000))));
        assertTrue(huge.getMessage().contains("megapixels"), huge.getMessage());
        assertThrows(ApiError.class, () -> of(withImages(pngHeader(6000, 6000), pngHeader(6000, 6000))), "the pixels of all of them count");
    }

    /** As OpenAI: at most 4 stop sequences. Each is searched for in every piece of the reply, and a stream holds back the longest. */
    @Test void stopSequencesAreLimited() {
        assertEquals(List.of("a", "b", "c", "d"), of("{" + MSG + ", \"stop\": [\"a\", \"b\", \"c\", \"d\"]}").stops());
        String longest = "x".repeat(ChatRequest.MAX_STOP_LENGTH);
        assertEquals(List.of(longest), of("{" + MSG + ", \"stop\": \"" + longest + "\"}").stops());
        for (String bad : List.of("[\"a\", \"b\", \"c\", \"d\", \"e\"]", "\"" + longest + "x\"", "[\"a\", \"" + longest + "x\"]")) {
            ApiError e = assertThrows(ApiError.class, () -> of("{" + MSG + ", \"stop\": " + bad + "}"), bad);
            assertEquals(400, e.status);
            assertEquals("stop", e.param);
        }
    }
}
