package org.liberationdata.fornax.jitllm;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Json;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChatJsonTest {

    @Test void parsesMessagesAndSampling() {
        ChatJson.Request r = ChatJson.parse("""
                {"model": "x", "messages": [{"role": "system", "content": "Be brief."},
                 {"role": "user", "content": [{"type": "text", "text": "Hi "}, {"type": "text", "text": "there"}]}],
                 "max_completion_tokens": 20, "temperature": 0, "top_p": 0.5, "seed": 7, "stop": ["\\n\\n"]}""");
        assertEquals(2, r.messages().size());
        assertEquals("system", r.messages().get(0).role());
        assertEquals("Hi there", r.messages().get(1).content());
        assertEquals(20, r.maxTokens());
        assertEquals(0f, r.temperature());
        assertEquals(0.5f, r.topP());
        assertEquals(7, r.seed());
        assertEquals(List.of("\n\n"), r.stop());
    }

    @Test void defaultsWhatIsLeftOut() {
        ChatJson.Request r = ChatJson.parse("{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"stop\": \"END\"}");
        assertEquals(512, r.maxTokens());
        assertEquals(-1, r.seed());
        assertEquals(0f, r.topP());
        assertEquals(List.of("END"), r.stop());
    }

    @Test void refusesWhatJitllmCannotDoAsABadRequest() {
        for (String req : List.of(
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"tools\": [{\"type\": \"function\"}]}",
                "{\"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"json_schema\": {\"type\": \"object\"}}",
                "{\"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"image_url\"}]}]}",
                "{\"messages\": []}",
                "not json")) {
            IllegalStateException e = assertThrows(IllegalStateException.class, () -> ChatJson.parse(req), req);
            assertTrue(e.getMessage().contains("(rc=7): "), e.getMessage());
        }
    }

    @Test void responseHasLlamaModelsShape() {
        Object doc = Json.parse(ChatJson.response(new ChatJson.Reply("Hello \"you\"", 5, 3, "length")));
        assertEquals("assistant", Json.get(doc, "message", "role"));
        assertEquals("Hello \"you\"", Json.get(doc, "message", "content"));
        assertNull(Json.get(doc, "message", "reasoning_content"));
        assertEquals("length", Json.get(doc, "finish_reason"));
        assertEquals(5L, Json.get(doc, "usage", "prompt_tokens"));
        assertEquals(3L, Json.get(doc, "usage", "completion_tokens"));
    }

    @Test void splitsALeadingThinkBlock() {
        Object doc = Json.parse(ChatJson.response(new ChatJson.Reply("<think>\nhmm\n</think>\n\nAnswer", 1, 1, "stop")));
        assertEquals("hmm", Json.get(doc, "message", "reasoning_content"));
        assertEquals("Answer", Json.get(doc, "message", "content"));
    }

    @Test void streamsOnlyTheContent() {
        assertEquals("The answer", streamed("<th", "ink>", "let me", " see</thi", "nk>\n\nThe", " answer"));
        assertEquals(" Hi there", streamed(" ", "Hi", " there"));
        assertEquals("", streamed("<think>never closed"));
    }

    @Test void splitsAReplyThatStartsInsideItsReasoning() {
        Object doc = Json.parse(ChatJson.response(new ChatJson.Reply("Okay, the user asks.\n</think>\n\n51", 1, 1, "stop")));
        assertEquals("Okay, the user asks.", Json.get(doc, "message", "reasoning_content"));
        assertEquals("51", Json.get(doc, "message", "content"));
        assertEquals("51", streamed("Okay,", " the user asks.\n</th", "ink>\n\n", "51"));
        assertEquals("51 ok", streamed("To solve it</think>", "", "\n\n", "51", " ok"));
    }

    @Test void letsOutAStartThatDoesNotLookLikeReasoning() {
        String start = "x".repeat(ChatJson.Content.UNSURE);
        List<String> out = new ArrayList<>();
        var c = new ChatJson.Content(out::add);
        c.accept(start);
        assertEquals(List.of(start), out);   // let out before the reply ends
        c.accept(" more");
        c.finish();
        assertEquals(start + " more", String.join("", out));
    }

    @Test void refusesToolMessagesAndUnknownRoles() {
        for (String req : List.of(
                "{\"messages\": [{\"role\": \"tool\", \"content\": \"42\"}]}",
                "{\"messages\": [{\"role\": \"assistant\", \"content\": null, \"tool_calls\": [{\"id\": \"1\"}]}]}",
                "{\"messages\": [{\"role\": \"wizard\", \"content\": \"hi\"}]}")) {
            IllegalStateException e = assertThrows(IllegalStateException.class, () -> ChatJson.parse(req), req);
            assertTrue(e.getMessage().contains("(rc=7): "), e.getMessage());
        }
        assertEquals("developer", ChatJson.parse("{\"messages\": [{\"role\": \"developer\", \"content\": \"x\"}]}").messages().get(0).role());
    }

    private static String streamed(String... pieces) {
        List<String> out = new ArrayList<>();
        var c = new ChatJson.Content(out::add);
        for (String p : pieces) c.accept(p);
        c.finish();
        return String.join("", out);
    }
}
