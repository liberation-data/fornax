package org.liberationdata.fornax.jitllm;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.liberationdata.fornax.Json;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Runs a real model: ./gradlew :fornax-jitllm:test -PjitllmModel=path/to/model.gguf, with TORNADOVM_HOME set. */
@EnabledIfSystemProperty(named = "jitllm.test.model", matches = ".+")
class JitLlmModelGpuTest {
    private static final String MODEL = System.getProperty("jitllm.test.model");

    @Test void chatsStreamsAndServesChatJson() {
        try (var llm = new JitLlmModel(MODEL, 2048)) {
            StringBuilder streamed = new StringBuilder();
            Result r = llm.chat(List.of(Message.system("Answer in one word."), Message.user("What is the capital of France?")),
                    Options.DEFAULT.withTemperature(0).withMaxTokens(16), streamed::append);
            System.out.println("chat: " + r);
            assertTrue(r.text().contains("Paris"), r.text());
            assertTrue(r.promptTokens() > 0 && r.generatedTokens() > 0);
            assertEquals(r.text(), streamed.toString());

            // a second call starts from an empty context: its prompt is only its own, shorter message
            Result again = llm.chat(List.of(Message.user("Say OK.")), Options.DEFAULT.withTemperature(0).withMaxTokens(8));
            assertTrue(again.promptTokens() < r.promptTokens(), "context carried over: " + again + " after " + r);

            String json = llm.chatJson("""
                    {"messages": [{"role": "user", "content": "Count from 1 to 50, separated by spaces."}],
                     "max_tokens": 8, "temperature": 0}""", null);
            System.out.println("chatJson: " + json);
            Object doc = Json.parse(json);
            assertEquals("length", Json.get(doc, "finish_reason"));
            assertEquals(8L, Json.get(doc, "usage", "completion_tokens"));
            assertFalse(((String) Json.get(doc, "message", "content")).isBlank());
        }
    }

    @Test void aPromptLongerThanTheContextIsABadRequest() {
        try (var llm = new JitLlmModel(MODEL, 64)) {
            String request = "{\"messages\": [{\"role\": \"user\", \"content\": " + Json.quote("kiln ".repeat(200)) + "}]}";
            IllegalStateException e = assertThrows(IllegalStateException.class, () -> llm.chatJson(request, null));
            System.out.println("too long: " + e.getMessage());
            assertTrue(e.getMessage().contains("(rc=2): "), e.getMessage());
        }
    }

    @Test void cancelStopsAGeneration() throws Exception {
        try (var llm = new JitLlmModel(MODEL, 2048)) {
            var f = llm.chatAsync(List.of(Message.user("Write a very long story about a kiln.")),
                    Options.DEFAULT.withTemperature(0).withMaxTokens(2000));
            Thread.sleep(3000);
            llm.cancel();
            Result r = f.get();
            System.out.println("cancelled after " + r.generatedTokens() + " tokens");
            assertTrue(r.generatedTokens() < 2000, r.toString());
        }
    }
}
