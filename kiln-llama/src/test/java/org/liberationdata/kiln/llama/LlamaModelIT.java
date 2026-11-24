package org.liberationdata.kiln.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.HfCache;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** Runs against a real model when it's in the Hugging Face cache and the native library is built; skipped otherwise. */
class LlamaModelIT {
    static final String REPO = "unsloth/Qwen3-4B-Instruct-2507-GGUF", FILE = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf";
    static LlamaModel model;

    @BeforeAll
    static void load() {
        var path = HfCache.find(REPO, FILE);
        assumeTrue(path.isPresent(), "model not downloaded: hf download " + REPO + " " + FILE);
        model = new LlamaModel(path.get().toString(), -1, 4096);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    @Test
    void chatStreamsAndCountsTokens() {
        var pieces = new StringBuilder();
        var r = model.chat(List.of(LlamaModel.Message.user("Reply with exactly the word: kiln")),
                LlamaModel.Options.DEFAULT.withTemperature(0).withMaxTokens(16), pieces::append);
        assertEquals(r.text(), pieces.toString());
        assertTrue(r.text().toLowerCase().contains("kiln"), r.text());
        assertTrue(r.promptTokens() > 0 && r.generatedTokens() > 0);
    }

    @Test
    void chatJsonReturnsToolCalls() {
        String req = """
            {"messages": [{"role": "user", "content": "What's the weather in Nashville right now?"}],
             "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
             "temperature": 0, "max_tokens": 128, "seed": 1}""";
        String res = model.chatJson(req, null);
        System.out.println(res);
        assertTrue(res.contains("\"finish_reason\":\"tool_calls\""), res);
        assertTrue(res.contains("get_weather") && res.contains("Nashville"), res);
    }

    @Test
    void chatJsonAnswersFromToolResult() {
        String req = """
            {"messages": [
               {"role": "user", "content": "What's the weather in Nashville right now?"},
               {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function",
                  "function": {"name": "get_weather", "arguments": "{\\\"city\\\": \\\"Nashville\\\"}"}}]},
               {"role": "tool", "tool_call_id": "call_0", "content": "{\\\"temp_f\\\": 71, \\\"sky\\\": \\\"clear\\\"}"}],
             "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
             "temperature": 0, "max_tokens": 128, "seed": 1}""";
        var streamed = new StringBuilder();
        String res = model.chatJson(req, streamed::append);
        System.out.println(res);
        assertTrue(res.contains("\"finish_reason\":\"stop\""), res);
        assertTrue(res.contains("71"), res);
        assertFalse(streamed.isEmpty());
    }
}
