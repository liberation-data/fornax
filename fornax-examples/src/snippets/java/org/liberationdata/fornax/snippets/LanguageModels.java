package org.liberationdata.fornax.snippets;

// --8<-- [start:load]
import org.liberationdata.fornax.llama.LlamaModel;
// --8<-- [end:load]
// --8<-- [start:chat]
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.List;
// --8<-- [end:chat]
// --8<-- [start:cancel]
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
// --8<-- [end:cancel]
// --8<-- [start:json-schema]
// --8<-- [start:tool-calls]
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
// --8<-- [end:json-schema]
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
// --8<-- [end:tool-calls]

/** The snippets of website/src/content/docs/docs/guides/language-models.md: compiled, never run. */
class LanguageModels {
    void load() {
        // --8<-- [start:load]
        try (var llm = new LlamaModel("unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M")) {   // all layers on the GPU, 8192 tokens
            // ...
            llm.cancel(); // --8<-- [skip]
        }

        try (var llm = new LlamaModel("unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M", -1, 32768)) {   // gpuLayers (-1 = all), contextSize (0 = the model's own)
            // ...
            llm.cancel(); // --8<-- [skip]
        }
        // --8<-- [end:load]
    }

    void chat(LlamaModel llm) {
        // --8<-- [start:chat]
        Result r = llm.chat(List.of(
                Message.system("You are a Nashville songwriter. Be concise."),
                Message.user("Give me three song titles about a robot.")), Options.DEFAULT);

        System.out.println(r.text());
        System.out.println(r.promptTokens() + " prompt tokens, " + r.generatedTokens() + " generated");
        // --8<-- [end:chat]
    }

    void options() {
        // --8<-- [start:options]
        Options exact = Options.DEFAULT.withTemperature(0).withMaxTokens(1024);   // greedy, for repeatable answers
        Options again = Options.DEFAULT.withSeed(42);                           // sampled, but the same each run
        Options custom = new Options(256, 0.8f, 0.95f, 0.05f, 7);              // every field at once
        // --8<-- [end:options]
    }

    void stream(LlamaModel llm) {
        // --8<-- [start:stream]
        llm.chat(List.of(Message.user("Write a haiku about a fornax.")), Options.DEFAULT, System.out::print);
        // --8<-- [end:stream]
    }

    void cancel(LlamaModel llm) {
        // --8<-- [start:cancel]
        var timer = Executors.newSingleThreadScheduledExecutor();
        timer.schedule(llm::cancel, 5, TimeUnit.SECONDS);              // give up after five seconds
        try {
            Result partial = llm.chat(List.of(Message.user("Tell me a long story.")), Options.DEFAULT.withMaxTokens(0),
                    piece -> {});
            System.out.println(partial.text());
        } finally {
            timer.shutdownNow();
        }
        // --8<-- [end:cancel]
    }

    void chatJson(LlamaModel llm) {
        // --8<-- [start:chat-json]
        String reply = llm.chatJson("""
                {"messages": [{"role": "user", "content": "Say hello in French."}],
                 "temperature": 0, "max_tokens": 64}""", null);
        // --8<-- [end:chat-json]
    }

    void jsonSchema(LlamaModel llm) throws Exception {
        // --8<-- [start:json-schema]
        var json = new ObjectMapper();
        String request = """
                {"messages": [{"role": "user",
                               "content": "Invent a country song: its title, its key and its tempo in beats per minute."}],
                 "json_schema": {"type": "object",
                                 "properties": {"title": {"type": "string"},
                                                "key": {"type": "string"},
                                                "bpm": {"type": "integer"}},
                                 "required": ["title", "key", "bpm"]},
                 "temperature": 0, "max_tokens": 256}""";

        JsonNode reply = json.readTree(llm.chatJson(request, null));
        JsonNode song = json.readTree(reply.at("/message/content").asText());   // the content is the JSON document
        System.out.println(song.get("title").asText() + ", " + song.get("key").asText() + ", " + song.get("bpm").asInt() + " bpm");
        // --8<-- [end:json-schema]
    }

    void toolCalls(LlamaModel llm) throws Exception {
        // --8<-- [start:tool-calls]
        var json = new ObjectMapper();
        ObjectNode request = (ObjectNode) json.readTree("""
                {"messages": [{"role": "user", "content": "What's the weather in Nashville right now?"}],
                 "tools": [{"type": "function", "function": {
                     "name": "get_weather",
                     "description": "Current weather for a city",
                     "parameters": {"type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"]}}}],
                 "temperature": 0, "max_tokens": 256, "seed": 1}""");
        ArrayNode messages = (ArrayNode) request.get("messages");

        JsonNode reply = json.readTree(llm.chatJson(request.toString(), null));
        while ("tool_calls".equals(reply.get("finish_reason").asText())) {
            JsonNode message = reply.get("message");
            messages.add(message);                                     // the assistant's turn, with its tool_calls
            for (JsonNode call : message.get("tool_calls")) {
                String name = call.at("/function/name").asText();
                JsonNode args = json.readTree(call.at("/function/arguments").asText());
                String result = name.equals("get_weather")
                        ? "{\"city\": \"" + args.get("city").asText() + "\", \"temp_c\": 24, \"sky\": \"clear\"}"
                        : "{\"error\": \"unknown tool\"}";
                messages.addObject().put("role", "tool").put("tool_call_id", call.get("id").asText()).put("content", result);
            }
            reply = json.readTree(llm.chatJson(request.toString(), null));
        }
        System.out.println(reply.at("/message/content").asText());
        // --8<-- [end:tool-calls]
    }

    void complete(LlamaModel llm) {
        // --8<-- [start:complete]
        Result r = llm.complete("The three most common glazes for stoneware are", Options.DEFAULT.withMaxTokens(64), null);
        // --8<-- [end:complete]
    }
}
