---
title: Language models
description: Chat with a GGUF model through llama.cpp from Java, stream the reply, constrain it to a JSON schema, and let the model call your tools.
---

`fornax-llama` runs any GGUF language model through [llama.cpp](https://github.com/ggml-org/llama.cpp): Qwen, Llama,
Gemma, Mistral and the rest. The class is `LlamaModel`. The same class embeds text and reranks documents: see
[Embeddings and reranking](/docs/guides/embeddings/).

## Load a model

<!-- snippet: LanguageModels#load -->
```java
import org.liberationdata.fornax.llama.LlamaModel;

try (var llm = new LlamaModel("unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M")) {   // all layers on the GPU, 8192 tokens
    // ...
}

try (var llm = new LlamaModel("unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M", -1, 32768)) {   // gpuLayers (-1 = all), contextSize (0 = the model's own)
    // ...
}
```

The first argument is a path to a `.gguf` file, or a model by name:

- **`owner/repo:TAG`**, a Hugging Face repo and a quant (`Q4_K_M`, `Q8_0`, matched against the repo's GGUF file
  names, ignoring case) or a file name. `owner/repo` alone takes the repo's only GGUF model. Add `@revision` for a
  branch, tag or commit other than `main`.
- **a catalogue alias**, such as `qwen3-4b` or `qwen3-vl-4b`, which brings the model's mmproj too (see
  [Downloading models](/docs/guides/models/)).

The first time, Fornax downloads the model into the Hugging Face cache, logging its progress (through
`System.Logger` `org.liberationdata.fornax.hub`). After that, the model loads from the cache with no network, under
`HF_HUB_OFFLINE` too. A path that names an existing file is always a path, and a URL is refused with the reference to
use instead. To download ahead of time, run `fornax pull unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M` or
`hf download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf`; both use the same cache.

The constructor loads the weights onto the GPU, and they stay there until `close()`. Load once and reuse the model
for many calls; [ModelPool](/docs/guides/model-pool/) helps when an app has several models to juggle.

A `LlamaModel` can be called from any thread; its calls run one at a time, in arrival order. From virtual threads,
use `chatAsync`, `chatJsonAsync` and `extractAsync`, which return a `CompletableFuture` and stream to a
`Flow.Subscriber`: see [Threading](/docs/guides/threading/).

## Chat

<!-- snippet: LanguageModels#chat -->
```java
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.List;

Result r = llm.chat(List.of(
        Message.system("You are a Nashville songwriter. Be concise."),
        Message.user("Give me three song titles about a robot.")), Options.DEFAULT);

System.out.println(r.text());
System.out.println(r.promptTokens() + " prompt tokens, " + r.generatedTokens() + " generated");
```

`chat` formats the messages with the model's chat template (through llama.cpp's built-in template support) and
generates until the end of the assistant's turn, the token limit, or a full context. Each call starts from an empty
context: send the whole conversation every time, with earlier replies as `Message.assistant(...)`.

### Options

`Options` is a record of the sampling settings:

| Field | `Options.DEFAULT` | Meaning |
|---|---|---|
| `maxTokens` | 512 | tokens to generate at most; 0 or less means until the end of the turn or a full context |
| `temperature` | 0.7 | 0 or less means greedy (always the most likely token) |
| `topP` | 0.9 | nucleus sampling |
| `minP` | 0.05 | min-p sampling |
| `seed` | -1 | -1 means random |

<!-- snippet: LanguageModels#options -->
```java
Options exact = Options.DEFAULT.withTemperature(0).withMaxTokens(1024);   // greedy, for repeatable answers
Options again = Options.DEFAULT.withSeed(42);                           // sampled, but the same each run
Options custom = new Options(256, 0.8f, 0.95f, 0.05f, 7);              // every field at once
```

## Streaming

Pass a `Consumer<String>` and each piece of text arrives as soon as it is generated. The call still returns the whole
`Result` at the end.

<!-- snippet: LanguageModels#stream -->
```java
llm.chat(List.of(Message.user("Write a haiku about a fornax.")), Options.DEFAULT, System.out::print);
```

The callback runs on the calling thread, between tokens, so keep it quick: append to a buffer, push to a queue or
write to a socket. If it throws, the generation stops and `chat` throws that exception. For a subscriber on its own
thread, with a bounded buffer, use `chatAsync(messages, options, subscriber)` (see [Threading](/docs/guides/threading/#streaming)).

To stop early, call `cancel()` from any other thread (or from the callback). The generation stops after the current
token and the call returns what it has so far. It works with or without a callback.

<!-- snippet: LanguageModels#cancel -->
```java
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

var timer = Executors.newSingleThreadScheduledExecutor();
timer.schedule(llm::cancel, 5, TimeUnit.SECONDS);              // give up after five seconds
try {
    Result partial = llm.chat(List.of(Message.user("Tell me a long story.")), Options.DEFAULT.withMaxTokens(0),
            piece -> {});
    System.out.println(partial.text());
} finally {
    timer.shutdownNow();
}
```

## Requests in OpenAI's format: `chatJson`

`chatJson` takes a chat completions request as JSON and returns the reply as JSON. It renders the model's own
(Jinja) chat template and parses its reply with the same code as `llama-server`, so it understands tool calls and
reasoning in each model's own format.

<!-- snippet: LanguageModels#chat-json -->
```java
String reply = llm.chatJson("""
        {"messages": [{"role": "user", "content": "Say hello in French."}],
         "temperature": 0, "max_tokens": 64}""", null);
```

The reply has this shape:

```json
{"message": {"role": "assistant", "content": "..."},
 "finish_reason": "stop",
 "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
```

`message` can also hold `reasoning_content` (a thinking model's reasoning, kept out of `content`) and `tool_calls`.
`finish_reason` is `stop`, `length` (the token limit or the context ran out) or `tool_calls`.

The request keys it reads:

| Key | Default | |
|---|---|---|
| `messages` | (required) | OpenAI chat messages, including `tool` messages with a `tool_call_id` |
| `tools` | none | OpenAI function tools |
| `tool_choice` | `auto` | `auto`, `none` or `required` |
| `parallel_tool_calls` | `true` | |
| `json_schema` | none | a JSON Schema the reply must follow (see below) |
| `enable_thinking` | `true` | `false` asks a thinking model to skip its reasoning, where its template allows that |
| `temperature`, `top_p`, `min_p`, `top_k` | 0.7, 0.95, 0.05, 40 | sampling |
| `max_tokens` | until done | |
| `seed` | random | |

Its defaults are llama-server's, so they differ a little from `Options.DEFAULT` (`top_p` 0.95 rather than 0.9, and no
token limit).

The second argument streams the message's `content` as it is generated (not its reasoning, and not tool calls), or
pass `null`. `cancel()` works here as it does for `chat`, including while the model reasons or writes a tool call.

Fornax has no JSON library of its own on `main`; the examples below use Jackson, but any JSON library works.

### A reply that follows a JSON schema

Put a JSON Schema at the top level of the request, under `json_schema`. llama.cpp turns it into a grammar, so the
model can only produce JSON that fits.

<!-- snippet: LanguageModels#json-schema -->
```java
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

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
```

Check `finish_reason`: with `length`, the model ran out of tokens and the JSON is cut off. Raise `max_tokens`.

`json_schema` is a top-level key, as in llama-server. OpenAI's `response_format` isn't read.

### Tool calls

Describe your tools in OpenAI's format. When the model wants one, the reply's `finish_reason` is `tool_calls` and
`message.tool_calls` says which function to call with which arguments. Run it, add the model's message and your
result to the conversation, and ask again.

<!-- snippet: LanguageModels#tool-calls -->
```java
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

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
```

Every tool call has an `id` (Fornax numbers them `call_0`, `call_1`, ... when the model doesn't give one), which your
`tool` message refers back to. How well a model picks tools and fills in arguments is up to the model; small models
do best with few tools and clear descriptions.

## Raw completion

`complete` skips the chat template and continues the text you give it exactly as written. Use it for base models, or
when you format the prompt yourself.

<!-- snippet: LanguageModels#complete -->
```java
Result r = llm.complete("The three most common glazes for stoneware are", Options.DEFAULT.withMaxTokens(64), null);
```

## Try it

```sh
./gradlew chat --args='"Write a haiku about a fornax"'
./gradlew chat --args='"Hello"' -Pmodel=/path/to/another-model.gguf
```

The example streams the reply and prints the prompt and generated token counts and the speed. On the Radeon 8060S
with Vulkan, Qwen3 4B Instruct (Q4_K_M) streams about 60 tokens a second.
