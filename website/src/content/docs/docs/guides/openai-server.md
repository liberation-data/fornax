---
title: OpenAI-compatible server
description: Serve Fornax's models over an OpenAI-compatible HTTP API with fornax server or FornaxServer, so any OpenAI client, agent framework or tool can use them with no Java code.
---

`fornax server` puts Fornax's engines behind an HTTP API in OpenAI's shape. Anything that talks to OpenAI (the
official Python and JavaScript clients, LangChain, an agent framework, a chat UI, `curl`) can then use local models
by pointing its base URL at the server:

```sh
fornax server -m qwen3-4b
curl -s localhost:54154/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:54154/v1", api_key="unused")
reply = client.chat.completions.create(model="qwen3-4b", messages=[{"role": "user", "content": "Hi"}])
print(reply.choices[0].message.content)
```

It is the module `fornax-server`, JDK only (`com.sun.net.httpserver`, no framework), which an app can also start
itself:

```java
import org.liberationdata.fornax.server.FornaxServer;

try (var server = FornaxServer.builder().port(54154).model("qwen3-4b").build().start()) {
    System.out.println("OpenAI base URL: " + server.url());   // http://127.0.0.1:54154/v1
    server.await();
}
```

## Endpoints

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | chat with a language or vision model, streamed or not |
| `GET /v1/models` | the models it was started with, those loaded, and those downloaded |
| `POST /v1/embeddings` | embeddings, as floats or base64 |
| `POST /v1/images/generations` | images, as base64 PNGs (`b64_json`) |
| `POST /v1/audio/speech` | text to speech, as a WAV |
| `POST /v1/audio/transcriptions` | speech to text, from a WAV file |

`GET /health` answers `{"status": "ok"}`. Errors are OpenAI's: the HTTP status and
`{"error": {"message": ..., "type": ..., "param": ..., "code": ...}}`, so a client's own exceptions work
(`openai.NotFoundError` for a model that isn't there, for one).

### Chat completions

Chat runs [`LlamaModel.chatJson`](/docs/guides/language-models/) underneath: the model's own chat template and
tool-call format, with llama-server's templates and parsers. A request takes:

- `messages`, with `system`, `user`, `assistant` and `tool` roles, as OpenAI sends them
- `tools` and `tool_choice` (`auto`, `none`, `required`, or a function by name); the reply's `tool_calls` and
  `finish_reason: "tool_calls"` come back as OpenAI's, and the tool results go back as `tool` messages
- `response_format`: `json_schema` (the reply is constrained to the schema) or `json_object`
- `temperature`, `top_p`, `seed`, `max_tokens` (or `max_completion_tokens`), and `top_k` and `min_p` beyond OpenAI's
- `stop`: a string or a list; the reply ends before the first one, and the model stops there
- `stream: true`: server-sent events in OpenAI's chunk format, ending with `data: [DONE]`;
  `stream_options: {"include_usage": true}` adds the token counts in a last chunk
- images, for a vision model (`qwen3-vl-4b`, `paddleocr-vl`, `medgemma-4b`): `image_url` content parts with a base64
  `data:` URI; the server doesn't fetch URLs
- `chat_template_kwargs: {"enable_thinking": false}` turns a reasoning model's thinking off

A reasoning model's thinking comes back in `reasoning_content`, apart from `content`. When streaming, the content
arrives as it is generated; the tool calls and the thinking come in a chunk at the end, before the finish reason.
`n` must be 1, and there are no logprobs.

When a client disconnects mid-stream, the generation stops, so the model is free for the next request. A
non-streaming request runs to the end.

### Embeddings

`input` is a string or a list of them (token arrays aren't taken). The model is `qwen3-embedding-0.6b` (Qwen3-Embedding
0.6B, 1024 dimensions) unless a request names another embedding GGUF by `org/name:TAG` or path (one the server was
started with, or any with `--allow-downloads`). `dimensions` keeps
the first n and normalises them again, which Qwen3-Embedding is trained for. `encoding_format: "base64"`, which the
OpenAI clients ask for by default, is supported. Vectors are normalised, so a dot product is the cosine similarity.
`usage` reports 0 tokens.

### Images

```python
import base64
r = client.images.generate(model="flux2-klein", prompt="a brick kiln at dusk", size="512x512",
                           extra_body={"seed": 7, "steps": 20})
open("kiln.png", "wb").write(base64.b64decode(r.data[0].b64_json))
```

`model` is an image alias (`flux2-klein`, `qwen-image`); `size` is `WIDTHxHEIGHT` in multiples of 16 (default
1024x1024); `n` makes up to 8, with seeds counting up from `seed`. Beyond OpenAI's: `seed`, `steps` (default 20)
and `cfg_scale` (4.0 for flux2-klein, 2.5 for the Qwen-Image models). Images come back as `b64_json` PNGs; the
server keeps no files, so `response_format: "url"` is refused.

### Speech and transcription

`/v1/audio/speech` speaks `input` with `kokoro` unless `model` names another speech alias. `voice` is one of the
model's voices (`fornax speak --list-voices`); OpenAI's voice names (`alloy`, `nova`, ...) get the model's default.
The reply is a WAV, so ask for `response_format: "wav"` (an OpenAI client asking for MP3 gets an error).

`/v1/audio/transcriptions` takes a WAV `file` (multipart, as the clients send it; convert other formats first,
`ffmpeg -i in.mp3 in.wav`), with `whisper-base.en` unless `model` names another (`whisper-large-v3-turbo` for 99
languages), and `language` and `prompt`. `response_format` is `json` (the default), `text`, `srt`, `vtt` or
`verbose_json`, which has the segments and each word's start and end.

## Models

A request's `model` is one of the `-m` models, or a [catalogue model](/docs/guides/models/#the-catalogue) that is
already downloaded (`fornax pull`), as is `qwen3-embedding-0.6b`. Any other is refused with a 404 (`model_not_found`, whose message says how to
allow it), so that whoever can call the server can't make it download many gigabytes. `--allow-downloads`
(`.allowDownloads(true)`) lets a request name anything Fornax loads: a catalogue alias, an `org/name[:TAG]` reference
or a path to a GGUF, downloaded the first time it's asked for.

A model is loaded on first use and kept loaded in a [`ModelPool`](/docs/guides/model-pool/): within `--budget-gb` (models' file sizes), the least recently used are
unloaded to make room, and a model unused for `--idle-minutes` is unloaded. A model answers one request at a time;
others wait for it.

The `-m` models are downloaded first if need be and loaded before the server starts answering. The first chat model
answers a request that names none. A reference or path whose name has "embed" in it is loaded for embeddings; any
other for chat.

## Options

```text
fornax server [-m MODEL]... [options]
  -m, --model M        a model to load at start (repeat for more): an alias, org/name:TAG or a path
  --port N             default 54154
  --host ADDR          default 127.0.0.1 (this machine only); 0.0.0.0 for every interface, with an API key
  --api-key K          require Authorization: Bearer K (default FORNAX_API_KEY, if set)
  --cors ORIGIN        let web pages from ORIGIN call it (repeat for more; * for any)
  --budget-gb N        memory for loaded models, by their files' size (default 24)
  --idle-minutes N     unload a model unused this long (default 15; 0 = never)
  --ctx N              context size for language and embedding models (default 8192; 0 = the model's own)
  --allow-downloads    let a request name any model, downloading it on first use (default: only the -m
                       models and catalogue models already downloaded)
```

`FornaxServer.builder()` has the same settings, plus the hub to download with, a download progress listener and a
log line consumer.

## Who may call it

The server listens on 127.0.0.1 unless `--host` says otherwise, so only programs on the same machine can reach it.
That alone doesn't keep out the web pages open in your browser, which can send requests to 127.0.0.1 too. So:

- **On loopback, the `Host` must be the server's own** (`127.0.0.1:54154`, `localhost:54154` or `[::1]:54154`). A
  page that rebinds its own domain name to 127.0.0.1 sends its own name, and is refused (403).
- **A request from a web page is refused** (403): browsers send an `Origin` with every cross-origin request.
  `--cors https://app.example` allows that origin, which then gets CORS headers, its preflight answered and the
  response readable. `--cors '*'` allows any origin.
- **`--api-key`** (or `FORNAX_API_KEY`) makes every request need `Authorization: Bearer <key>` (401 otherwise), which
  is what the OpenAI clients send as their `api_key`. Set one before listening on another address: the `Host` check
  is off then, as other machines reach it by other names.
