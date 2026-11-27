package org.liberationdata.fornax.jitllm;

import org.liberationdata.fornax.Json;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Locale;
import java.util.Set;
import java.util.function.Consumer;
import java.util.regex.Pattern;

/**
 * {@link JitLlmModel#chatJson}'s request and response, in the shape {@code LlamaModel.chatJson} takes and gives, so
 * fornax-server serves either engine. Needs no jitllm classes.
 */
final class ChatJson {
    private ChatJson() {}

    /** Return code 7 is what fornax-server reports as a 400 (bad input), as it does for llama.cpp's. */
    static IllegalStateException badRequest(String message) {
        return new IllegalStateException("jitllm (rc=7): " + message);
    }

    /** A chat: the whole conversation and its sampling. seed below 0 for a random one; topP 0 for the engine's own. */
    record Request(List<Message> messages, int maxTokens, float temperature, float topP, long seed, List<String> stop) {
        Request {
            messages = List.copyOf(messages);
            stop = List.copyOf(stop);
            if (messages.isEmpty()) throw badRequest("messages is empty");
            if (maxTokens < 1) throw badRequest("max_tokens must be at least 1");
        }

        /** maxTokens 0 (or less) generates until the model stops or the context fills, as LlamaModel does. */
        static Request of(List<Message> messages, Options o) {
            return new Request(messages, o.maxTokens() > 0 ? o.maxTokens() : Integer.MAX_VALUE, o.temperature(), o.topP(), o.seed(), List.of());
        }
    }

    /** What the engine gave. finishReason: OpenAI's "stop" or "length". */
    record Reply(String text, int promptTokens, int generatedTokens, String finishReason) {
        Result result() { return new Result(text, promptTokens, generatedTokens); }
    }

    private static final Set<String> ROLES = Set.of("system", "developer", "user", "assistant");

    /** An OpenAI chat completions request's messages and sampling; what jitllm can't do is refused. */
    static Request parse(String json) {
        Object doc;
        try { doc = Json.parse(json); }
        catch (RuntimeException e) { throw badRequest("the request isn't JSON: " + e.getMessage()); }
        if (!(doc instanceof Map<?, ?> req)) throw badRequest("the request isn't a JSON object");
        for (String unsupported : List.of("json_schema", "grammar"))
            if (req.get(unsupported) != null) throw badRequest(unsupported + " isn't supported by the jitllm engine");
        if (req.get("tools") instanceof List<?> tools && !tools.isEmpty())
            throw badRequest("tools aren't supported by the jitllm engine");
        if (req.get("response_format") instanceof Map<?, ?> rf && !"text".equals(rf.get("type")))
            throw badRequest("response_format " + rf.get("type") + " isn't supported by the jitllm engine");
        if (!(req.get("messages") instanceof List<?> list)) throw badRequest("messages must be an array");
        List<Message> messages = new ArrayList<>(list.size());
        for (Object o : list) {
            if (!(o instanceof Map<?, ?> m) || !(m.get("role") instanceof String role))
                throw badRequest("each message needs a role");
            if (!ROLES.contains(role.toLowerCase(Locale.ROOT)))
                throw badRequest("the jitllm engine takes system, developer, user and assistant messages, not " + role);
            if (m.get("tool_calls") instanceof List<?> calls && !calls.isEmpty())
                throw badRequest("tool calls aren't supported by the jitllm engine");
            messages.add(new Message(role, content(m.get("content"))));
        }
        Object max = req.get("max_completion_tokens") != null ? req.get("max_completion_tokens") : req.get("max_tokens");
        return new Request(messages,
                max == null ? Options.DEFAULT.maxTokens() : number(max, "max_tokens").intValue(),
                req.get("temperature") == null ? Options.DEFAULT.temperature() : number(req.get("temperature"), "temperature").floatValue(),
                req.get("top_p") == null ? 0f : number(req.get("top_p"), "top_p").floatValue(),
                req.get("seed") == null ? -1 : number(req.get("seed"), "seed").longValue(),
                stops(req.get("stop")));
    }

    /**
     * The response, as LlamaModel.chatJson gives it. A reply's reasoning, a leading {@code <think>} block or the text
     * before a {@code </think>} whose opening tag was in the prompt, is {@code reasoning_content}, as llama.cpp's
     * parser gives it.
     */
    static String response(Reply r) {
        String[] parts = splitThinking(r.text());
        StringBuilder b = new StringBuilder("{\"message\": {\"role\": \"assistant\", \"content\": ").append(Json.quote(parts[1]));
        if (parts[0] != null) b.append(", \"reasoning_content\": ").append(Json.quote(parts[0]));
        return b.append("}, \"finish_reason\": ").append(Json.quote(r.finishReason()))
                .append(", \"usage\": {\"prompt_tokens\": ").append(r.promptTokens())
                .append(", \"completion_tokens\": ").append(r.generatedTokens()).append("}}").toString();
    }

    private static final String OPEN = "<think>", CLOSE = "</think>";

    /**
     * {reasoning or null, content}. Some models (DeepSeek-R1 distills, Qwen3 thinking variants) have the opening
     * {@code <think>} in their prompt, so their reply starts inside the reasoning and has only the closing tag.
     */
    static String[] splitThinking(String text) {
        String t = text.stripLeading();
        boolean opened = t.startsWith(OPEN);
        int from = opened ? OPEN.length() : 0;
        int end = t.indexOf(CLOSE, from);
        if (end < 0) return opened ? new String[]{t.substring(from).strip(), ""} : new String[]{null, text};
        return new String[]{t.substring(from, end).strip(), t.substring(end + CLOSE.length()).stripLeading()};
    }

    /**
     * Streams what {@link #response} puts in {@code content}, as llama.cpp streams only the content. A reply that
     * starts with {@code <think>} is held back to its {@code </think>}. One without the tag might have started inside
     * the reasoning (its {@code <think>} in the prompt), so its start is held back until a {@code </think>} makes it
     * reasoning, or it has run {@link #UNSURE} characters without one and without opening the way reasoning does
     * ("Okay, the user..."). {@link #finish} lets out what is held back once the reply is complete.
     */
    static final class Content implements Consumer<String> {
        /** How much of a reply's start is held back, without a tag or a reasoning-like opening, before it is let out. */
        static final int UNSURE = 200;
        private static final Pattern REASONING = Pattern.compile(
                "(?i)(okay|alright|hmm+|let me|let's|i need|we need|we are|the user)\\b.*", Pattern.DOTALL);

        private final Consumer<String> out;
        private final StringBuilder held = new StringBuilder();
        private boolean deciding = true, thinking, open;   // open: the content is being let out
        private boolean trim;                              // after a </think>: whitespace before the content is dropped

        Content(Consumer<String> out) { this.out = out; }

        @Override public void accept(String piece) {
            if (open) { emit(piece); return; }
            held.append(piece);
            String t = held.toString().stripLeading();
            if (deciding) {
                if (t.isEmpty() || OPEN.startsWith(t)) return;   // not enough yet
                if (t.startsWith(OPEN)) { deciding = false; thinking = true; }
            }
            int end = held.indexOf(CLOSE);
            if (end >= 0) {
                String rest = held.substring(end + CLOSE.length()).stripLeading();
                held.setLength(0);
                deciding = thinking = false;
                open = trim = true;
                emit(rest);
                return;
            }
            if (deciding && t.length() >= UNSURE && !REASONING.matcher(t).matches()) release();
        }

        /** The reply is complete: a start still held back without a {@code </think>} was content after all. */
        void finish() {
            if (deciding && !open) release();
        }

        private void emit(String piece) {
            if (trim) {
                piece = piece.stripLeading();
                if (piece.isEmpty()) return;
                trim = false;
            }
            out.accept(piece);
        }

        private void release() {
            deciding = false;
            open = true;
            String text = held.toString();
            held.setLength(0);
            if (!text.isEmpty()) out.accept(text);
        }
    }

    /** A message's content: a string, or OpenAI's array of parts, of which only text is taken. */
    private static String content(Object c) {
        if (c == null) return "";
        if (c instanceof String s) return s;
        if (c instanceof List<?> parts) {
            StringBuilder b = new StringBuilder();
            for (Object p : parts) {
                if (!(p instanceof Map<?, ?> part)) throw badRequest("a content part must be an object");
                Object type = part.get("type");
                if ("text".equals(type)) b.append(part.get("text") instanceof String s ? s : "");
                else throw badRequest("content of type " + type + " isn't supported by the jitllm engine (it takes text only)");
            }
            return b.toString();
        }
        throw badRequest("content must be a string or an array of parts");
    }

    private static List<String> stops(Object stop) {
        if (stop == null) return List.of();
        if (stop instanceof String s) return List.of(s);
        if (stop instanceof List<?> l) {
            List<String> out = new ArrayList<>();
            for (Object o : l) if (o instanceof String s && !s.isEmpty()) out.add(s);
            return out;
        }
        throw badRequest("stop must be a string or an array of strings");
    }

    private static Number number(Object v, String name) {
        if (v instanceof Number n) return n;
        throw badRequest(name + " must be a number");
    }
}
