package org.liberationdata.fornax.jitllm;

import org.liberationdata.fornax.Json;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

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
     * The response, as LlamaModel.chatJson gives it. A reply that starts with a {@code <think>} block has that block
     * as {@code reasoning_content}, as llama.cpp's parser gives it.
     */
    static String response(Reply r) {
        String[] parts = splitThinking(r.text());
        StringBuilder b = new StringBuilder("{\"message\": {\"role\": \"assistant\", \"content\": ").append(Json.quote(parts[1]));
        if (parts[0] != null) b.append(", \"reasoning_content\": ").append(Json.quote(parts[0]));
        return b.append("}, \"finish_reason\": ").append(Json.quote(r.finishReason()))
                .append(", \"usage\": {\"prompt_tokens\": ").append(r.promptTokens())
                .append(", \"completion_tokens\": ").append(r.generatedTokens()).append("}}").toString();
    }

    /** {reasoning or null, content}. */
    static String[] splitThinking(String text) {
        String t = text.stripLeading();
        if (t.startsWith("<think>")) {
            int end = t.indexOf("</think>");
            if (end < 0) return new String[]{t.substring(7).strip(), ""};
            return new String[]{t.substring(7, end).strip(), t.substring(end + 8).stripLeading()};
        }
        return new String[]{null, text};
    }

    /**
     * Streams what {@link #response} puts in {@code content}: pieces in a leading {@code <think>} block are held back,
     * as llama.cpp streams only the content.
     */
    static Consumer<String> contentOnly(Consumer<String> onContent) {
        if (onContent == null) return null;
        StringBuilder head = new StringBuilder();   // the start of the reply, until we know whether it thinks
        boolean[] state = {false, false};           // {decided, thinking}
        return piece -> {
            if (state[0] && !state[1]) { onContent.accept(piece); return; }
            head.append(piece);
            if (!state[0]) {
                String t = head.toString().stripLeading();
                if (t.isEmpty() || "<think>".startsWith(t)) return;   // not enough yet
                state[0] = true;
                state[1] = t.startsWith("<think>");
                if (!state[1]) { onContent.accept(head.toString()); head.setLength(0); return; }
            }
            int end = head.indexOf("</think>");
            if (end < 0) return;
            String rest = head.substring(end + 8).stripLeading();
            head.setLength(0);
            state[1] = false;
            if (!rest.isEmpty()) onContent.accept(rest);
        };
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
