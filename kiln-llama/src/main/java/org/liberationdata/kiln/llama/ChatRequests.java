package org.liberationdata.kiln.llama;

import org.liberationdata.kiln.llama.LlamaModel.Image;
import org.liberationdata.kiln.llama.LlamaModel.Message;
import org.liberationdata.kiln.llama.LlamaModel.Options;

import java.util.List;

/** Builds {@link LlamaModel#chatJson} requests from {@link Message}s (apart from LlamaModel, so it needs no native library). */
final class ChatRequests {
    private ChatRequests() {}

    /** The request for {@link LlamaModel#extract}. Adds the messages' images to images, in the order the request uses them. */
    static String extract(List<Message> messages, String jsonSchema, Options o, List<Image> images) {
        return "{\"messages\": " + messages(messages, images) + ", \"json_schema\": " + jsonSchema
                + ", \"max_tokens\": " + o.maxTokens() + ", \"temperature\": " + Math.max(0f, o.temperature())
                + ", \"top_p\": " + o.topP() + ", \"min_p\": " + o.minP() + ", \"top_k\": 0, \"seed\": " + o.seed() + "}";
    }

    /** OpenAI-format messages, with {"type": "image", "index": i} parts before the text for their images (added to images). */
    static String messages(List<Message> messages, List<Image> images) {
        StringBuilder b = new StringBuilder("[");
        for (Message m : messages) {
            if (b.length() > 1) b.append(", ");
            b.append("{\"role\": ").append(Json.quote(m.role())).append(", \"content\": ");
            String text = m.content() == null ? "" : m.content();
            if (m.images().isEmpty()) {
                b.append(Json.quote(text));
            } else {
                b.append('[');
                for (Image img : m.images()) {
                    b.append("{\"type\": \"image\", \"index\": ").append(images.size()).append("}, ");
                    images.add(img);
                }
                b.append("{\"type\": \"text\", \"text\": ").append(Json.quote(text)).append("}]");
            }
            b.append('}');
        }
        return b.append(']').toString();
    }
}
