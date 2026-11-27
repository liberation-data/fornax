package org.liberationdata.fornax.snippets;

// --8<-- [start:chat]
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Image;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;

import java.nio.file.Path;
import java.util.List;
// --8<-- [end:chat]
// --8<-- [start:images]
import java.awt.image.BufferedImage;
// --8<-- [end:images]

/** The snippets of website/src/content/docs/docs/guides/vision.md: compiled, never run. */
class Vision {
    void chat() {
        // --8<-- [start:chat]
        try (var vlm = new LlamaModel(LlamaModel.Weights.QWEN3_VL_4B)) {        // the GGUF and its mmproj, 8192 tokens of context
            var question = Message.user("What is in this picture?", Image.read(Path.of("photo.jpg")));
            vlm.chat(List.of(question), Options.DEFAULT.withTemperature(0).withMaxTokens(512), System.out::print);
        }
        // --8<-- [end:chat]
    }

    void images(byte[] bytes, BufferedImage bufferedImage) {
        // --8<-- [start:images]
        Image a = Image.read(Path.of("scan.png"));          // a file ImageIO can read: PNG, JPEG, BMP, GIF, ...
        Image b = Image.decode(bytes);                      // encoded bytes, from an upload say
        Image c = Image.of(bufferedImage);                  // any Java2D image
        // --8<-- [end:images]
    }

    void load() {
        // --8<-- [start:load]
        try (var vlm = new LlamaModel("/models/model.gguf", "/models/mmproj.gguf")) {   // or (gguf, mmproj, gpuLayers, contextSize)
            // ...
            vlm.cancel(); // --8<-- [skip]
        }
        // --8<-- [end:load]
    }

    void extract() {
        // --8<-- [start:extract]
        String schema = """
                {"type": "object",
                 "properties": {
                   "vendor": {"type": "string"},
                   "invoice_number": {"type": "string"},
                   "date": {"type": "string", "description": "YYYY-MM-DD"},
                   "total": {"type": "number"}},
                 "required": ["vendor", "invoice_number", "date", "total"]}""";

        try (var vlm = new LlamaModel(LlamaModel.Weights.QWEN3_VL_4B)) {
            String json = vlm.extract(
                    List.of(Message.user("Extract this invoice's fields.", Image.read(Path.of("invoice.png")))),
                    schema, Options.DEFAULT.withTemperature(0).withMaxTokens(1024));

            Object invoice = org.liberationdata.fornax.llama.Json.parse(json);
            System.out.println(org.liberationdata.fornax.llama.Json.get(invoice, "total"));
        }
        // --8<-- [end:extract]
    }

    void chatJson(LlamaModel vlm) {
        // --8<-- [start:chat-json]
        String reply = vlm.chatJson("""
                {"messages": [{"role": "user", "content": [
                     {"type": "image", "index": 0},
                     {"type": "text", "text": "Read this invoice."}]}],
                 "json_schema": {"type": "object", "properties": {"total": {"type": "number"}}, "required": ["total"]},
                 "temperature": 0}""",
                List.of(Image.read(Path.of("invoice.png"))), null);
        // --8<-- [end:chat-json]
    }
}
