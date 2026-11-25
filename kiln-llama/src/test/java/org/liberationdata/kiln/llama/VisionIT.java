package org.liberationdata.kiln.llama;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.llama.LlamaModel.Image;
import org.liberationdata.kiln.llama.LlamaModel.Message;
import org.liberationdata.kiln.llama.LlamaModel.Options;
import org.liberationdata.kiln.llama.LlamaModel.Weights;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.Base64;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Runs Qwen3-VL 4B on small generated images when it's in the Hugging Face cache and the native library is built;
 * skipped otherwise.
 */
class VisionIT {
    static final Weights W = Weights.QWEN3_VL_4B;
    static final Options GREEDY = Options.DEFAULT.withTemperature(0).withMaxTokens(32);
    static LlamaModel model;

    @BeforeAll
    static void load() {
        assumeTrue(W.downloaded(), "model not downloaded: " + W.downloadCommand());
        model = new LlamaModel(W, -1, 4096);
    }

    @AfterAll
    static void close() { if (model != null) model.close(); }

    /** A red square on white. */
    static BufferedImage redSquare() {
        var img = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 256, 256);
        g.setColor(Color.RED);
        g.fillRect(64, 64, 128, 128);
        g.dispose();
        return img;
    }

    @Test
    void supportsVision() { assertTrue(model.supportsVision()); }

    @Test
    void chatNamesTheColourOfASquare() {
        var pieces = new StringBuilder();
        var r = model.chat(List.of(Message.user("What colour is the square? Answer in one word.", Image.of(redSquare()))),
                GREEDY, pieces::append);
        System.out.println(r);
        assertTrue(r.text().toLowerCase().contains("red"), r.text());
        assertEquals(r.text(), pieces.toString());
        assertTrue(r.promptTokens() > 20, "image tokens should count: " + r.promptTokens());
    }

    @Test
    void completeWithAMarker() {
        String prompt = "<|im_start|>user\n<|vision_start|>" + LlamaModel.mediaMarker() + "<|vision_end|>"
                + "What colour is the square? Answer in one word.<|im_end|>\n<|im_start|>assistant\n";
        var r = model.complete(prompt, List.of(Image.of(redSquare())), GREEDY, null);
        assertTrue(r.text().toLowerCase().contains("red"), r.text());
    }

    @Test
    void extractFollowsTheSchema() {
        String schema = """
            {"type": "object", "properties": {"shape": {"type": "string", "enum": ["circle", "square", "triangle"]},
             "colour": {"type": "string"}}, "required": ["shape", "colour"]}""";
        String json = model.extract(List.of(Message.user("Describe the shape.", Image.of(redSquare()))), schema,
                Options.DEFAULT.withTemperature(0).withMaxTokens(64));
        System.out.println(json);
        var m = (Map<?, ?>) Json.parse(json);
        assertEquals("square", m.get("shape"));
        assertTrue(m.get("colour").toString().toLowerCase().contains("red"), json);
    }

    @Test
    void chatJsonTakesAnImageUrl() throws Exception {
        var png = new ByteArrayOutputStream();
        ImageIO.write(redSquare(), "png", png);
        String url = "data:image/png;base64," + Base64.getEncoder().encodeToString(png.toByteArray());
        String req = """
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "%s"}},
               {"type": "text", "text": "What colour is the square? Answer in one word."}]}],
             "temperature": 0, "max_tokens": 16}""".formatted(url);
        String res = model.chatJson(req, List.of(), null);
        assertTrue(res.toLowerCase().contains("red"), res);
    }

    @Test
    void textOnlyStillWorks() {
        var r = model.chat(List.of(Message.user("Reply with exactly the word: kiln")), GREEDY);
        assertTrue(r.text().toLowerCase().contains("kiln"), r.text());
    }

    @Test
    void badImageIndexIsAnError() {
        String req = """
            {"messages": [{"role": "user", "content": [{"type": "image", "index": 3}, {"type": "text", "text": "hi"}]}]}""";
        var e = assertThrows(IllegalStateException.class, () -> model.chatJson(req, List.of(Image.of(redSquare())), null));
        assertTrue(e.getMessage().contains("out of range"), e.getMessage());
    }
}
