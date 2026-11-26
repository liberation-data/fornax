package org.liberationdata.kiln.llama;

import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.Wire;
import org.liberationdata.kiln.llama.LlamaModel.Image;
import org.liberationdata.kiln.llama.LlamaModel.Message;
import org.liberationdata.kiln.llama.LlamaModel.Options;
import org.liberationdata.kiln.llama.LlamaModel.Result;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/** An isolated model's encoding of its calls, both ways, as LlamaService decodes them: no native library needed. */
class LlamaServiceTest {
    static final Image RED = new Image(1, 2, new byte[]{(byte) 255, 0, 0, (byte) 255, 0, 0});
    static final Image BLUE = new Image(2, 1, new byte[]{0, 0, (byte) 255, 0, 0, (byte) 255});
    static final Options ODD = new Options(-1, 0f, 0.5f, 0.25f, 42);

    @Test
    void chatArgumentsRoundTripWithImagesNullContentAndOptions() {
        var messages = List.of(Message.system("be brief"), Message.user("which colour? ünïcode", RED, BLUE),
                new Message("user", null, List.of(BLUE)), Message.assistant(""));
        var in = new Wire.In(LlamaService.chatArgs(messages, ODD, true));
        List<Message> back = LlamaService.messages(in);
        assertEquals(messages.size(), back.size());
        for (int i = 0; i < messages.size(); i++) {
            assertEquals(messages.get(i).role(), back.get(i).role());
            assertEquals(messages.get(i).content(), back.get(i).content());
            assertImages(messages.get(i).images(), back.get(i).images());
        }
        assertEquals(ODD, LlamaService.options(in));
        assertTrue(in.bool());
    }

    @Test
    void chatJsonAndCompleteArgumentsRoundTrip() {
        var in = new Wire.In(LlamaService.chatJsonArgs("{\"messages\": []}", List.of(RED), false));
        assertEquals("{\"messages\": []}", in.string());
        assertImages(List.of(RED), LlamaService.images(in));
        assertFalse(in.bool());

        in = new Wire.In(LlamaService.completeArgs("<__media__> what?", List.of(), Options.DEFAULT, true));
        assertEquals("<__media__> what?", in.string());
        assertEquals(List.of(), LlamaService.images(in));
        assertEquals(Options.DEFAULT, LlamaService.options(in));
        assertTrue(in.bool());
    }

    @Test
    void resultsRoundTrip() {
        var r = new Result("forty-two ✓", 17, 3);
        assertEquals(r, LlamaService.decodeResult(LlamaService.result(r)));
        assertEquals("", LlamaService.decodeResult(LlamaService.result(new Result("", 0, 0))).text());
    }

    static void assertImages(List<Image> expected, List<Image> actual) {
        assertEquals(expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++) {
            assertEquals(expected.get(i).width(), actual.get(i).width());
            assertEquals(expected.get(i).height(), actual.get(i).height());
            assertArrayEquals(expected.get(i).rgb(), actual.get(i).rgb());
        }
    }
}
