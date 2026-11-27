package org.liberationdata.fornax.studio;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class GgufHeaderTest {
    @TempDir Path dir;

    /** A GGUF header (no tensors) with the given metadata, as llama.cpp's gguf writer lays it out. */
    static byte[] gguf(Object... keysAndValues) {
        var out = new ByteArrayOutputStream();
        out.writeBytes("GGUF".getBytes(StandardCharsets.US_ASCII));
        out.writeBytes(le(4).putInt(3).array());
        out.writeBytes(le(8).putLong(0).array());
        out.writeBytes(le(8).putLong(keysAndValues.length / 2).array());
        for (int i = 0; i < keysAndValues.length; i += 2) {
            string(out, (String) keysAndValues[i]);
            switch (keysAndValues[i + 1]) {
                case Boolean b -> { out.writeBytes(le(4).putInt(7).array()); out.write(b ? 1 : 0); }
                case String s -> { out.writeBytes(le(4).putInt(8).array()); string(out, s); }
                case Integer n -> { out.writeBytes(le(4).putInt(5).array()); out.writeBytes(le(4).putInt(n).array()); }
                case String[] a -> {   // an array of strings, as a tokenizer's vocabulary is
                    out.writeBytes(le(4).putInt(9).array());
                    out.writeBytes(le(4).putInt(8).array());
                    out.writeBytes(le(8).putLong(a.length).array());
                    for (String s : a) string(out, s);
                }
                case float[] a -> {
                    out.writeBytes(le(4).putInt(9).array());
                    out.writeBytes(le(4).putInt(6).array());
                    out.writeBytes(le(8).putLong(a.length).array());
                    for (float f : a) out.writeBytes(le(4).putFloat(f).array());
                }
                default -> throw new IllegalArgumentException(String.valueOf(keysAndValues[i + 1]));
            }
        }
        return out.toByteArray();
    }

    private static ByteBuffer le(int n) { return ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN); }

    private static void string(ByteArrayOutputStream out, String s) {
        byte[] b = s.getBytes(StandardCharsets.UTF_8);
        out.writeBytes(le(8).putLong(b.length).array());
        out.writeBytes(b);
    }

    private Path write(String name, byte[] bytes) throws Exception {
        Path p = dir.resolve(name);
        Files.write(p, bytes);
        return p;
    }

    @Test void readsScalarsAndStringsPastArrays() throws Exception {
        Path p = write("m.gguf", gguf("tokenizer.ggml.tokens", new String[] {"a", "bc"}, "scores", new float[] {1f, 2f},
                "general.architecture", "qwen3", "qwen3.block_count", 28, "clip.has_vision_encoder", true));
        assertEquals(Map.of("general.architecture", "qwen3", "qwen3.block_count", 28, "clip.has_vision_encoder", true),
                GgufHeader.read(p));
    }

    @Test void aProjectorThatTakesOrMakesAudioSpeaksAudio() throws Exception {
        assertTrue(GgufHeader.speaksAudio(write("asr.gguf", gguf("clip.has_audio_encoder", true))));
        assertTrue(GgufHeader.speaksAudio(write("tts.gguf", gguf("clip.has_gen_audio_encoder", true))));
        assertFalse(GgufHeader.speaksAudio(write("vision.gguf", gguf("clip.has_vision_encoder", true, "clip.has_audio_encoder", false))));
    }

    @Test void notAGgufOrCutShortIsReadAsFarAsItGoes() throws Exception {
        assertEquals(Map.of(), GgufHeader.read(write("zeros.gguf", new byte[64])));
        assertEquals(Map.of(), GgufHeader.read(dir.resolve("missing.gguf")));
        byte[] whole = gguf("general.architecture", "llama", "clip.has_audio_encoder", true);
        Path cut = write("cut.gguf", Arrays.copyOf(whole, whole.length - 1));
        assertEquals(Map.of("general.architecture", "llama"), GgufHeader.read(cut));
        assertFalse(GgufHeader.speaksAudio(cut));
    }

    @Test void anAbsurdStringLengthStopsTheRead() throws Exception {
        byte[] b = gguf("k", "v");
        // The key's length (bytes 24..31) claims 2^40 bytes.
        ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).putLong(24, 1L << 40);
        assertEquals(Map.of(), GgufHeader.read(write("huge.gguf", b)));
    }
}
