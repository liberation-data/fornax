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

    @Test void aStringCutShortIsNotRead() throws Exception {
        byte[] whole = gguf("general.architecture", "llama", "general.name", "Llama 3");
        assertEquals(Map.of("general.architecture", "llama"), GgufHeader.read(write("cut.gguf", Arrays.copyOf(whole, whole.length - 2))));
    }

    @Test void readsEveryScalarTypeAndStopsAtAnUnknownOne() throws Exception {
        var b = ByteBuffer.allocate(512).order(ByteOrder.LITTLE_ENDIAN);
        b.put("GGUF".getBytes(StandardCharsets.US_ASCII)).putInt(3).putLong(0).putLong(13);
        b.putLong(2).put("u8".getBytes()).putInt(0).put((byte) 0xff);
        b.putLong(2).put("i8".getBytes()).putInt(1).put((byte) -1);
        b.putLong(3).put("u16".getBytes()).putInt(2).putShort((short) 0xffff);
        b.putLong(3).put("i16".getBytes()).putInt(3).putShort((short) -1);
        b.putLong(3).put("u32".getBytes()).putInt(4).putInt(-1);
        b.putLong(3).put("i32".getBytes()).putInt(5).putInt(-1);
        b.putLong(3).put("f32".getBytes()).putInt(6).putFloat(1.5f);
        b.putLong(4).put("bool".getBytes()).putInt(7).put((byte) 1);
        b.putLong(3).put("u64".getBytes()).putInt(10).putLong(1L << 40);
        b.putLong(3).put("i64".getBytes()).putInt(11).putLong(-1);
        b.putLong(3).put("f64".getBytes()).putInt(12).putDouble(2.5);
        b.putLong(1).put("x".getBytes()).putInt(13).put((byte) 0);
        b.putLong(5).put("after".getBytes()).putInt(7).put((byte) 1);
        assertEquals(Map.ofEntries(Map.entry("u8", 255), Map.entry("i8", -1), Map.entry("u16", 65535), Map.entry("i16", -1),
                        Map.entry("u32", 4294967295L), Map.entry("i32", -1), Map.entry("f32", 1.5f), Map.entry("bool", true),
                        Map.entry("u64", 1L << 40), Map.entry("i64", -1L), Map.entry("f64", 2.5)),
                GgufHeader.read(write("types.gguf", Arrays.copyOf(b.array(), b.position()))));
    }

    @Test void anArrayLongerThanTheFileStopsTheRead() throws Exception {
        // Skipping past the end of the file ends the read rather than spinning on a skip that makes no progress:
        // an array cut short, and one claiming 2^40 floats.
        byte[] whole = gguf("general.architecture", "llama", "scores", new float[] {1f, 2f});
        assertEquals(Map.of("general.architecture", "llama"), GgufHeader.read(write("cut.gguf", Arrays.copyOf(whole, whole.length - 1))));
        ByteBuffer.wrap(whole).order(ByteOrder.LITTLE_ENDIAN).putLong(whole.length - 16, 1L << 40);
        assertEquals(Map.of("general.architecture", "llama"), GgufHeader.read(write("long.gguf", whole)));
    }

    @Test void anAbsurdStringLengthStopsTheRead() throws Exception {
        byte[] b = gguf("k", "v");
        // The key's length (bytes 24..31) claims 2^40 bytes.
        ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).putLong(24, 1L << 40);
        assertEquals(Map.of(), GgufHeader.read(write("huge.gguf", b)));
    }

    @Test void aHostileArrayStopsTheReadWithoutThrowing() throws Exception {
        // An array claiming 2^62 u64s (its byte count overflows a long) or 2^63+ elements (negative as a long).
        // An array of an element type GGUF doesn't have.
        var bad = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
        bad.put("GGUF".getBytes(StandardCharsets.US_ASCII)).putInt(3).putLong(0).putLong(2);
        bad.putLong(1).put((byte) 'a').putInt(9).putInt(13).putLong(1).put((byte) 0);
        bad.putLong(1).put((byte) 'b').putInt(7).put((byte) 1);
        assertEquals(Map.of(), GgufHeader.read(write("bad.gguf", Arrays.copyOf(bad.array(), bad.position()))));
        for (long n : new long[] {1L << 62, -1L}) {
            var b = ByteBuffer.allocate(64).order(ByteOrder.LITTLE_ENDIAN);
            b.put("GGUF".getBytes(StandardCharsets.US_ASCII)).putInt(3).putLong(0).putLong(2);
            b.putLong(1).put((byte) 'a').putInt(9).putInt(10).putLong(n);
            b.putLong(1).put((byte) 'b').putInt(7).put((byte) 1);
            assertEquals(Map.of(), GgufHeader.read(write("array.gguf", Arrays.copyOf(b.array(), b.position()))));
        }
        // Arrays nested a million deep (llama.cpp rejects any nesting) would overflow the stack.
        var deep = ByteBuffer.allocate(40 + 12 * 1_000_000).order(ByteOrder.LITTLE_ENDIAN);
        deep.put("GGUF".getBytes(StandardCharsets.US_ASCII)).putInt(3).putLong(0).putLong(1).putLong(1).put((byte) 'a').putInt(9);
        while (deep.remaining() >= 12) deep.putInt(9).putLong(1);
        assertEquals(Map.of(), GgufHeader.read(write("deep.gguf", deep.array())));
    }
}
