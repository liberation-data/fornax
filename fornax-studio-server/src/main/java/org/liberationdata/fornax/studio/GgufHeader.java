package org.liberationdata.fornax.studio;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * A GGUF file's metadata, read from its header without loading the model: the scalar and string values by key
 * (arrays, such as a tokenizer's vocabulary, are skipped). Enough to tell what a file is, as {@link #onlySpeech} does.
 */
final class GgufHeader {
    private GgufHeader() {}

    // GGUF value types (gguf.h): their sizes in bytes, or 0 for a string (8) or an array (9).
    private static final int STRING = 8, ARRAY = 9;
    private static final int[] SIZE = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    // A key or string longer than this is a corrupt or hostile header, not metadata.
    private static final long MAX_STRING = 1 << 20;

    /** The file's scalar and string metadata by key: none if it isn't a GGUF, what comes before any damage if it is. */
    static Map<String, Object> read(Path file) {
        Map<String, Object> out = new LinkedHashMap<>();
        try (var in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file), 1 << 16))) {
            byte[] magic = in.readNBytes(4);
            if (!new String(magic, StandardCharsets.US_ASCII).equals("GGUF")) return out;
            in.readInt();                       // version
            readU64(in);                        // tensor count
            long keys = readU64(in);
            for (long k = 0; k < keys; k++) {
                String key = readString(in);
                int type = Integer.reverseBytes(in.readInt());
                Object v = readValue(in, type);
                if (v != null) out.put(key, v);
            }
        } catch (IOException | IllegalArgumentException | ArithmeticException e) {
            // A truncated, corrupt or unreadable file: whatever was read before it.
        }
        return out;
    }

    // Projectors of models that speak aloud and also chat in text (LFM3-Audio): Chat keeps them.
    private static final Set<String> CHATTY_VOICES = Set.of("lfm3a_gen");
    // Projectors of models that hear audio only to transcribe it (Qwen3-ASR, GLM-ASR, Voxtral Realtime, Nemotron
    // Transcribe, VibeVoice-ASR): Chat leaves them out. Qwen3-Omni hears with qwen3a too, but it sees as well, and chats.
    private static final Set<String> TRANSCRIBERS = Set.of("qwen3a", "glma", "voxtral_rt", "nemotron_transcribe", "vibevoice_asr");

    /**
     * Whether a model's projector ({@code mmproj-*.gguf}) makes it a speech model and not one to chat with: text to
     * speech, voice cloning or speech recognition. Those share llama.cpp architectures with chat models (Soprano is
     * qwen3), so the projector is what tells them apart. A model that does both, as audio chat models do, chats: it
     * belongs in Chat as well as wherever its audio is used.
     *
     * <p>A projector that makes audio is a voice unless it's one known to chat as well; one that only hears audio is
     * a chat model (Ultravox, Voxtral, Qwen2.5-Omni) unless it's a known transcriber. So an unknown voice stays out of
     * Chat, where it couldn't answer, and an unknown listener stays in. One that sees as well as hears (an omni model)
     * chats. A projector with audio alone names its type {@code clip.projector_type}; one with several modalities names
     * each, as {@code clip.audio.projector_type}.
     */
    static boolean onlySpeech(Path mmproj) {
        Map<String, Object> h = read(mmproj);
        if (Boolean.TRUE.equals(h.get("clip.has_gen_audio_encoder")))
            return !CHATTY_VOICES.contains(String.valueOf(h.get("clip.gen.audio.projector_type")));
        Object hears = h.getOrDefault("clip.audio.projector_type", h.get("clip.projector_type"));
        return Boolean.TRUE.equals(h.get("clip.has_audio_encoder")) && !Boolean.TRUE.equals(h.get("clip.has_vision_encoder"))
                && TRANSCRIBERS.contains(String.valueOf(hears));
    }

    private static Object readValue(DataInputStream in, int type) throws IOException {
        return switch (type) {
            case 0 -> in.readUnsignedByte();
            case 1 -> (int) in.readByte();
            case 2 -> (int) Short.reverseBytes(in.readShort()) & 0xffff;
            case 3 -> (int) Short.reverseBytes(in.readShort());
            case 4 -> Integer.toUnsignedLong(Integer.reverseBytes(in.readInt()));
            case 5 -> Integer.reverseBytes(in.readInt());
            case 6 -> Float.intBitsToFloat(Integer.reverseBytes(in.readInt()));
            case 7 -> in.readUnsignedByte() != 0;
            case STRING -> readString(in);
            case ARRAY -> { skipArray(in); yield null; }
            case 10, 11 -> readU64(in);
            case 12 -> Double.longBitsToDouble(Long.reverseBytes(in.readLong()));
            default -> throw new IllegalArgumentException("unknown GGUF value type " + type);
        };
    }

    private static void skipArray(DataInputStream in) throws IOException {
        int type = Integer.reverseBytes(in.readInt());
        long n = length(in);
        // llama.cpp rejects nested arrays too; following them would let a hostile file overflow the stack.
        if (type < 0 || type >= SIZE.length || type == ARRAY) throw new IllegalArgumentException("GGUF array of type " + type);
        if (type == STRING) for (long i = 0; i < n; i++) skip(in, length(in));
        else skip(in, Math.multiplyExact(n, SIZE[type]));
    }

    private static String readString(DataInputStream in) throws IOException {
        long n = length(in);
        if (n > MAX_STRING) throw new IllegalArgumentException("GGUF string of " + n + " bytes");
        byte[] b = new byte[(int) n];
        in.readFully(b);                    // readNBytes would hand back a string cut short by the end of the file
        return new String(b, StandardCharsets.UTF_8);
    }

    private static long length(DataInputStream in) throws IOException {
        long n = readU64(in);
        if (n < 0) throw new IllegalArgumentException("GGUF length " + Long.toUnsignedString(n));
        return n;
    }

    private static long readU64(DataInputStream in) throws IOException { return Long.reverseBytes(in.readLong()); }

    private static void skip(InputStream in, long n) throws IOException {
        while (n > 0) {
            long s = in.skip(n);
            if (s <= 0) {
                if (in.read() < 0) throw new EOFException();
                s = 1;
            }
            n -= s;
        }
    }
}
