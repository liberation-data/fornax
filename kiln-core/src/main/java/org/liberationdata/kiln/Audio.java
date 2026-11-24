package org.liberationdata.kiln;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Stereo float PCM in [-1, 1]. */
public record Audio(float[] left, float[] right, int sampleRate) {
    public double seconds() { return left.length / (double) sampleRate; }

    public void writeWav(File f) throws IOException {
        int n = left.length;
        ByteBuffer b = ByteBuffer.allocate(44 + n * 4).order(ByteOrder.LITTLE_ENDIAN);
        b.put("RIFF".getBytes()).putInt(36 + n * 4).put("WAVE".getBytes())
         .put("fmt ".getBytes()).putInt(16).putShort((short) 1).putShort((short) 2)
         .putInt(sampleRate).putInt(sampleRate * 4).putShort((short) 4).putShort((short) 16)
         .put("data".getBytes()).putInt(n * 4);
        for (int i = 0; i < n; i++) { b.putShort(pcm16(left[i])); b.putShort(pcm16(right[i])); }
        try (var out = new FileOutputStream(f)) { out.write(b.array()); }
    }

    private static short pcm16(float v) { return (short) Math.round(Math.max(-1f, Math.min(1f, v)) * 32767f); }
}
