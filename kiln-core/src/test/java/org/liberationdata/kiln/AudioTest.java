package org.liberationdata.kiln;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class AudioTest {
    static float[] sine(double hz, int rate, double seconds) {
        float[] x = new float[(int) (rate * seconds)];
        for (int i = 0; i < x.length; i++) x[i] = (float) (0.5 * Math.sin(2 * Math.PI * hz * i / rate));
        return x;
    }

    static double rms(float[] x, int from, int to) {
        double s = 0;
        for (int i = from; i < to; i++) s += x[i] * x[i];
        return Math.sqrt(s / (to - from));
    }

    @Test
    void resamplingKeepsATone() {
        float[] x = sine(440, 24000, 1.0);
        float[] y = Audio.resample(x, 24000, 16000);
        assertEquals(16000, y.length);
        float[] expected = sine(440, 16000, 1.0);
        double err = 0;
        for (int i = 200; i < y.length - 200; i++) err = Math.max(err, Math.abs(y[i] - expected[i]));
        assertTrue(err < 0.01, "max error " + err);
    }

    @Test
    void downsamplingRemovesWhatTheNewRateCantHold() {
        float[] x = sine(10000, 44100, 0.5);                   // above 16 kHz's 8 kHz Nyquist
        float[] y = Audio.resample(x, 44100, 16000);
        assertTrue(rms(y, 100, y.length - 100) < 0.01, "aliased: rms " + rms(y, 100, y.length - 100));
    }

    @Test
    void upsamplingKeepsTheLength() {
        assertEquals(48000, Audio.resample(sine(100, 16000, 1.0), 16000, 48000).length);
        Audio a = Audio.ofMono(sine(100, 16000, 1.0), 16000);
        assertSame(a, a.resample(16000));
        assertEquals(24000, a.resample(24000).sampleRate());
    }

    @Test
    void mixesDownToMono() {
        Audio a = new Audio(new float[] {1f, 0f}, new float[] {0f, -1f}, 8000);
        assertArrayEquals(new float[] {0.5f, -0.5f}, a.mono());
    }

    @Test
    void writesAndReadsAWav(@TempDir Path dir) throws IOException {
        Audio a = new Audio(sine(300, 22050, 0.1), sine(600, 22050, 0.1), 22050);
        File f = dir.resolve("a.wav").toFile();
        a.writeWav(f);
        Audio b = Audio.readWav(f);
        assertEquals(22050, b.sampleRate());
        assertEquals(a.left().length, b.left().length);
        for (int i = 0; i < a.left().length; i++) {
            assertEquals(a.left()[i], b.left()[i], 1e-4);
            assertEquals(a.right()[i], b.right()[i], 1e-4);
        }
    }

    /** A mono 32-bit float WAV with an extra chunk before the data, the way many tools write them. */
    @Test
    void readsMonoFloatWithOtherChunks() throws IOException {
        float[] x = {0.25f, -0.5f, 1f};
        ByteBuffer b = ByteBuffer.allocate(12 + 8 + 16 + 8 + 4 + 8 + x.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        b.put("RIFF".getBytes()).putInt(b.capacity() - 8).put("WAVE".getBytes())
         .put("fmt ".getBytes()).putInt(16).putShort((short) 3).putShort((short) 1).putInt(16000).putInt(64000)
         .putShort((short) 4).putShort((short) 32)
         .put("LIST".getBytes()).putInt(4).put("INFO".getBytes())
         .put("data".getBytes()).putInt(x.length * 4);
        for (float v : x) b.putFloat(v);
        Audio a = Audio.readWav(b.array());
        assertEquals(16000, a.sampleRate());
        assertArrayEquals(x, a.left());
        assertArrayEquals(x, a.right());
    }

    @Test
    void reads24BitPcm() throws IOException {
        ByteBuffer b = ByteBuffer.allocate(44 + 6).order(ByteOrder.LITTLE_ENDIAN);
        b.put("RIFF".getBytes()).putInt(42).put("WAVE".getBytes())
         .put("fmt ".getBytes()).putInt(16).putShort((short) 1).putShort((short) 1).putInt(48000).putInt(144000)
         .putShort((short) 3).putShort((short) 24)
         .put("data".getBytes()).putInt(6)
         .put((byte) 0).put((byte) 0).put((byte) 0x40)          // +0.5
         .put((byte) 0).put((byte) 0).put((byte) 0xc0);         // -0.5
        assertArrayEquals(new float[] {0.5f, -0.5f}, Audio.readWav(b.array()).left());
    }

    @Test
    void rejectsWhatIsntAWav() {
        assertThrows(IOException.class, () -> Audio.readWav("not a wav file at all".getBytes()));
    }
}
