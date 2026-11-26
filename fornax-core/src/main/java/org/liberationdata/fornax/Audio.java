package org.liberationdata.fornax;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;

/** Stereo float PCM in [-1, 1]. */
public record Audio(float[] left, float[] right, int sampleRate) {
    public double seconds() { return left.length / (double) sampleRate; }

    /** Mono audio: the same samples on both sides. */
    public static Audio ofMono(float[] samples, int sampleRate) { return new Audio(samples, samples.clone(), sampleRate); }

    /**
     * {@code seconds} of this audio from {@code start} (both in seconds), padded with silence where it runs past the
     * end: always {@code ceil(seconds * sampleRate)} samples.
     */
    public Audio section(double start, double seconds) {
        if (start < 0 || seconds <= 0 || !Double.isFinite(start) || !Double.isFinite(seconds))
            throw new IllegalArgumentException("section needs start >= 0 and seconds > 0, got " + start + ", " + seconds);
        long from = Math.round(start * sampleRate);
        int n = (int) Math.ceil(seconds * sampleRate - 1e-9);
        float[] l = new float[n], r = new float[n];
        int copy = (int) Math.max(0, Math.min(n, left.length - from));
        if (copy > 0) {
            System.arraycopy(left, (int) from, l, 0, copy);
            System.arraycopy(right, (int) from, r, 0, copy);
        }
        return new Audio(l, r, sampleRate);
    }

    /** The two sides mixed down to one channel. */
    public float[] mono() {
        float[] m = new float[left.length];
        for (int i = 0; i < m.length; i++) m[i] = (left[i] + right[i]) * 0.5f;
        return m;
    }

    /** This audio at another sample rate (band-limited, so downsampling doesn't alias). */
    public Audio resample(int rate) {
        if (rate == sampleRate) return this;
        return new Audio(resample(left, sampleRate, rate), resample(right, sampleRate, rate), rate);
    }

    /**
     * Resamples one channel with a Hann-windowed sinc filter, cut off just below the lower of the two Nyquist
     * frequencies. Good enough for speech and music going into a model; not a mastering-grade converter.
     */
    public static float[] resample(float[] x, int from, int to) {
        if (from <= 0 || to <= 0) throw new IllegalArgumentException("sample rates must be positive: " + from + " -> " + to);
        if (from == to) return x.clone();
        // Output sample i sits at input position i * m / l. Its fractional part takes only l values (the phases), so
        // the filter is computed once per phase.
        int g = gcd(from, to), l = to / g, m = from / g;
        int n = (int) ((long) x.length * l / m);
        double cutoff = 0.95 * Math.min(1.0, (double) l / m);  // of the input's Nyquist frequency
        double half = 16 / cutoff;                             // 16 sinc zero crossings each side, in input samples
        int taps = (int) Math.ceil(half), width = 2 * taps + 1;
        float[][] kernel = new float[l][width];
        for (int p = 0; p < l; p++) {
            double frac = p / (double) l, sum = 0;
            for (int k = 0; k < width; k++) {
                double d = k - taps - frac;                    // input sample minus output position
                double v = 0;
                if (Math.abs(d) < half) {
                    double t = d * cutoff;
                    v = (t == 0 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t)) * (0.5 + 0.5 * Math.cos(Math.PI * d / half));
                }
                kernel[p][k] = (float) v;
                sum += v;
            }
            for (int k = 0; k < width; k++) kernel[p][k] /= (float) sum;   // unity gain at DC
        }
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            long pos = (long) i * m;
            int base = (int) (pos / l) - taps;
            float[] kp = kernel[(int) (pos % l)];
            double acc = 0;
            int k0 = Math.max(0, -base), k1 = Math.min(width, x.length - base);
            for (int k = k0; k < k1; k++) acc += x[base + k] * kp[k];
            y[i] = (float) acc;
        }
        return y;
    }

    private static int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }

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

    /**
     * Reads a WAV file: 8, 16, 24 or 32-bit integer PCM or 32/64-bit float, any number of channels (a mono file plays
     * on both sides; beyond two, the first two are kept).
     */
    public static Audio readWav(File f) throws IOException {
        return readWav(Files.readAllBytes(f.toPath()));
    }

    /** Reads a WAV file's bytes; see {@link #readWav(File)}. A file cut short is an IOException too. */
    public static Audio readWav(byte[] bytes) throws IOException {
        try {
            return parseWav(bytes);
        } catch (IndexOutOfBoundsException e) {
            throw new IOException("truncated WAV file", e);
        }
    }

    private static Audio parseWav(byte[] bytes) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        if (bytes.length < 12 || b.getInt(0) != 0x46464952 || b.getInt(8) != 0x45564157) throw new IOException("not a WAV file");
        int format = 0, channels = 0, rate = 0, bits = 0, pos = 12;
        while (pos + 8 <= bytes.length) {
            int id = b.getInt(pos), size = b.getInt(pos + 4), body = pos + 8;
            if (id == 0x20746d66) {                              // "fmt "
                format = b.getShort(body) & 0xffff;
                channels = b.getShort(body + 2) & 0xffff;
                rate = b.getInt(body + 4);
                bits = b.getShort(body + 14) & 0xffff;
                if (format == 0xfffe && size >= 26) format = b.getShort(body + 24) & 0xffff;  // WAVE_FORMAT_EXTENSIBLE
            } else if (id == 0x61746164) {                       // "data"
                if (channels == 0) throw new IOException("WAV data before its format chunk");
                int len = Math.min(size < 0 ? Integer.MAX_VALUE : size, bytes.length - body);
                return decode(b, body, len, format, channels, rate, bits);
            }
            if (size < 0) break;
            long next = (long) body + size + (size & 1);
            if (next > bytes.length) break;
            pos = (int) next;
        }
        throw new IOException("WAV file has no data chunk");
    }

    private static Audio decode(ByteBuffer b, int off, int len, int format, int channels, int rate, int bits) throws IOException {
        boolean isFloat = format == 3;
        if (!(format == 1 || isFloat) || (isFloat && bits != 32 && bits != 64) || (!isFloat && (bits < 8 || bits > 32 || bits % 8 != 0)))
            throw new IOException("unsupported WAV encoding: format " + format + ", " + bits + " bits");
        int bytesPer = bits / 8, frame = bytesPer * channels, n = len / frame;
        float[] l = new float[n], r = new float[n];
        for (int i = 0; i < n; i++) {
            int p = off + i * frame;
            l[i] = sample(b, p, bits, isFloat);
            r[i] = channels > 1 ? sample(b, p + bytesPer, bits, isFloat) : l[i];
        }
        return new Audio(l, r, rate);
    }

    private static float sample(ByteBuffer b, int p, int bits, boolean isFloat) {
        if (isFloat) return bits == 32 ? b.getFloat(p) : (float) b.getDouble(p);
        return switch (bits) {
            case 8 -> ((b.get(p) & 0xff) - 128) / 128f;
            case 16 -> b.getShort(p) / 32768f;
            case 24 -> ((b.get(p) & 0xff) | (b.get(p + 1) & 0xff) << 8 | b.get(p + 2) << 16) / 8388608f;
            default -> b.getInt(p) / 2147483648f;
        };
    }

    private static short pcm16(float v) { return (short) Math.round(Math.max(-1f, Math.min(1f, v)) * 32767f); }
}
