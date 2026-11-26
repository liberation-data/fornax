package org.liberationdata.fornax;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

/**
 * A small binary encoding for what crosses the boundary to a {@link Worker}: a service's config, a call's arguments,
 * its result and its events. Fields are written and read back in the same order; there are no names or tags.
 *
 * <pre>{@code
 * byte[] args = new Wire.Out().string(prompt).integer(maxTokens).bytes(rgb).toBytes();
 * var in = new Wire.In(args);
 * String prompt = in.string(); int maxTokens = in.integer(); byte[] rgb = in.bytes();
 * }</pre>
 */
public final class Wire {
    /** Writes fields in order. */
    public static final class Out {
        private final ByteArrayOutputStream buf = new ByteArrayOutputStream();
        private final DataOutputStream out = new DataOutputStream(buf);

        public Out integer(int v) { try { out.writeInt(v); } catch (IOException e) { throw new UncheckedIOException(e); } return this; }

        public Out longValue(long v) { try { out.writeLong(v); } catch (IOException e) { throw new UncheckedIOException(e); } return this; }

        public Out floatValue(float v) { try { out.writeFloat(v); } catch (IOException e) { throw new UncheckedIOException(e); } return this; }

        public Out bool(boolean v) { try { out.writeBoolean(v); } catch (IOException e) { throw new UncheckedIOException(e); } return this; }

        /** A byte array, or null. */
        public Out bytes(byte[] b) {
            try {
                if (b == null) out.writeInt(-1);
                else { out.writeInt(b.length); out.write(b); }
            } catch (IOException e) { throw new UncheckedIOException(e); }
            return this;
        }

        /** A string as UTF-8 (any length), or null. */
        public Out string(String s) { return bytes(s == null ? null : s.getBytes(StandardCharsets.UTF_8)); }

        public byte[] toBytes() { return buf.toByteArray(); }
    }

    /** Reads fields back in the order they were written. */
    public static final class In {
        private final DataInputStream in;

        public In(byte[] b) { in = new DataInputStream(new ByteArrayInputStream(b)); }

        public int integer() { try { return in.readInt(); } catch (IOException e) { throw new UncheckedIOException(e); } }

        public long longValue() { try { return in.readLong(); } catch (IOException e) { throw new UncheckedIOException(e); } }

        public float floatValue() { try { return in.readFloat(); } catch (IOException e) { throw new UncheckedIOException(e); } }

        public boolean bool() { try { return in.readBoolean(); } catch (IOException e) { throw new UncheckedIOException(e); } }

        public byte[] bytes() {
            try {
                int n = in.readInt();
                if (n < 0) return null;
                if (n > in.available()) throw new java.io.EOFException(n + " bytes, with " + in.available() + " left");
                byte[] b = new byte[n];
                in.readFully(b);
                return b;
            } catch (IOException e) { throw new UncheckedIOException(e); }
        }

        public String string() {
            byte[] b = bytes();
            return b == null ? null : new String(b, StandardCharsets.UTF_8);
        }
    }

    private Wire() {}
}
