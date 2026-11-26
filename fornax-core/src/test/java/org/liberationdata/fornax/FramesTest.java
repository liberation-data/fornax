package org.liberationdata.fornax;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.StandardProtocolFamily;
import java.net.UnixDomainSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Frames a reader can't trust: truncated, or with length fields that are negative or far bigger than the frame. Each
 * must fail the read with an IOException (which ends the connection and fails the calls in flight), never an
 * unchecked exception or an allocation sized by the bad field, either of which would kill the reader thread and leave
 * the callers waiting forever.
 */
class FramesTest {
    Path dir;
    SocketChannel sender, receiver;

    @BeforeEach
    void connect() throws IOException {
        dir = Files.createTempDirectory("fornax-frames-test-");
        try (ServerSocketChannel server = ServerSocketChannel.open(StandardProtocolFamily.UNIX)) {
            var addr = UnixDomainSocketAddress.of(dir.resolve("s"));
            server.bind(addr);
            sender = SocketChannel.open(addr);
            receiver = server.accept();
        }
    }

    @AfterEach
    void close() throws IOException {
        sender.close();
        receiver.close();
        try (Stream<Path> s = Files.walk(dir)) {
            for (Path p : s.sorted(Comparator.reverseOrder()).toList()) Files.deleteIfExists(p);
        }
    }

    /** A frame whose body is {@code body}, with a correct outer length. */
    void send(byte[] body) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(4 + body.length).putInt(body.length).put(body).flip();
        while (b.hasRemaining()) sender.write(b);
    }

    static Wire.Out head() { return new Wire.Out().integer(Frames.CALL).longValue(1).longValue(2).string("m"); }

    IOException readFails() {
        try (var reader = new Frames.Reader(receiver, dir)) {
            return assertThrows(IOException.class, reader::read);
        }
    }

    @Test
    void aGoodFrameReadsBack() throws IOException {
        try (var w = new Frames.Writer(sender, dir, 1 << 20, "p"); var r = new Frames.Reader(receiver, dir)) {
            w.send(Frames.CALL, 7, 8, "echo", new byte[]{1, 2, 3});
            Frames.Frame f = r.read();
            assertEquals(7, f.id());
            assertEquals("echo", f.name());
            assertArrayEquals(new byte[]{1, 2, 3}, f.payload());
        }
    }

    @Test
    void aTruncatedHeader() throws IOException {
        send(new byte[]{0, 0, 0});
        readFails();
    }

    @Test
    void aStringLengthFarBeyondTheFrame() throws IOException {
        send(new Wire.Out().integer(Frames.CALL).longValue(1).longValue(2).integer(Integer.MAX_VALUE - 8).toBytes());
        readFails();
    }

    @Test
    void aNegativeSlotSize() throws IOException {
        Files.write(dir.resolve("p1"), new byte[1 << 16]);
        send(head().integer(1).string("p1").integer(-1).toBytes());
        readFails();
    }

    @Test
    void aNegativeFileSize() throws IOException {
        Files.write(dir.resolve("px1"), new byte[16]);
        send(head().integer(2).string("px1").integer(-5).toBytes());
        readFails();
    }

    @Test
    void wireRejectsALengthBeyondItsInput() {
        byte[] b = new Wire.Out().integer(Integer.MAX_VALUE).toBytes();
        assertThrows(UncheckedIOException.class, () -> new Wire.In(b).bytes());
        assertThrows(UncheckedIOException.class, () -> new Wire.In(new byte[]{0, 0, 0, 5, 'a'}).string());
        assertNull(new Wire.In(new Wire.Out().bytes(null).toBytes()).bytes());
    }
}
