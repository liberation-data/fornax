package org.liberationdata.fornax.server;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

class MultipartTest {
    @Test void fieldsAndAFile() {
        byte[] file = {0, 13, 10, 45, 45, (byte) 0xff};   // binary, with CR LF and "--" in it
        String head = "--XyZ\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-base.en\r\n"
                + "--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"talk.wav\"\r\nContent-Type: audio/wav\r\n\r\n";
        byte[] tail = "\r\n--XyZ--\r\n".getBytes(StandardCharsets.UTF_8);
        byte[] body = new byte[head.length() + file.length + tail.length];
        System.arraycopy(head.getBytes(StandardCharsets.UTF_8), 0, body, 0, head.length());
        System.arraycopy(file, 0, body, head.length(), file.length);
        System.arraycopy(tail, 0, body, head.length() + file.length, tail.length);
        var parts = Multipart.parse("multipart/form-data; boundary=XyZ", body);
        assertEquals("whisper-base.en", parts.get("model").text());
        assertNull(parts.get("model").fileName());
        assertEquals("talk.wav", parts.get("file").fileName());
        assertArrayEquals(file, parts.get("file").bytes());
        assertArrayEquals(file, Multipart.parse("multipart/form-data; boundary=\"XyZ\"", body).get("file").bytes(), "a quoted boundary");
    }

    @Test void mistakes() {
        byte[] b = "x".getBytes(StandardCharsets.UTF_8);
        assertThrows(ApiError.class, () -> Multipart.parse("application/json", b));
        assertThrows(ApiError.class, () -> Multipart.parse("multipart/form-data", b));
        assertThrows(ApiError.class, () -> Multipart.parse("multipart/form-data; boundary=q", b));
        assertThrows(ApiError.class, () -> Multipart.parse("multipart/form-data; boundary=q",
                "--q\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\nunterminated".getBytes(StandardCharsets.UTF_8)));
    }

    /**
     * The boundary is the header's, so its length is the client's choice: a long one of dashes, over a body of dashes,
     * made each search for it cost its length at every byte (20000 dashes over 1 MB took 7 s, a 64 MB body hours).
     */
    @Test void aBoundaryLongerThanRfc2046sIsRefusedAtOnce() {
        String longest = "a".repeat(70);
        assertEquals("x", Multipart.parse("multipart/form-data; boundary=" + longest,
                ("--" + longest + "\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\nx\r\n--" + longest + "--\r\n")
                        .getBytes(StandardCharsets.UTF_8)).get("a").text(), "70 characters, RFC 2046's most, is taken");

        byte[] dashes = new byte[1 << 20];
        Arrays.fill(dashes, (byte) '-');
        for (int i = 20_001; i < dashes.length; i += 20_002) dashes[i] = 'x';
        var e = assertTimeoutPreemptively(Duration.ofSeconds(2), () -> assertThrows(ApiError.class,
                () -> Multipart.parse("multipart/form-data; boundary=" + "-".repeat(20_000), dashes)));
        assertEquals(400, e.status);
        assertTrue(e.getMessage().contains("70"), e.getMessage());
        assertThrows(ApiError.class, () -> Multipart.parse("multipart/form-data; boundary=" + "a".repeat(71), dashes));
    }
}
