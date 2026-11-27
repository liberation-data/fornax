package org.liberationdata.fornax.server;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;

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
}
