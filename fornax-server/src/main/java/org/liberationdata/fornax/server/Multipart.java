package org.liberationdata.fornax.server;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A {@code multipart/form-data} body, as {@code POST /v1/audio/transcriptions} takes its file: each part by its field
 * name, with its bytes and its file name (null for a plain field). Enough of RFC 7578 for what clients send.
 */
final class Multipart {
    private Multipart() {}

    record Part(String name, String fileName, byte[] bytes) {
        String text() { return new String(bytes, StandardCharsets.UTF_8); }
    }

    private static final Pattern BOUNDARY = Pattern.compile("(?i)boundary=(\"([^\"]+)\"|[^;\\s]+)");
    private static final Pattern NAME = Pattern.compile("(?i)(?:^|;)\\s*name=\"([^\"]*)\"");
    private static final Pattern FILE_NAME = Pattern.compile("(?i)(?:^|;)\\s*filename=\"([^\"]*)\"");

    /** The parts of body, by name (a name given twice keeps the last; {@code name[]} fields keep their brackets). */
    static Map<String, Part> parse(String contentType, byte[] body) {
        if (contentType == null || !contentType.toLowerCase(Locale.ROOT).startsWith("multipart/form-data"))
            throw ApiError.badRequest("the body must be multipart/form-data", null);
        Matcher bm = BOUNDARY.matcher(contentType);
        if (!bm.find()) throw ApiError.badRequest("multipart/form-data without a boundary", null);
        byte[] delimiter = ("--" + (bm.group(2) != null ? bm.group(2) : bm.group(1))).getBytes(StandardCharsets.ISO_8859_1);
        Map<String, Part> parts = new LinkedHashMap<>();
        int at = indexOf(body, delimiter, 0);
        if (at < 0) throw ApiError.badRequest("multipart body without its boundary", null);
        while (true) {
            int start = at + delimiter.length;
            if (start + 1 < body.length && body[start] == '-' && body[start + 1] == '-') break;   // the closing delimiter
            start = skipLine(body, start);
            int headersEnd = indexOf(body, "\r\n\r\n".getBytes(StandardCharsets.ISO_8859_1), start);
            if (headersEnd < 0) throw ApiError.badRequest("a multipart part without its headers' end", null);
            String headers = new String(body, start, headersEnd - start, StandardCharsets.UTF_8);
            int dataStart = headersEnd + 4;
            byte[] next = ("\r\n" + new String(delimiter, StandardCharsets.ISO_8859_1)).getBytes(StandardCharsets.ISO_8859_1);
            int end = indexOf(body, next, dataStart);
            if (end < 0) throw ApiError.badRequest("a multipart part without its boundary after it", null);
            String disposition = null;
            for (String line : headers.split("\r\n"))
                if (line.regionMatches(true, 0, "Content-Disposition:", 0, 20)) disposition = line.substring(20).strip();
            if (disposition != null) {
                Matcher n = NAME.matcher(disposition), f = FILE_NAME.matcher(disposition);
                if (n.find()) parts.put(n.group(1), new Part(n.group(1), f.find() ? f.group(1) : null, Arrays.copyOfRange(body, dataStart, end)));
            }
            at = end + 2;
        }
        return parts;
    }

    private static int skipLine(byte[] b, int i) {
        while (i < b.length && b[i] != '\n') i++;
        return i + 1;
    }

    static int indexOf(byte[] hay, byte[] needle, int from) {
        outer:
        for (int i = Math.max(0, from); i <= hay.length - needle.length; i++) {
            for (int j = 0; j < needle.length; j++) if (hay[i + j] != needle[j]) continue outer;
            return i;
        }
        return -1;
    }
}
