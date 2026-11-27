package org.liberationdata.fornax.server;

import java.util.Map;

/**
 * A failed request, as OpenAI's API reports one: an HTTP status and
 * {@code {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}}.
 */
final class ApiError extends RuntimeException {
    private static final long serialVersionUID = 1L;

    final int status;
    final String type, param, code;

    /** The longest message an error carries: one that quotes what was sent (a model's name, a bad number) is cut short. */
    static final int MAX_MESSAGE = 1000;

    ApiError(int status, String type, String message, String param, String code) {
        super(shorten(message));
        this.status = status;
        this.type = type;
        this.param = param;
        this.code = code;
    }

    private static String shorten(String m) {
        if (m == null || m.length() <= MAX_MESSAGE) return m;
        int end = Character.isHighSurrogate(m.charAt(MAX_MESSAGE - 1)) ? MAX_MESSAGE - 1 : MAX_MESSAGE;   // never half a pair
        return m.substring(0, end) + "...";
    }

    static ApiError badRequest(String message, String param) {
        return new ApiError(400, "invalid_request_error", message, param, null);
    }

    static ApiError noSuchModel(String message) {
        return new ApiError(404, "invalid_request_error", message, "model", "model_not_found");
    }

    /**
     * A 500. Its {@link #body} cuts each local path in the message to its file name ({@link #withoutPaths}); its
     * message keeps them, for whoever runs the server ({@link FornaxServer#start} fails with it).
     */
    static ApiError server(String message) { return new ApiError(500, "server_error", message, null, null); }

    /**
     * A 500 for an unexpected exception: its message (or its class, without one), whose body has each local path cut
     * to its file name, since engine errors name model and cache files and a caller has no business knowing the
     * directories.
     */
    static ApiError server(Throwable e) {
        return server(e.getMessage() != null ? e.getMessage() : e.toString());
    }

    /**
     * Unix (/a/b/file) and Windows (C:\a\file, C:/a/file) paths with a directory, as their last segment. A directory
     * may have spaces in it (C:\Users\John Smith, Application Support), as long as no word of it has a colon, comma
     * or semicolon, so prose after a file name ("model.gguf: bad tensor a/b") isn't taken for one. The quantifiers are
     * possessive (nothing needs to backtrack: a directory ends at a slash, a word at a space) so that a long message
     * doesn't overflow the stack, as a greedy group repeated once per word or directory does.
     */
    static String withoutPaths(String message) {
        return PATH.matcher(message).replaceAll("$1");
    }

    private static final java.util.regex.Pattern PATH = java.util.regex.Pattern.compile(
            "(?:(?<![\\w.:/])/|\\b[A-Za-z]:[/\\\\])"
            + "(?:(?:[^\\s/\\\\'\"()]++|[^\\s/\\\\'\"():,;]++(?: [^\\s/\\\\'\"():,;]++)++)[/\\\\])++"
            + "([^\\s/\\\\'\"()]*+)");

    /** What the caller is sent: a 500's message has its local paths cut ({@link #withoutPaths}). */
    Map<String, Object> body() {
        String message = status == 500 && getMessage() != null ? withoutPaths(getMessage()) : getMessage();
        return Json.obj("error", Json.obj("message", message, "type", type, "param", param, "code", code));
    }
}
