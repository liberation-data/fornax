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

    static ApiError server(String message) { return new ApiError(500, "server_error", message, null, null); }

    Map<String, Object> body() {
        return Json.obj("error", Json.obj("message", getMessage(), "type", type, "param", param, "code", code));
    }
}
