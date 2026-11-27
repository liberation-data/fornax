package org.liberationdata.fornax.server;

import com.sun.net.httpserver.Headers;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Who may call the server. Listening on loopback alone doesn't keep out the web pages open in the user's browser: a
 * page can send a "simple" POST to 127.0.0.1 with no CORS preflight, and DNS rebinding gets round the same-origin
 * check. So, as Fornax Studio's server does:
 * <ul>
 *   <li>on a loopback address, a request whose {@code Host} isn't a loopback name or address (localhost, [::1],
 *       127.x.y.z) and this server's port is
 *       refused (a rebound name); bound to another address, the server is meant to be reached by other names, and
 *       the API key is what keeps callers out</li>
 *   <li>a request with an {@code Origin} (a browser sends one with every cross-origin request, and with every POST) is
 *       refused unless the origin is one allowed with {@code --cors}; an allowed one gets CORS headers, and its
 *       preflight an answer</li>
 *   <li>with an API key, a request without {@code Authorization: Bearer <key>} is refused (compared in constant
 *       time); a preflight carries no credentials, so it needs none</li>
 * </ul>
 * The reasons are generic, and never repeat what was sent.
 */
final class Guard {
    private static final Pattern HEADER_LIST = Pattern.compile("[A-Za-z0-9-]+(\\s*,\\s*[A-Za-z0-9-]+)*");
    /** Any address in 127.0.0.0/8, all of it loopback (a name can't be rebound to a literal address). */
    private static final Pattern LOOPBACK_V4 = Pattern.compile("127(\\.(25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)){3}");

    private final boolean checkHost;
    private final Set<String> origins;   // "*" allows any
    private final byte[] apiKey;         // null for none

    /**
     * @param checkHost whether the Host must be a loopback name and the port (the server is bound to loopback)
     * @param origins   the origins a browser may call from ({@code https://app.example}), or {@code *} for any
     * @param apiKey    the key every request must carry as a bearer token, or null or blank for none
     */
    Guard(boolean checkHost, Set<String> origins, String apiKey) {
        this.checkHost = checkHost;
        this.origins = Set.copyOf(origins);
        this.apiKey = apiKey == null || apiKey.isBlank() ? null : apiKey.strip().getBytes(StandardCharsets.UTF_8);
    }

    /** Why the request is refused, or null. {@code port} is the server's own; {@code preflight}: a CORS preflight. */
    ApiError refusal(Headers h, int port, boolean preflight) {
        if (checkHost) {
            List<String> hosts = h.get("Host");
            String host = hosts == null || hosts.size() != 1 ? null : hosts.get(0).toLowerCase(Locale.ROOT);
            if (host == null || !loopbackHost(host, port))
                return forbidden("this server only answers requests to its own loopback address (wrong Host)");
        }
        if (h.containsKey("Origin") && allowedOrigin(h) == null)
            return forbidden("requests from web pages are refused unless their origin is allowed (start the server with --cors ORIGIN)");
        if (apiKey != null && !preflight) {
            String auth = h.getFirst("Authorization");
            byte[] given = auth != null && auth.regionMatches(true, 0, "Bearer ", 0, 7)
                    ? auth.substring(7).strip().getBytes(StandardCharsets.UTF_8) : new byte[0];
            if (!MessageDigest.isEqual(apiKey, given))
                return new ApiError(401, "invalid_request_error", "a valid API key is required (Authorization: Bearer <key>)",
                        null, "invalid_api_key");
        }
        return null;
    }

    /** localhost, [::1] or a 127.x.y.z address, with this server's port. */
    static boolean loopbackHost(String host, int port) {
        String suffix = ":" + port;
        if (!host.endsWith(suffix)) return false;
        String name = host.substring(0, host.length() - suffix.length());
        return name.equals("localhost") || name.equals("[::1]") || LOOPBACK_V4.matcher(name).matches();
    }

    /** The request's Origin when it may call (one Origin, allowed), else null. */
    String allowedOrigin(Headers h) {
        List<String> all = h.get("Origin");
        if (all == null || all.size() != 1) return null;
        String origin = all.get(0).strip();
        if (origin.isEmpty()) return null;
        return origins.contains("*") || origins.contains(origin) ? origin : null;
    }

    /** The CORS headers for a request from an allowed origin; for a preflight, what it may send too. */
    void cors(Headers request, Headers response, boolean preflight) {
        String origin = allowedOrigin(request);
        if (origin == null) return;
        response.set("Access-Control-Allow-Origin", origin);
        response.add("Vary", "Origin");
        if (!preflight) return;
        response.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        String asked = request.getFirst("Access-Control-Request-Headers");
        response.set("Access-Control-Allow-Headers", asked != null && HEADER_LIST.matcher(asked.strip()).matches()
                ? asked.strip() : "Authorization, Content-Type");
        response.set("Access-Control-Max-Age", "600");
    }

    private static ApiError forbidden(String message) {
        return new ApiError(403, "permission_error", message, null, "forbidden");
    }
}
