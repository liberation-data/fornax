package org.liberationdata.fornax.http;

import java.util.Locale;
import java.util.regex.Pattern;

/**
 * The Host check shared by fornax-server and fornax-studio-server. A server bound to loopback is still reachable from
 * the web pages open in the user's browser, through DNS rebinding: a page's own name is pointed at 127.0.0.1, and its
 * requests carry that name as their Host. Answering only a Host that is a loopback address (or localhost) refuses them.
 *
 * <p>For Fornax's own servers, not an API for applications: it may change in any release.
 */
public final class Loopback {
    /** Any address in 127.0.0.0/8, all of it loopback (a name can't be rebound to a literal address). */
    private static final Pattern V4 = Pattern.compile("127(\\.(25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)){3}");

    private Loopback() {}

    /** Whether host (a Host header) is localhost, [::1] or a 127.x.y.z address, with the given port; case aside. */
    public static boolean host(String host, int port) {
        if (host == null) return false;
        String h = host.toLowerCase(Locale.ROOT), suffix = ":" + port;
        if (!h.endsWith(suffix)) return false;
        String name = h.substring(0, h.length() - suffix.length());
        return name.equals("localhost") || name.equals("[::1]") || V4.matcher(name).matches();
    }
}
