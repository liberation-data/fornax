package org.liberationdata.kiln.cli;

import java.util.Locale;

/** Human-readable sizes and durations. */
final class Fmt {
    private Fmt() {}

    /** Decimal units, as Hugging Face shows them: 639446688 -> "639 MB". */
    static String bytes(long n) {
        if (n < 1000) return n + " B";
        String[] units = {"kB", "MB", "GB", "TB"};
        double v = n;
        int u = -1;
        while (v >= 1000 && u < units.length - 1) { v /= 1000; u++; }
        return String.format(Locale.ROOT, v >= 100 || u < 1 ? "%.0f %s" : "%.1f %s", v, units[u]);
    }

    static String seconds(double s) {
        if (s < 60) return String.format(Locale.ROOT, "%.1fs", s);
        long t = Math.round(s);
        return t >= 3600 ? String.format(Locale.ROOT, "%d:%02d:%02d", t / 3600, t / 60 % 60, t % 60)
                : String.format(Locale.ROOT, "%d:%02d", t / 60, t % 60);
    }
}
