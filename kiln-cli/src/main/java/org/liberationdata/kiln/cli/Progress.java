package org.liberationdata.kiln.cli;

import java.io.PrintStream;
import java.util.Locale;
import java.util.function.LongConsumer;

/** A download progress line on stderr: redrawn in place on a terminal, every few seconds otherwise. */
final class Progress implements LongConsumer {
    private final String label;
    private final long total;
    private final PrintStream out;
    private final boolean terminal;
    private final long start = System.nanoTime();
    private long first = -1, lastDraw, done;

    Progress(String label, long total, PrintStream out, boolean terminal) {
        this.label = label;
        this.total = total;
        this.out = out;
        this.terminal = terminal;
        this.lastDraw = terminal ? 0 : start;   // off a terminal: a line every few seconds, not one at 0%
    }

    @Override public void accept(long bytes) {
        if (first < 0) first = bytes;
        done = bytes;
        long now = System.nanoTime();
        if (now - lastDraw >= (terminal ? 200_000_000L : 5_000_000_000L)) {
            lastDraw = now;
            draw(false);
        }
    }

    void finish() { draw(true); }

    private void draw(boolean last) {
        double secs = (System.nanoTime() - start) / 1e9;
        double rate = secs > 0 ? (done - first) / secs : 0;
        String line = line(label, done, total, rate, last);
        if (terminal) out.print("\r\033[K" + line + (last ? "\n" : ""));
        else out.println(line);
        out.flush();
    }

    static String line(String label, long done, long total, double bytesPerSec, boolean last) {
        StringBuilder b = new StringBuilder("  ").append(label.length() > 48 ? "..." + label.substring(label.length() - 45) : label);
        if (total > 0) b.append(String.format(Locale.ROOT, "  %3d%%  %s / %s", Math.min(100, done * 100 / total), Fmt.bytes(done), Fmt.bytes(total)));
        else b.append("  ").append(Fmt.bytes(done));
        if (bytesPerSec > 0) b.append("  ").append(Fmt.bytes((long) bytesPerSec)).append("/s");
        if (!last && bytesPerSec > 0 && total > done) b.append("  ").append(Fmt.seconds((total - done) / bytesPerSec)).append(" left");
        return b.toString();
    }
}
