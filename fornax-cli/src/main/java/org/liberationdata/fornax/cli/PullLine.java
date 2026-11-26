package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.hub.PullListener;
import org.liberationdata.fornax.hub.PullProgress;

import java.io.PrintStream;
import java.util.Locale;

/**
 * {@code fornax pull}'s progress, drawn from {@link PullProgress}: a line per part, redrawn in place on a terminal (and
 * every few seconds otherwise) while it downloads, with the part's bytes, the rate and the time left for the whole pull.
 */
final class PullLine implements PullListener {
    private final PrintStream out;
    private final boolean terminal;
    private long lastDraw;
    private boolean drawn;   // a progress line is on screen, not yet ended

    PullLine(PrintStream out, boolean terminal) {
        this.out = out;
        this.terminal = terminal;
        this.lastDraw = System.nanoTime();   // off a terminal: the first line after a few seconds, not one at 0%
    }

    @Override public void progress(PullProgress p) {
        switch (p.phase()) {
            case CHECKING -> {}
            case PRESENT -> end("  " + label(p) + "  " + Fmt.bytes(p.bytesTotal()) + "  (already downloaded)");
            case WAITING -> end("  " + label(p) + "  waiting for another download of the same file");
            case DOWNLOADING -> {
                long now = System.nanoTime();
                if (now - lastDraw < (terminal ? 200_000_000L : 5_000_000_000L)) return;
                lastDraw = now;
                if (terminal) { out.print("\r\033[K" + line(p, false)); drawn = true; }
                else out.println(line(p, false));
                out.flush();
            }
            case DONE -> end(line(p, true));
        }
    }

    private void end(String line) {
        out.print(terminal && drawn ? "\r\033[K" : "");
        out.println(line);
        out.flush();
        drawn = false;
    }

    private static String label(PullProgress p) {
        String name = p.part().primary().fileName();
        if (name.length() > 48) name = "..." + name.substring(name.length() - 45);
        return (p.count() > 1 ? "[" + (p.index() + 1) + "/" + p.count() + "] " : "") + name;
    }

    /** One part's line: its bytes (and percent), the rate, and the time left for the whole pull. */
    static String line(PullProgress p, boolean done) {
        StringBuilder b = new StringBuilder("  ").append(label(p));
        long have = p.bytesDone(), total = p.bytesTotal();
        if (total > 0) b.append(String.format(Locale.ROOT, "  %3d%%  %s / %s", Math.min(100, have * 100 / total), Fmt.bytes(have), Fmt.bytes(total)));
        else b.append("  ").append(Fmt.bytes(have));
        if (p.bytesPerSecond() > 0) b.append("  ").append(Fmt.bytes((long) p.bytesPerSecond())).append("/s");
        if (!done) p.eta().ifPresent(eta -> b.append("  ").append(Fmt.seconds(eta.toMillis() / 1000.0)).append(" left"));
        return b.toString();
    }
}
