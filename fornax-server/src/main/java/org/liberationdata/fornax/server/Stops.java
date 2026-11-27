package org.liberationdata.fornax.server;

import java.util.List;
import java.util.function.Consumer;

/**
 * A request's {@code stop} sequences, which the engine doesn't take: the reply's content passes through, held back
 * only as far as it could be the start of a stop sequence, and ends before the first one. The caller stops the
 * generation when {@link #accept} says so.
 */
final class Stops {
    private final List<String> stops;
    private final int hold;   // the longest stop, less one: what may be the start of one
    private final Consumer<String> out;
    private final StringBuilder pending = new StringBuilder(), sent = new StringBuilder();
    private boolean stopped;

    Stops(List<String> stops, Consumer<String> out) {
        this.stops = stops.stream().filter(s -> !s.isEmpty()).toList();
        this.hold = this.stops.stream().mapToInt(String::length).max().orElse(1) - 1;
        this.out = out;
    }

    /** Takes a piece of content; true once a stop sequence has been reached (pieces after it are dropped). */
    boolean accept(String piece) {
        if (stopped) return true;
        if (stops.isEmpty()) { emit(piece); return false; }
        pending.append(piece);
        int at = -1;
        for (String s : stops) {
            int i = pending.indexOf(s);
            if (i >= 0 && (at < 0 || i < at)) at = i;
        }
        if (at >= 0) {
            emit(pending.substring(0, at));
            pending.setLength(0);
            stopped = true;
            return true;
        }
        int safe = pending.length() - hold;
        if (safe > 0 && Character.isHighSurrogate(pending.charAt(safe - 1))) safe--;   // never split a pair
        if (safe > 0) {
            emit(pending.substring(0, safe));
            pending.delete(0, safe);
        }
        return false;
    }

    /** The end of the content: what was held back goes out, unless a stop sequence was reached. */
    void finish() {
        if (!stopped && !pending.isEmpty()) emit(pending.toString());
        pending.setLength(0);
    }

    boolean stopped() { return stopped; }

    /** The content passed on so far. */
    String text() { return sent.toString(); }

    private void emit(String s) {
        if (s.isEmpty()) return;
        sent.append(s);
        out.accept(s);
    }
}
