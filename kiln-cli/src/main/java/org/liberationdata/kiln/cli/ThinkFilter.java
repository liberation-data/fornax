package org.liberationdata.kiln.cli;

import java.util.function.Consumer;

/**
 * Hides a reasoning model's {@code <think>…</think>} blocks (Qwen3, DeepSeek-R1, ...) in streamed text, or shows them
 * when asked. Tags split across pieces are handled: a piece ending in what may be the start of a tag is held back
 * until the next piece says. The blank lines after a hidden block go with it.
 */
final class ThinkFilter implements Consumer<String> {
    private static final String OPEN = "<think>", CLOSE = "</think>";

    private final Consumer<String> out;
    private final boolean show;
    private final StringBuilder pending = new StringBuilder(), reply = new StringBuilder();
    private boolean inside, trimNext;

    /** @param show pass the blocks through, tags and all, instead of hiding them */
    ThinkFilter(Consumer<String> out, boolean show) {
        this.out = out;
        this.show = show;
    }

    @Override public void accept(String piece) {
        pending.append(piece);
        drain(false);
    }

    /** Sends what is held back: call once the reply is complete. */
    void finish() { drain(true); }

    /** The reply without its think blocks: what the history keeps. */
    String reply() { return reply.toString().strip(); }

    private void drain(boolean end) {
        while (true) {
            String tag = inside ? CLOSE : OPEN;
            int i = pending.indexOf(tag);
            if (i >= 0) {
                emit(pending.substring(0, i));
                if (show) out.accept(tag);
                pending.delete(0, i + tag.length());
                inside = !inside;
                if (!inside) trimNext = true;
                continue;
            }
            int keep = end ? 0 : heldBack(tag);
            emit(pending.substring(0, pending.length() - keep));
            pending.delete(0, pending.length() - keep);
            return;
        }
    }

    /** The length of the longest end of {@code pending} that {@code tag} starts with. */
    private int heldBack(String tag) {
        for (int n = Math.min(tag.length() - 1, pending.length()); n > 0; n--)
            if (tag.startsWith(pending.substring(pending.length() - n))) return n;
        return 0;
    }

    private void emit(String text) {
        if (text.isEmpty()) return;
        if (inside) {
            if (show) out.accept(text);
            return;
        }
        if (trimNext && !show) {
            text = text.stripLeading();
            if (text.isEmpty()) return;
        }
        trimNext = false;
        reply.append(text);
        out.accept(text);
    }
}
