package org.liberationdata.fornax.cli;

import java.util.function.Consumer;
import java.util.regex.Pattern;

/**
 * Hides a reasoning model's {@code <think>…</think>} blocks (Qwen3, DeepSeek-R1, ...) in streamed text, or shows them
 * when asked. Tags split across pieces are handled: a piece ending in what may be the start of a tag is held back
 * until the next piece says. The blank lines after a hidden block go with it.
 *
 * <p>Some models (Qwen3 Thinking, DeepSeek-R1 distills) are trained with the opening {@code <think>} in the prompt,
 * so their reply starts inside the reasoning and has only the closing {@code </think>}. Fornax's chat path formats with
 * llama.cpp's built-in templates, which don't add it, and Java can't see the prompt, so the filter decides from the
 * reply: until a tag shows up it holds the start back, up to {@link #UNSURE} characters, or for as long as it opens
 * the way reasoning does ("Okay, the user..."). A {@code </think>} in that time makes what came before reasoning.
 * One that comes later, after the start was let out, still ends the reasoning: the tag is hidden and the reply (for
 * the history) starts after it. With {@code startsInside} (an earlier reply of the chat opened inside) the reply is
 * taken as reasoning from the start.
 */
final class ThinkFilter implements Consumer<String> {
    private static final String OPEN = "<think>", CLOSE = "</think>";
    /** How much of a reply's start is held back, without a tag or a reasoning-like opening, before it is let out. */
    static final int UNSURE = 200;
    private static final Pattern REASONING = Pattern.compile(
            "(?i)\\s*(okay|alright|hmm+|let me|let's|i need|we need|we are|the user)\\b");

    private final Consumer<String> out;
    private final boolean show;
    private final StringBuilder pending = new StringBuilder(), reply = new StringBuilder();
    private boolean inside, trimNext, unsure = true, sawTag, openedInside;

    /** @param show pass the blocks through, tags and all, instead of hiding them */
    ThinkFilter(Consumer<String> out, boolean show) { this(out, show, false); }

    /** @param startsInside the reply starts inside a think block whose {@code <think>} was in the prompt */
    ThinkFilter(Consumer<String> out, boolean show, boolean startsInside) {
        this.out = out;
        this.show = show;
        if (startsInside) {
            inside = sawTag = openedInside = true;
            unsure = false;
            if (show) out.accept(OPEN);
        }
    }

    @Override public void accept(String piece) {
        pending.append(piece);
        drain(false);
    }

    /** Sends what is held back: call once the reply is complete. */
    void finish() { drain(true); }

    /** The reply without its think blocks: what the history keeps. */
    String reply() { return reply.toString().strip(); }

    /** Whether the reply started inside a think block, its {@code <think>} not in the reply. */
    boolean openedInside() { return openedInside; }

    private void drain(boolean end) {
        while (true) {
            if (unsure) {
                int close = pending.indexOf(CLOSE);
                String start = pending.toString().stripLeading();
                if (start.startsWith(OPEN) || close < 0 && end) unsure = false;
                else if (close >= 0 && pending.lastIndexOf(OPEN, close) < 0) {
                    // the reply opened inside: everything before </think> was reasoning
                    unsure = false;
                    inside = sawTag = openedInside = true;
                    if (show) out.accept(OPEN);
                } else if (close < 0 && (OPEN.startsWith(start)
                        || pending.length() <= UNSURE || REASONING.matcher(start).lookingAt())) return;
                else unsure = false;
            }
            String tag = inside ? CLOSE : OPEN;
            int i = pending.indexOf(tag);
            int late = inside || sawTag ? -1 : pending.indexOf(CLOSE);
            if (late >= 0 && (i < 0 || late < i)) {
                // a </think> after the start was let out: what came before was reasoning after all
                emit(pending.substring(0, late));
                if (show) out.accept(CLOSE);
                pending.delete(0, late + CLOSE.length());
                reply.setLength(0);
                sawTag = openedInside = true;
                continue;
            }
            if (i >= 0) {
                emit(pending.substring(0, i));
                if (show) out.accept(tag);
                pending.delete(0, i + tag.length());
                inside = !inside;
                sawTag = true;
                if (!inside) trimNext = true;
                continue;
            }
            int keep = end ? 0 : Math.max(heldBack(tag), sawTag || inside ? 0 : heldBack(CLOSE));
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
