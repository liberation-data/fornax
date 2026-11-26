package org.liberationdata.fornax;

/**
 * A streaming subscriber fell so far behind that its buffer filled, so Fornax stopped the generation.
 *
 * <p>Why stop rather than wait: waiting would hold the GPU idle for the slowest consumer, and dropping items would
 * hand it a reply with holes in it. A subscriber that wants everything requests {@code Long.MAX_VALUE} and keeps up,
 * or is given a bigger buffer ({@link ModelThread#streamOn}).
 */
public final class StreamOverflowException extends IllegalStateException {
    private static final long serialVersionUID = 1L;

    public StreamOverflowException(int capacity) {
        super("the subscriber fell " + capacity + " items behind, so the generation was stopped;"
                + " request more, keep up, or give it a bigger buffer");
    }
}
