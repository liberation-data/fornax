package org.liberationdata.fornax.hub;

import org.liberationdata.fornax.hub.ModelSet.Part;

import java.time.Duration;
import java.util.Optional;

/**
 * Where a {@link FornaxHub#pull} is, as a download button or progress line shows it. Sent when a part changes phase and,
 * while bytes arrive, about ten times a second.
 *
 * @param set            the set being pulled
 * @param part           the part this event is about
 * @param index          the part's position in the set, from 0
 * @param count          how many parts the set has
 * @param phase          what is happening to the part
 * @param bytesDone      the part's bytes on disk, counting what an earlier, interrupted pull left
 * @param bytesTotal     the part's size, or -1 when the Hub didn't say
 * @param setBytesDone   the bytes on disk of the parts this pull downloads (present parts don't count)
 * @param setBytesTotal  the size of the parts this pull downloads, which is what the bar fills to
 * @param bytesPerSecond the download rate so far in this pull (resumed bytes excluded), 0 before any arrive
 */
public record PullProgress(ModelSet set, Part part, int index, int count, Phase phase, long bytesDone, long bytesTotal,
                           long setBytesDone, long setBytesTotal, double bytesPerSecond) {

    public enum Phase {
        /** Asking the Hub for the part's commit, size and hash. */
        CHECKING,
        /** Already on disk (in the cache or a model folder): nothing to download. */
        PRESENT,
        /** Another process (or thread) is downloading the same file; this pull waits for it and then uses its file. */
        WAITING,
        /** Bytes are arriving. */
        DOWNLOADING,
        /** Downloaded, verified and linked into the cache. */
        DONE
    }

    /** The set's progress from 0 to 1, or -1 when its size isn't known. */
    public double fraction() { return setBytesTotal > 0 ? Math.min(1.0, (double) setBytesDone / setBytesTotal) : setBytesTotal == 0 ? 1.0 : -1; }

    /** The time left at the rate so far, when both are known. */
    public Optional<Duration> eta() {
        if (bytesPerSecond <= 0 || setBytesTotal < 0) return Optional.empty();
        return Optional.of(Duration.ofMillis((long) (Math.max(0, setBytesTotal - setBytesDone) / bytesPerSecond * 1000)));
    }
}
