package org.liberationdata.fornax.hub;

import org.liberationdata.fornax.hub.ModelSet.Part;

/**
 * Receives a pull's {@link PullProgress}, on the pulling thread. Return quickly: the download waits while it runs.
 * An exception thrown here stops the pull (its partial file stays, so a later pull resumes).
 */
@FunctionalInterface
public interface PullListener {
    PullListener NONE = p -> {};

    void progress(PullProgress progress);

    /**
     * Logs a pull to {@code System.Logger} {@code org.liberationdata.fornax.hub} at INFO: each file as it starts and
     * finishes, and the set's progress about every ten seconds between. What an engine constructor uses when it
     * downloads a model on first use, so a long download isn't silent. Silence it through the logging configuration.
     */
    static PullListener logging() {
        System.Logger log = System.getLogger("org.liberationdata.fornax.hub");
        return new PullListener() {
            private long next;
            private Part started;

            @Override public void progress(PullProgress p) {
                switch (p.phase()) {
                    case DOWNLOADING -> {
                        long now = System.nanoTime();
                        if (p.part() != started) {
                            started = p.part();
                            next = now + 10_000_000_000L;
                            log.log(System.Logger.Level.INFO, "Downloading " + p.set().id() + ": " + p.part().primary().repo()
                                    + " " + p.part().primary().file() + (p.bytesTotal() < 0 ? "" : " (" + FornaxHub.bytes(p.bytesTotal()) + ")"));
                        } else if (now >= next) {
                            next = now + 10_000_000_000L;
                            double f = p.fraction();
                            log.log(System.Logger.Level.INFO, p.set().id() + ": " + (f < 0 ? FornaxHub.bytes(p.setBytesDone())
                                    : String.format(java.util.Locale.ROOT, "%.0f%%", f * 100)) + p.eta().map(d -> ", "
                                    + Math.max(1, d.toSeconds()) + " s left").orElse(""));
                        }
                    }
                    case WAITING -> log.log(System.Logger.Level.INFO, p.set().id() + ": waiting for another download of "
                            + p.part().primary().file());
                    case DONE -> log.log(System.Logger.Level.INFO, p.set().id() + ": " + p.part().primary().file() + " done");
                    default -> {}
                }
            }
        };
    }
}
