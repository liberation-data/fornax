package org.liberationdata.kiln.hub;

/**
 * Receives a pull's {@link PullProgress}, on the pulling thread. Return quickly: the download waits while it runs.
 * An exception thrown here stops the pull (its partial file stays, so a later pull resumes).
 */
@FunctionalInterface
public interface PullListener {
    PullListener NONE = p -> {};

    void progress(PullProgress progress);
}
