package org.liberationdata.kiln.hub;

import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * A pull running on its own thread ({@link KilnHub#pullAsync}), for a UI that shows its progress and offers to cancel
 * it. Cancelling stops the transfer at once (the stream in flight is closed) and leaves the partial file, so pulling
 * the same set again resumes where it stopped.
 */
public final class PullTask {
    private final ModelSet set;
    private final CompletableFuture<Map<String, Path>> result;
    private final HubHttp.Cancel cancel;

    PullTask(ModelSet set, CompletableFuture<Map<String, Path>> result, HubHttp.Cancel cancel) {
        this.set = set;
        this.result = result;
        this.cancel = cancel;
        result.whenComplete((r, e) -> { if (result.isCancelled()) cancel.cancel(); });   // cancelling the future cancels the pull
    }

    public ModelSet set() { return set; }

    /**
     * The set's files by role once every part is on disk. It fails with the pull's {@link HubException} or
     * {@link java.io.IOException}, or with a {@link java.util.concurrent.CancellationException} after {@link #cancel}.
     */
    public CompletableFuture<Map<String, Path>> result() { return result; }

    /** Stops the pull. The future completes with a CancellationException once the pulling thread has stopped. */
    public void cancel() { cancel.cancel(); }

    public boolean isDone() { return result.isDone(); }
}
