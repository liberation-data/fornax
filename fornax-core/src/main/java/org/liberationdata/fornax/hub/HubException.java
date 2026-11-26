package org.liberationdata.fornax.hub;

import java.io.IOException;

/**
 * A pull that failed for a reason worth telling the user in words: the message says what to do (accept a repo's terms,
 * free some disk), and {@link #reason()} lets an app react to it (Studio can offer a token field on
 * {@link Reason#ACCESS_DENIED}). A dropped connection is retried and resumed before it becomes one of these.
 */
public final class HubException extends IOException {
    private static final long serialVersionUID = 1L;

    public enum Reason {
        /** HTTP 401 or 403: a gated or private repo, and no token, or a token without access. */
        ACCESS_DENIED,
        /** HTTP 404: no such repo, revision or file. */
        NOT_FOUND,
        /** HTTP 429. */
        RATE_LIMITED,
        /** The downloaded bytes' hash isn't the one the Hub gave; the partial file was deleted. */
        CHECKSUM_MISMATCH,
        /** Not enough free disk for what is to be downloaded; nothing was downloaded. */
        DISK_SPACE,
        /** {@code HF_HUB_OFFLINE} is set and a file is missing. */
        OFFLINE,
        /** Any other answer the Hub shouldn't give. */
        PROTOCOL
    }

    private final Reason reason;
    private final String repo;

    public HubException(Reason reason, String repo, String message) {
        super(message);
        this.reason = reason;
        this.repo = repo;
    }

    public Reason reason() { return reason; }

    /** The repo the failure is about, or null when it is about none (the disk). */
    public String repo() { return repo; }
}
