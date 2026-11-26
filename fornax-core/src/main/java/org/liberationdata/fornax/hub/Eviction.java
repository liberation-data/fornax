package org.liberationdata.fornax.hub;

import org.liberationdata.fornax.hub.ModelSet.Part;

import java.nio.file.Path;
import java.util.List;

/**
 * What {@link FornaxHub#evict} did.
 *
 * @param bytesFreed the size of the blobs, copies and partial files it deleted
 * @param removed    the snapshot links, blobs and partial files it deleted
 * @param kept       parts it left because another set on disk uses the same file
 */
public record Eviction(ModelSet set, long bytesFreed, List<Path> removed, List<Part> kept) {
    public Eviction { removed = List.copyOf(removed); kept = List.copyOf(kept); }
}
