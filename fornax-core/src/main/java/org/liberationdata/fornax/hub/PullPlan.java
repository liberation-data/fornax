package org.liberationdata.fornax.hub;

import org.liberationdata.fornax.hub.ModelSet.Part;
import org.liberationdata.fornax.hub.ModelSet.Source;

import java.nio.file.Path;
import java.util.List;

/**
 * What a {@link FornaxHub#pull} would download, with sizes from the Hub: for a "Download (9.8 GB)" button before the
 * user commits to it, and for the free-space check a pull makes before it starts.
 */
public record PullPlan(ModelSet set, List<Item> items) {

    /**
     * One part. A present part has {@code present} set and nothing else; a needed one has the Hub's commit, etag
     * (its blob name and hash) and size, and how much an earlier, interrupted pull left on disk.
     */
    public record Item(Part part, Path present, Source source, String commit, String etag, long size, long bytesOnDisk) {
        public boolean needed() { return present == null; }

        /** What is left to download, 0 when present, -1 when the Hub gave no size. */
        public long bytesToDownload() { return present != null ? 0 : size < 0 ? -1 : Math.max(0, size - bytesOnDisk); }
    }

    public PullPlan { items = List.copyOf(items); }

    public boolean complete() { return items.stream().noneMatch(Item::needed); }

    /** The bytes left to download (unknown sizes count as 0). */
    public long bytesToDownload() { return items.stream().mapToLong(i -> Math.max(0, i.bytesToDownload())).sum(); }

    /** The full size of the parts to download, counting what is already partly there. */
    public long bytesTotal() { return items.stream().filter(Item::needed).mapToLong(i -> Math.max(0, i.size())).sum(); }
}
