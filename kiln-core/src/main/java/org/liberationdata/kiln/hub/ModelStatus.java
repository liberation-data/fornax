package org.liberationdata.kiln.hub;

import org.liberationdata.kiln.hub.ModelSet.Part;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * What of a {@link ModelSet} is on disk, read from the disk alone (no network), so it is cheap enough to compute for
 * every set each time a model list is drawn. See {@link KilnHub#plan} for the sizes of missing parts.
 */
public record ModelStatus(ModelSet set, List<PartStatus> parts) {

    public enum State {
        /** On disk, complete. */
        PRESENT,
        /** A download was started and not finished; the next pull resumes it. For a set: some parts are on disk. */
        PARTIAL,
        /** Not on disk. */
        MISSING
    }

    /** Where a present or partial part was found. */
    public enum Where { HF_CACHE, MODEL_FOLDER }

    /**
     * One part's state.
     *
     * @param path        the file (a partial part's {@code .incomplete} file); null when missing
     * @param where       where {@code path} is; null when missing
     * @param bytesOnDisk how much of it is on disk
     * @param size        its full size: known when present, or partial from the pull that started it; -1 when missing
     */
    public record PartStatus(Part part, State state, Path path, Where where, long bytesOnDisk, long size) {}

    public ModelStatus { parts = List.copyOf(parts); }

    /** PRESENT when every part is, MISSING when none has any bytes on disk, PARTIAL otherwise. */
    public State state() {
        if (parts.stream().allMatch(p -> p.state() == State.PRESENT)) return State.PRESENT;
        if (parts.stream().allMatch(p -> p.state() == State.MISSING)) return State.MISSING;
        return State.PARTIAL;
    }

    /** Whether every part is present, so the model can be loaded. */
    public boolean available() { return state() == State.PRESENT; }

    public long bytesOnDisk() { return parts.stream().mapToLong(PartStatus::bytesOnDisk).sum(); }

    /** The present parts' files, by role. */
    public Map<String, Path> paths() {
        Map<String, Path> out = new LinkedHashMap<>();
        for (PartStatus p : parts) if (p.state() == State.PRESENT) out.put(p.part().role(), p.path());
        return out;
    }
}
