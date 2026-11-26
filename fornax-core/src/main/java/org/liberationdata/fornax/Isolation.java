package org.liberationdata.fornax;

import java.util.Locale;

/**
 * Where a model's engine runs. {@link #IN_PROCESS} (the default) calls it through FFM in this JVM: no overhead, but a
 * native crash (an abort, a segfault) takes the JVM with it. {@link #PROCESS} runs it in a {@link Worker}, a child JVM:
 * a crash fails the calls in flight with a {@link WorkerCrashedException}, and the worker restarts on the next call.
 * See the isolation guide for the costs.
 */
public enum Isolation {
    IN_PROCESS, PROCESS;

    /**
     * From {@code -Dfornax.isolation} or {@code FORNAX_ISOLATION}: {@code process}, or {@code inprocess} (the default).
     * A worker always runs its own models in-process.
     */
    public static Isolation configured() {
        String v = System.getProperty("fornax.isolation");
        if (v == null || v.isBlank()) v = System.getenv("FORNAX_ISOLATION");
        return parse(v);
    }

    static Isolation parse(String v) {
        if (v == null || v.isBlank()) return IN_PROCESS;
        return switch (v.trim().toLowerCase(Locale.ROOT).replace("-", "").replace("_", "")) {
            case "process" -> PROCESS;
            case "inprocess" -> IN_PROCESS;
            default -> throw new IllegalArgumentException("fornax.isolation must be process or inprocess, not " + v);
        };
    }
}
