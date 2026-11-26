package org.liberationdata.kiln;

import java.util.List;
import java.util.OptionalInt;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A {@link Worker} process died, or stopped answering, while a call was in flight or while it started. Its models are
 * gone; the next call starts a new worker and loads them again. The JVM that made the call is unaffected.
 */
public final class WorkerCrashedException extends RuntimeException {
    private static final long serialVersionUID = 1L;
    private static final Pattern FATAL = Pattern.compile("#\\s+(SIG[A-Z]+) \\(0x");

    private final long pid;
    private final int exitCode;
    private final transient List<String> lastOutput;

    WorkerCrashedException(String worker, long pid, int exitCode, String reason, List<String> lastOutput) {
        super(message(worker, pid, exitCode, reason, lastOutput));
        this.pid = pid;
        this.exitCode = exitCode;
        this.lastOutput = List.copyOf(lastOutput);
    }

    /** The same crash, for another call that was in flight (an exception's stack belongs to one thread). */
    WorkerCrashedException copy() {
        var e = new WorkerCrashedException(getMessage(), pid, exitCode, lastOutput);
        e.setStackTrace(new StackTraceElement[0]);
        return e;
    }

    private WorkerCrashedException(String message, long pid, int exitCode, List<String> lastOutput) {
        super(message);
        this.pid = pid;
        this.exitCode = exitCode;
        this.lastOutput = lastOutput;
    }

    /** The worker's process id. */
    public long pid() { return pid; }

    /** Its exit code as {@link Process#exitValue()} reports it (128 + the signal, for one a signal ended), or -1. */
    public int exitCode() { return exitCode; }

    /** The signal that ended the process (6 for an abort, 9 for a kill), if a signal did. */
    public OptionalInt signal() {
        return exitCode > 128 && exitCode < 128 + 65 ? OptionalInt.of(exitCode - 128) : OptionalInt.empty();
    }

    /**
     * The fatal signal the worker JVM's crash handler reported ({@code SIGSEGV}, {@code SIGBUS}, ...), or null. A
     * segfault in native code shows up here: the JVM's handler catches it, writes its {@code hs_err} report (the path
     * is in {@link #lastOutput()}), and exits with code 1, since Kiln turns core dumps off for workers.
     */
    public String fatalSignal() { return fatal(lastOutput); }

    /** The worker's last lines of output (standard output and error), oldest first: the engine's last log lines. */
    public List<String> lastOutput() { return lastOutput; }

    private static String fatal(List<String> lines) {
        for (String l : lines) {
            Matcher m = FATAL.matcher(l);
            if (m.find()) return m.group(1);
        }
        return null;
    }

    private static String message(String worker, long pid, int exitCode, String reason, List<String> out) {
        StringBuilder b = new StringBuilder("kiln worker ").append(worker).append(" (pid ").append(pid).append(") ");
        b.append(reason == null ? "crashed" : reason);
        if (exitCode > 128 && exitCode < 128 + 65) b.append(": killed by ").append(signalName(exitCode - 128));
        else if (exitCode >= 0) b.append(": exit code ").append(exitCode);
        String fatal = fatal(out);
        if (fatal != null) b.append(", after a fatal ").append(fatal);
        b.append(". Its models are unloaded; the next call restarts it.");
        if (!out.isEmpty()) {
            b.append(" Last output:");
            for (String l : out.subList(Math.max(0, out.size() - 12), out.size())) b.append("\n  ").append(l);
        }
        return b.toString();
    }

    static String signalName(int sig) {
        String name = switch (sig) {
            case 4 -> "SIGILL"; case 6 -> "SIGABRT"; case 7 -> "SIGBUS"; case 8 -> "SIGFPE"; case 9 -> "SIGKILL";
            case 11 -> "SIGSEGV"; case 13 -> "SIGPIPE"; case 15 -> "SIGTERM";
            default -> null;
        };
        return name == null ? "signal " + sig : name + " (signal " + sig + ")";
    }
}
