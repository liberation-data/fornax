package org.liberationdata.kiln;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Runs a main class in a fresh JVM with this test's class path: for what can't be shown in the test JVM, such as
 * a crash (which would take the test JVM with it) or virtual-thread scheduler settings (read once, at start).
 */
final class ForkedJvm {
    record Result(int exitCode, String output) {}

    static Result run(Class<?> main, List<String> jvmArgs, String... args) throws IOException, InterruptedException {
        List<String> cmd = new ArrayList<>();
        cmd.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        cmd.add("--enable-native-access=ALL-UNNAMED");
        cmd.addAll(jvmArgs);
        cmd.add("-cp");
        cmd.add(System.getProperty("java.class.path"));
        cmd.add(main.getName());
        cmd.addAll(List.of(args));
        Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        byte[] out = p.getInputStream().readAllBytes();
        if (!p.waitFor(5, TimeUnit.MINUTES)) { p.destroyForcibly(); throw new IllegalStateException("forked JVM hung"); }
        String text = new String(out);
        System.out.println("---- forked " + main.getSimpleName() + " " + String.join(" ", args) + " (exit " + p.exitValue() + ")\n" + text);
        return new Result(p.exitValue(), text);
    }

    /** The value printed on a line "key=value", as a long. */
    static long value(String output, String key) {
        for (String line : output.split("\n")) if (line.startsWith(key + "=")) return Long.parseLong(line.substring(key.length() + 1).trim());
        throw new AssertionError("no " + key + "= in:\n" + output);
    }

    private ForkedJvm() {}
}
