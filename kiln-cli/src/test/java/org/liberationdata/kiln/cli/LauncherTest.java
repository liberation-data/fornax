package org.liberationdata.kiln.cli;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * bin/kiln from the installed distribution ({@code installDist}, which the test task runs first): as it ships, reached
 * through a symlink on PATH, and rendered the way an installer such as Kiln Studio renders it, with its own java, jars
 * folder and natives folder.
 */
@DisabledOnOs(OS.WINDOWS)
class LauncherTest {
    @TempDir Path tmp;

    static Path install() { return Path.of(System.getProperty("kiln.cli.install", "build/install/kiln-cli")); }

    static String java() { return ProcessHandle.current().info().command().orElse("java"); }

    record Run(int code, String output) {}

    Run run(Path launcher, Map<String, String> env, String... args) throws Exception {
        List<String> cmd = new ArrayList<>(List.of(launcher.toString()));
        cmd.addAll(List.of(args));
        var pb = new ProcessBuilder(cmd).redirectErrorStream(true);
        pb.environment().remove("KILN_JAVA");
        pb.environment().remove("KILN_CLI_LIB");
        pb.environment().remove("KILN_LIBRARY_PATH");
        pb.environment().put("JAVA_HOME", Path.of(java()).getParent().getParent().toString());
        pb.environment().putAll(env);
        Process p = pb.start();
        String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        return new Run(p.waitFor(), out);
    }

    @Test
    void runsAsShippedAndThroughASymlink() throws Exception {
        Path kiln = install().resolve("bin/kiln");
        assertTrue(Files.isExecutable(kiln), kiln + " is executable");
        Run r = run(kiln, Map.of(), "--version");
        assertEquals(0, r.code(), r.output());
        assertTrue(r.output().startsWith("kiln "), r.output());

        Path onPath = Files.createDirectories(tmp.resolve("bin")).resolve("kiln");
        Files.createSymbolicLink(onPath, kiln.toAbsolutePath());
        assertEquals(0, run(onPath, Map.of(), "--version").code());
    }

    @Test
    void runsRenderedWithAnAppsJavaJarsAndNatives() throws Exception {
        Path natives = Files.createDirectories(tmp.resolve("natives"));
        Path rendered = render(Map.of("@KILN_JAVA@", java(), "@KILN_CLI_LIB@", install().resolve("lib").toAbsolutePath().toString(),
                "@KILN_LIBRARY_PATH@", natives.toString()));
        Run r = run(rendered, Map.of("JAVA_HOME", "/nonexistent"), "info");
        assertEquals(0, r.code(), r.output());
        assertTrue(r.output().contains("Natives   searched " + natives), r.output());
        assertTrue(r.output().contains(java().replaceAll("/bin/java$", "")), r.output());

        // the environment still wins over what was rendered
        Run bad = run(rendered, Map.of("KILN_JAVA", "/nonexistent/java"), "--version");
        assertEquals(1, bad.code(), bad.output());
        assertTrue(bad.output().contains("Java 22 or newer"), bad.output());
    }

    @Test
    void saysWhenTheJarsAreMissing() throws Exception {
        Path rendered = render(Map.of("@KILN_CLI_LIB@", tmp.resolve("nowhere").toString()));
        Run r = run(rendered, Map.of(), "--version");
        assertEquals(1, r.code());
        assertTrue(r.output().contains("KILN_CLI_LIB"), r.output());
    }

    /** bin/kiln with placeholders replaced, as an installer writes it into a folder on PATH. */
    Path render(Map<String, String> values) throws IOException {
        String text = Files.readString(install().resolve("bin/kiln"));
        for (var e : values.entrySet()) {
            assertTrue(text.contains("'" + e.getKey() + "'"), e.getKey());
            text = text.replace(e.getKey(), e.getValue());
        }
        Path out = Files.createDirectories(tmp.resolve("local-bin")).resolve("kiln");
        Files.writeString(out, text);
        Files.setPosixFilePermissions(out, PosixFilePermissions.fromString("rwxr-xr-x"));
        return out;
    }
}
