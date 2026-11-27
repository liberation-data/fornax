package org.liberationdata.fornax.cli;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * website/public/install.sh (fornax4j.org/install.sh), run with sh against this build's distZip laid out as a release
 * lays it out (fornax-cli-&lt;version&gt;-&lt;platform&gt;.zip and SHA256SUMS), read through FORNAX_DIST_URL=file://...,
 * into a throwaway HOME.
 */
@DisabledOnOs(OS.WINDOWS)
class InstallScriptTest {
    static final String VERSION = System.getProperty("fornax.cli.version", "dev");
    static final List<String> PLATFORMS = List.of("linux-x86_64-vulkan", "macos-aarch64-metal");

    @TempDir Path tmp;
    Path home;
    Path dist;

    record Run(int code, String output) {}

    @BeforeEach
    void release() throws Exception {
        home = Files.createDirectories(tmp.resolve("home"));
        dist = Files.createDirectories(tmp.resolve("release"));
        Path zip = Path.of(System.getProperty("fornax.cli.distZip", "build/distributions/fornax-cli-" + VERSION + ".zip"));
        StringBuilder sums = new StringBuilder();
        // The same zip under each platform's name, so the test runs on either kind of machine the script supports
        for (String platform : PLATFORMS) {
            String name = "fornax-cli-" + VERSION + "-" + platform + ".zip";
            Files.copy(zip, dist.resolve(name));
            sums.append(sha256(zip)).append("  ").append(name).append('\n');
        }
        Files.writeString(dist.resolve("SHA256SUMS"), sums);
    }

    Run install(Map<String, String> env) throws Exception {
        Path script = Path.of(System.getProperty("fornax.install.script", "../website/public/install.sh"));
        var pb = new ProcessBuilder("sh", script.toString()).redirectErrorStream(true);
        Map<String, String> e = pb.environment();
        for (String v : List.of("FORNAX_JAVA", "FORNAX_HOME", "FORNAX_BIN_DIR", "FORNAX_VERSION", "XDG_DATA_HOME")) e.remove(v);
        e.put("HOME", home.toString());
        e.put("JAVA_HOME", Path.of(LauncherTest.java()).getParent().getParent().toString());
        e.put("FORNAX_DIST_URL", "file://" + dist);
        e.putAll(env);
        Process p = pb.start();
        String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        return new Run(p.waitFor(), out);
    }

    Run install() throws Exception { return install(Map.of()); }

    Path link() { return home.resolve(".local/bin/fornax"); }

    Path data() { return home.resolve(".local/share/fornax"); }

    @Test
    void installsChecksAndLinksAndEditsNoStartupFile() throws Exception {
        Run r = install();
        assertEquals(0, r.code(), r.output());
        assertTrue(r.output().contains("as SHA256SUMS says"), r.output());
        assertTrue(Files.isSymbolicLink(link()), r.output());
        assertEquals(data().toRealPath().resolve("current/bin/fornax"), Files.readSymbolicLink(link()));
        assertEquals(Path.of("fornax-cli-" + VERSION), Files.readSymbolicLink(data().resolve("current")));
        assertTrue(r.output().contains("Java:    "), r.output());

        // ~/.local/bin isn't on this PATH: the line to add is printed, and no startup file is written
        assertTrue(r.output().contains("export PATH=\"$HOME/.local/bin:$PATH\""), r.output());
        try (Stream<Path> files = Files.list(home)) {
            assertEquals(List.of(home.resolve(".local")), files.toList());
        }
        try (Stream<Path> files = Files.list(data())) {
            assertFalse(files.anyMatch(f -> f.getFileName().toString().startsWith(".install.")), "work folder left behind");
        }

        var pb = new ProcessBuilder(link().toString(), "--version").redirectErrorStream(true);
        pb.environment().put("JAVA_HOME", Path.of(LauncherTest.java()).getParent().getParent().toString());
        Process p = pb.start();
        String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        assertEquals(0, p.waitFor(), out);
        assertEquals("fornax " + VERSION, out.strip());
    }

    @Test
    void runAgainReplacesItAndDeletesEarlierVersions() throws Exception {
        assertEquals(0, install().code());
        Path older = Files.createDirectories(data().resolve("fornax-cli-0.0.1/bin"));
        Run r = install(Map.of("FORNAX_VERSION", "v" + VERSION));
        assertEquals(0, r.code(), r.output());
        assertFalse(Files.exists(older), "the earlier version is still there");
        assertTrue(Files.isRegularFile(data().resolve("current/lib/fornax-cli-" + VERSION + ".jar")));
    }

    @Test
    void refusesAZipThatDoesntMatchItsChecksum() throws Exception {
        String sums = Files.readString(dist.resolve("SHA256SUMS")).replaceAll("(?m)^[0-9a-f]{64}", "0".repeat(64));
        Files.writeString(dist.resolve("SHA256SUMS"), sums);
        Run r = install();
        assertEquals(1, r.code(), r.output());
        assertTrue(r.output().contains("not installing it"), r.output());
        assertFalse(Files.exists(link()) || Files.isSymbolicLink(link()));
        try (Stream<Path> files = Files.list(data())) {
            assertEquals(List.of(), files.toList(), "something was left in " + data());
        }
    }

    @Test
    void refusesAnotherVersionThanTheOneAskedFor() throws Exception {
        Run r = install(Map.of("FORNAX_VERSION", "9.9.9"));
        assertEquals(1, r.code(), r.output());
        assertTrue(r.output().contains("asked for 9.9.9"), r.output());
    }

    @Test
    void leavesAFornaxItDidntInstallAlone() throws Exception {
        Files.createDirectories(link().getParent());
        Files.writeString(link(), "#!/bin/sh\necho studio\n");
        Run r = install();
        assertEquals(1, r.code(), r.output());
        assertTrue(r.output().contains("isn't this script's link"), r.output());
        assertEquals("#!/bin/sh\necho studio\n", Files.readString(link()));
    }

    @Test
    void installsWithTheFoldersGiven() throws Exception {
        Path where = tmp.resolve("opt/fornax");
        Path bin = Files.createDirectories(tmp.resolve("bin"));
        Map<String, String> env = new HashMap<>(Map.of("FORNAX_HOME", where.toString(), "FORNAX_BIN_DIR", bin.toString()));
        env.put("PATH", bin + ":" + System.getenv("PATH"));
        Run r = install(env);
        assertEquals(0, r.code(), r.output());
        assertEquals(where.toRealPath().resolve("current/bin/fornax"), Files.readSymbolicLink(bin.resolve("fornax")));
        assertFalse(r.output().contains("export PATH"), r.output());
        assertFalse(Files.exists(home.resolve(".local")));
    }

    @Test
    void saysHowToGetJavaWhenItsTooOld() throws Exception {
        Path java = tmp.resolve("java");
        Files.writeString(java, "#!/bin/sh\necho 'openjdk version \"17.0.2\" 2022-01-18' >&2\n");
        Files.setPosixFilePermissions(java, PosixFilePermissions.fromString("rwxr-xr-x"));
        Run r = install(Map.of("FORNAX_JAVA", java.toString()));
        assertEquals(0, r.code(), r.output());  // fornax is installed all the same
        assertTrue(r.output().contains("is Java 17"), r.output());
        assertTrue(r.output().contains("sdk install java"), r.output());

        Run none = install(Map.of("FORNAX_JAVA", tmp.resolve("nowhere/java").toString()));
        assertEquals(0, none.code(), none.output());
        assertTrue(none.output().contains("doesn't run"), none.output());
    }

    static String sha256(Path file) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)));
    }
}
