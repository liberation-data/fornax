package org.liberationdata.fornax.cli;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

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
    static final Path ZIP =
            Path.of(System.getProperty("fornax.cli.distZip", "build/distributions/fornax-cli-" + VERSION + ".zip"));

    @TempDir Path tmp;
    Path home;
    Path dist;

    record Run(int code, String output) {}

    @BeforeEach
    void release() throws Exception {
        home = Files.createDirectories(tmp.resolve("home"));
        dist = Files.createDirectories(tmp.resolve("release"));
        publish(ZIP);
    }

    /** Puts zip in the release under each platform's name, so a test runs on either kind of machine the script supports. */
    void publish(Path zip) throws Exception {
        StringBuilder sums = new StringBuilder();
        for (String platform : PLATFORMS) {
            String name = "fornax-cli-" + VERSION + "-" + platform + ".zip";
            Files.copy(zip, dist.resolve(name), StandardCopyOption.REPLACE_EXISTING);
            sums.append(sha256(zip)).append("  ").append(name).append('\n');
        }
        Files.writeString(dist.resolve("SHA256SUMS"), sums);
    }

    /** An executable shell script at file, to stand in for a command (java, uname, fornax). */
    static Path script(Path file, String body) throws Exception {
        Files.createDirectories(file.getParent());
        Files.writeString(file, "#!/bin/sh\n" + body);
        Files.setPosixFilePermissions(file, PosixFilePermissions.fromString("rwxr-xr-x"));
        return file;
    }

    /** What is in folder: in the data folder after a refused install, nothing (no version, no work folder). */
    List<Path> leftIn(Path folder) throws Exception {
        try (Stream<Path> files = Files.list(folder)) { return files.toList(); }
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
        assertEquals(List.of(), leftIn(data()), "something was left in " + data());
    }

    @Test
    void refusesAReleaseWithNoSha256Sums() throws Exception {
        Files.delete(dist.resolve("SHA256SUMS"));
        Run r = install();
        assertEquals(1, r.code(), r.output());
        assertTrue(r.output().contains("no SHA256SUMS at file://" + dist), r.output());
        assertTrue(r.output().contains("Releases before 0.3.0"), r.output());
        assertEquals(List.of(), leftIn(data()));
    }

    @Test
    void refusesAReleaseWithNoZipForThisPlatform() throws Exception {
        String sums = Files.readString(dist.resolve("SHA256SUMS"));
        for (String platform : PLATFORMS) sums = sums.replace("-" + platform + ".zip", "-windows-x86_64-cuda.zip");
        Files.writeString(dist.resolve("SHA256SUMS"), sums);
        Run r = install();
        assertEquals(1, r.code(), r.output());
        assertTrue(r.output().contains("SHA256SUMS lists none"), r.output());
        assertEquals(List.of(), leftIn(data()));
    }

    @Test
    void refusesAZipWithoutTheCommandInIt() throws Exception {
        Path zip = tmp.resolve("no-bin.zip");
        try (var out = new ZipOutputStream(Files.newOutputStream(zip))) {
            out.putNextEntry(new ZipEntry("fornax-cli-" + VERSION + "/lib/fornax-cli-" + VERSION + ".jar"));
            out.closeEntry();
        }
        publish(zip);
        Run r = install();
        assertEquals(1, r.code(), r.output());
        assertTrue(r.output().contains("doesn't hold fornax-cli-" + VERSION + "/bin/fornax"), r.output());
        assertFalse(Files.exists(link()) || Files.isSymbolicLink(link()));
        assertEquals(List.of(), leftIn(data()));
    }

    @Test
    void refusesACurrentThatIsntALink() throws Exception {
        Files.createDirectories(data().resolve("current"));
        Run r = install();
        assertEquals(1, r.code(), r.output());
        assertTrue(r.output().contains("current is there and isn't a link"), r.output());
        assertEquals(List.of(data().resolve("current")), leftIn(data()));
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
    void installsUnderXdgDataHome() throws Exception {
        Path xdg = tmp.resolve("xdg");
        Run r = install(Map.of("XDG_DATA_HOME", xdg.toString()));
        assertEquals(0, r.code(), r.output());
        assertEquals(xdg.toRealPath().resolve("fornax/current/bin/fornax"), Files.readSymbolicLink(link()));
        assertFalse(Files.exists(data()));
    }

    @Test
    void saysWhenAnotherFornaxComesFirstOnPath() throws Exception {
        Path other = script(tmp.resolve("other/fornax"), "echo studio\n");
        Run r = install(Map.of("PATH", other.getParent() + ":" + link().getParent() + ":" + System.getenv("PATH")));
        assertEquals(0, r.code(), r.output());
        assertTrue(r.output().contains("Another fornax comes first on PATH: " + other), r.output());
        assertFalse(r.output().contains("export PATH"), r.output());
        assertTrue(Files.isSymbolicLink(link()));
    }

    @Test
    void picksTheNativesByUname() throws Exception {
        // uname (and sysctl, which says whether a Mac's shell runs under Rosetta) stood in for, first on PATH
        Path fake = tmp.resolve("fake");
        script(fake.resolve("uname"), "case \"$1\" in -s) echo \"$FAKE_OS\" ;; -m) echo \"$FAKE_ARCH\" ;; esac\n");
        script(fake.resolve("sysctl"), "echo \"$FAKE_ROSETTA\"\n");
        String path = fake + ":" + System.getenv("PATH");

        Run arm = install(Map.of("PATH", path, "FAKE_OS", "Linux", "FAKE_ARCH", "aarch64"));
        assertEquals(1, arm.code(), arm.output());
        assertTrue(arm.output().contains("no natives for Linux aarch64"), arm.output());
        Run intel = install(Map.of("PATH", path, "FAKE_OS", "Darwin", "FAKE_ARCH", "x86_64", "FAKE_ROSETTA", "0"));
        assertEquals(1, intel.code(), intel.output());
        assertTrue(intel.output().contains("no natives for Intel Macs"), intel.output());
        assertFalse(Files.exists(link()) || Files.isSymbolicLink(link()));

        Run rosetta = install(Map.of("PATH", path, "FAKE_OS", "Darwin", "FAKE_ARCH", "x86_64", "FAKE_ROSETTA", "1"));
        assertEquals(0, rosetta.code(), rosetta.output());
        assertTrue(rosetta.output().contains("for macos-aarch64-metal"), rosetta.output());
    }

    @Test
    void saysHowToGetJavaWhenItsTooOld() throws Exception {
        Path java = script(tmp.resolve("java"), "echo 'openjdk version \"17.0.2\" 2022-01-18' >&2\n");
        Run r = install(Map.of("FORNAX_JAVA", java.toString()));
        assertEquals(0, r.code(), r.output());  // fornax is installed all the same
        assertTrue(r.output().contains("is Java 17"), r.output());
        assertTrue(r.output().contains("sdk install java"), r.output());

        // Java 8 says "1.8.0_..."
        Path java8 = script(tmp.resolve("java8"), "echo 'openjdk version \"1.8.0_292\"' >&2\n");
        Run eight = install(Map.of("FORNAX_JAVA", java8.toString()));
        assertEquals(0, eight.code(), eight.output());
        assertTrue(eight.output().contains("is Java 8 ("), eight.output());

        Run none = install(Map.of("FORNAX_JAVA", tmp.resolve("nowhere/java").toString()));
        assertEquals(0, none.code(), none.output());
        assertTrue(none.output().contains("doesn't run"), none.output());
    }

    @Test
    void refusesAZipSha256SumsListsButTheReleaseHasnt() throws Exception {
        Files.delete(dist.resolve("fornax-cli-" + VERSION + "-linux-x86_64-vulkan.zip"));
        Files.delete(dist.resolve("fornax-cli-" + VERSION + "-macos-aarch64-metal.zip"));
        Run r = install();
        assertEquals(1, r.code(), r.output());
        assertTrue(r.output().contains("couldn't download file://" + dist + "/fornax-cli-" + VERSION + "-"), r.output());
        assertEquals(List.of(), leftIn(data()));
    }

    @Test
    void refusesANameOrChecksumInSha256SumsThatIsntOne() throws Exception {
        // The version in the zip's name becomes a folder name, so a name like this one must go no further
        String sums = Files.readString(dist.resolve("SHA256SUMS"))
                .replace("fornax-cli-" + VERSION + "-", "fornax-cli-../x-");
        Files.writeString(dist.resolve("SHA256SUMS"), sums);
        Run name = install();
        assertEquals(1, name.code(), name.output());
        assertTrue(name.output().contains("not a version: ../x"), name.output());

        publish(ZIP);
        sums = Files.readString(dist.resolve("SHA256SUMS")).replaceAll("(?m)^[0-9a-f]{64}", "abc");
        Files.writeString(dist.resolve("SHA256SUMS"), sums);
        Run sum = install();
        assertEquals(1, sum.code(), sum.output());
        assertTrue(sum.output().contains("not a SHA-256 for fornax-cli-" + VERSION), sum.output());
        assertEquals(List.of(), leftIn(data()));
    }

    @Test
    void downloadsFromTheReleaseAskedForOrTheLatest() throws Exception {
        // curl stood in for: it says what it was asked for, and fails as it does on a 404
        Path log = tmp.resolve("curl.log");
        Path fake = script(tmp.resolve("fake/curl"), "printf '%s\\n' \"$@\" >> \"$FAKE_LOG\"; exit 22\n").getParent();
        Map<String, String> env = new HashMap<>(Map.of("FORNAX_DIST_URL", "", "FAKE_LOG", log.toString()));
        env.put("PATH", fake + ":" + System.getenv("PATH"));

        Run latest = install(env);
        assertEquals(1, latest.code(), latest.output());
        String repo = "https://github.com/liberation-data/fornax/releases/";
        assertTrue(Files.readAllLines(log).contains(repo + "latest/download/SHA256SUMS"), Files.readString(log));
        assertTrue(latest.output().contains("Releases before 0.3.0"), latest.output());

        Files.delete(log);
        env.put("FORNAX_VERSION", "v0.3.0");
        Run asked = install(env);
        assertEquals(1, asked.code(), asked.output());
        List<String> args = Files.readAllLines(log);
        assertTrue(args.contains(repo + "download/v0.3.0/SHA256SUMS"), args.toString());
        assertTrue(args.contains("=https"), "curl may leave https: " + args);
    }

    @Test
    void refusesAWgetThatCantKeepToHttps() throws Exception {
        // A PATH with every command this one has but curl, and BusyBox's wget, whose options have no --https-only
        Path bin = Files.createDirectories(tmp.resolve("no-curl"));
        for (String dir : System.getenv("PATH").split(":")) {
            if (!Files.isDirectory(Path.of(dir))) continue;
            try (Stream<Path> commands = Files.list(Path.of(dir))) {
                for (Path c : (Iterable<Path>) commands::iterator) {
                    String name = c.getFileName().toString();
                    Path link = bin.resolve(name);
                    boolean taken = Files.exists(link, LinkOption.NOFOLLOW_LINKS);
                    if (!name.equals("curl") && !name.equals("wget") && !taken) {
                        Files.createSymbolicLink(link, c);
                    }
                }
            }
        }
        script(bin.resolve("wget"),
                "echo 'BusyBox v1.36.1 multi-call binary.'; echo 'Usage: wget [-cqS] [--spider] [-O FILE] URL...'; exit 1\n");
        Run r = install(Map.of("PATH", bin.toString(), "FORNAX_DIST_URL", "https://fornax4j.invalid"));
        assertEquals(1, r.code(), r.output());
        assertTrue(r.output().contains("needs curl, or a wget with --https-only"), r.output());
        assertFalse(r.output().contains("no SHA256SUMS"), r.output());
        assertEquals(List.of(), leftIn(data()));
    }

    static String sha256(Path file) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)));
    }
}
