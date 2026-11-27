package org.liberationdata.fornax.jitllm;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TornadoJvmTest {

    @Test void usesTheSdksOwnArgfile(@TempDir Path sdk) throws Exception {
        Files.writeString(sdk.resolve("tornado-argfile"), "-server\n");
        Files.writeString(sdk.resolve("tornado-argfile.template"), "unused\n");
        assertEquals(sdk.resolve("tornado-argfile"), TornadoJvm.argfile(sdk));
    }

    @Test void writesOneFromAReleaseSdksTemplate(@TempDir Path sdk, @TempDir Path cache) throws Exception {
        String before = System.getProperty("fornax.cache.dir");
        System.setProperty("fornax.cache.dir", cache.toString());
        try { writesOneFromTheTemplate(sdk, cache); }
        finally { if (before == null) System.clearProperty("fornax.cache.dir"); else System.setProperty("fornax.cache.dir", before); }
    }

    private static void writesOneFromTheTemplate(Path sdk, Path cache) throws Exception {
        Files.writeString(sdk.resolve("tornado-argfile.template"),
                "-Djava.library.path=${TORNADOVM_HOME}/lib\n--module-path .:${TORNADOVM_HOME}/share/java/tornado\n");
        Path argfile = TornadoJvm.argfile(sdk);
        assertEquals(cache.resolve("tornadovm"), argfile.getParent());
        if (Files.getFileStore(cache).supportsFileAttributeView("posix"))
            assertEquals("rwx------", java.nio.file.attribute.PosixFilePermissions.toString(Files.getPosixFilePermissions(argfile.getParent())));
        assertEquals("-Djava.library.path=" + sdk + "/lib\n--module-path .:" + sdk + "/share/java/tornado\n",
                Files.readString(argfile));
    }

    @Test void quotesAnSdkPathWithSpaces(@TempDir Path dir) throws Exception {
        Path sdk = Files.createDirectories(dir.resolve("Tornado SDKs").resolve("tornadovm"));
        String argfile = TornadoJvm.expand("# a comment with ${TORNADOVM_HOME}\n-server\n-Dfornax.sdk.lib=${TORNADOVM_HOME}/lib\n", sdk);
        assertEquals("# a comment with ${TORNADOVM_HOME}\n-server\n\"-Dfornax.sdk.lib=" + sdk + "/lib\"\n", argfile);
        // and the JVM reads it back as one option
        Path file = Files.writeString(dir.resolve("args"), argfile);
        Process p = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(), "@" + file,
                "-XshowSettings:properties", "-version").redirectErrorStream(true).start();
        String out = new String(p.getInputStream().readAllBytes());
        assertEquals(0, p.waitFor(), out);
        assertTrue(out.contains("fornax.sdk.lib = " + sdk + "/lib"), out);
    }
}
