package org.liberationdata.fornax.jitllm;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

/**
 * The JVM options a worker needs to run jitllm: the TornadoVM SDK's {@code tornado-argfile} (its module path, native
 * library path and module options), the Vector API, and jitllm's own switches. A release SDK that has only
 * {@code tornado-argfile.template} gets one written from it, with the SDK's path in place of {@code ${TORNADOVM_HOME}}. The SDK is {@code -Dfornax.tornadovm.home},
 * else {@code TORNADOVM_HOME}; use one built for this JDK and your GPU's backend (CUDA, OpenCL or Metal).
 * Every {@code -Djitllm.*} and {@code -Dtornado.*} property of this JVM goes to the worker too.
 */
public final class TornadoJvm {
    private TornadoJvm() {}

    /** The TornadoVM SDK, or an IllegalStateException that says how to set one. */
    public static Path home() {
        String h = System.getProperty("fornax.tornadovm.home");
        if (h == null || h.isBlank()) h = System.getenv("TORNADOVM_HOME");
        if (h == null || h.isBlank())
            throw new IllegalStateException("jitllm needs a TornadoVM SDK: set TORNADOVM_HOME (or -Dfornax.tornadovm.home) "
                    + "to one for this JDK and your GPU (https://github.com/beehive-lab/TornadoVM/releases)");
        Path home = Path.of(h).toAbsolutePath().normalize();
        if (!Files.isRegularFile(home.resolve("tornado-argfile")) && !Files.isRegularFile(home.resolve("tornado-argfile.template")))
            throw new IllegalStateException(home + " has no tornado-argfile: is it a TornadoVM SDK (8.0 or newer)?");
        return home;
    }

    /**
     * The SDK's argfile: its own {@code tornado-argfile}, else one written from {@code tornado-argfile.template} (as
     * the SDK's {@code tornado} launcher does) into the temporary directory, as the SDK may not be writable.
     */
    static Path argfile(Path home) {
        Path argfile = home.resolve("tornado-argfile");
        if (Files.isRegularFile(argfile)) return argfile;
        try {
            String text = Files.readString(home.resolve("tornado-argfile.template"), StandardCharsets.UTF_8)
                    .replace("${TORNADOVM_HOME}", home.toString());
            Path dir = Files.createDirectories(Path.of(System.getProperty("java.io.tmpdir"), "fornax-tornadovm"));
            Path out = dir.resolve("tornado-argfile-" + Integer.toHexString(home.toString().hashCode()));
            Path tmp = Files.createTempFile(dir, "argfile", ".tmp");
            Files.writeString(tmp, text, StandardCharsets.UTF_8);
            Files.move(tmp, out, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            return out;
        } catch (IOException e) {
            throw new UncheckedIOException("can't write a tornado-argfile for " + home + ": " + e.getMessage(), e);
        }
    }

    /** The options for a worker JVM, after Fornax's own. */
    public static List<String> workerArgs() {
        List<String> args = new ArrayList<>();
        args.add("@" + argfile(home()));
        args.add("--add-modules");
        args.add("jdk.incubator.vector");
        args.add("-Dtornado.tvm.maxbytecodesize=65536");
        args.add("-Duse.tornadovm=true");
        for (String key : System.getProperties().stringPropertyNames().stream().sorted().toList())
            if (key.startsWith("jitllm.") || key.startsWith("tornado.") && !key.startsWith("tornado.load."))
                args.add("-D" + key + "=" + System.getProperty(key));
        return args;
    }
}
