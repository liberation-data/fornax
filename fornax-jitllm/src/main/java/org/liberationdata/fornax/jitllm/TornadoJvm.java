package org.liberationdata.fornax.jitllm;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * The JVM options a worker needs to run jitllm: the TornadoVM SDK's {@code tornado-argfile} (its module path, native
 * library path and module options), the Vector API, and jitllm's own switches. The SDK is {@code -Dfornax.tornadovm.home},
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
        Path home = Path.of(h);
        if (!Files.isRegularFile(home.resolve("tornado-argfile")))
            throw new IllegalStateException(home + " has no tornado-argfile: is it a TornadoVM SDK (8.0 or newer)?");
        return home;
    }

    /** The options for a worker JVM, after Fornax's own. */
    public static List<String> workerArgs() {
        List<String> args = new ArrayList<>();
        args.add("@" + home().resolve("tornado-argfile"));
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
