package org.liberationdata.kiln;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Finds and binds one of Kiln's native libraries.
 *
 * <p>Search order for {@code libNAME.so} (or {@code .dylib}):
 * <ol>
 *   <li>system property {@code kiln.NAME.lib}: an exact file path</li>
 *   <li>each directory in system property {@code kiln.library.path}</li>
 *   <li>each directory in env {@code KILN_LIBRARY_PATH}</li>
 *   <li>the platform loader (LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, system dirs)</li>
 * </ol>
 *
 * <p>Before the first library loads, Kiln sets the engines' tunables that it gives a default to, as environment
 * variables the engines read (an existing environment variable always wins):
 * <ul>
 *   <li>{@code kiln.gpu.graphs} (default {@code false}): ggml's CUDA/HIP graph capture. ROCm schedules a HIP graph
 *   recursively, one stack frame per node, which overflows a JVM thread's stack on large models (LTX), so it is off.
 *   Sets {@code GGML_CUDA_DISABLE_GRAPHS}.</li>
 *   <li>{@code kiln.vulkan.maxNodesPerSubmit} (no default): graph nodes per Vulkan submission. Lower it (8) when long
 *   submissions hit the GPU driver's timeout. Sets {@code GGML_VK_MAX_NODES_PER_SUBMIT}.</li>
 * </ul>
 */
public final class NativeLibrary {
    private static final Linker LINKER = Linker.nativeLinker();

    private final String name;
    private final SymbolLookup lookup;

    private NativeLibrary(String name, SymbolLookup lookup) {
        this.name = name;
        this.lookup = lookup;
    }

    public static NativeLibrary load(String name) {
        Tunables.apply();
        String file = System.mapLibraryName(name);
        String exact = System.getProperty("kiln." + name + ".lib");
        if (exact != null) return new NativeLibrary(name, SymbolLookup.libraryLookup(Path.of(exact), Arena.global()));
        List<String> dirs = new ArrayList<>();
        addPath(dirs, System.getProperty("kiln.library.path"));
        addPath(dirs, System.getenv("KILN_LIBRARY_PATH"));
        for (String d : dirs) {
            Path p = Path.of(d, file);
            if (Files.isRegularFile(p)) return new NativeLibrary(name, SymbolLookup.libraryLookup(p, Arena.global()));
        }
        try {
            return new NativeLibrary(name, SymbolLookup.libraryLookup(file, Arena.global()));
        } catch (IllegalArgumentException e) {
            throw new UnsatisfiedLinkError("Kiln could not find " + file + " (searched " + dirs
                    + " and the platform loader). Set -Dkiln.library.path=<dir> or -Dkiln." + name + ".lib=<file>.");
        }
    }

    /** Engine settings passed as environment variables, set once, before any engine reads them. */
    static final class Tunables {
        private static boolean applied;

        static synchronized void apply() {
            if (applied) return;
            applied = true;
            apply(System.getProperties());
        }

        /** The tunables from these properties (System's, except in tests), each set unless its variable already is. */
        static void apply(java.util.Properties properties) {
            if (!Boolean.parseBoolean(properties.getProperty("kiln.gpu.graphs", "false"))) setDefault("GGML_CUDA_DISABLE_GRAPHS", "1");
            String nodes = properties.getProperty("kiln.vulkan.maxNodesPerSubmit");
            if (nodes != null) setDefault("GGML_VK_MAX_NODES_PER_SUBMIT", nodes);
        }

        /** setenv(name, value, 0): the process environment native code sees, unless it is already set. */
        static void setDefault(String name, String value) {
            if (System.getenv(name) != null) return;
            try (Arena arena = Arena.ofConfined()) {
                MethodHandle setenv = LINKER.downcallHandle(LINKER.defaultLookup().find("setenv").orElseThrow(),
                        FunctionDescriptor.of(java.lang.foreign.ValueLayout.JAVA_INT, java.lang.foreign.ValueLayout.ADDRESS,
                                java.lang.foreign.ValueLayout.ADDRESS, java.lang.foreign.ValueLayout.JAVA_INT));
                int rc = (int) setenv.invokeExact(arena.allocateFrom(name), arena.allocateFrom(value), 0);
                if (rc != 0) throw new IllegalStateException("setenv " + name + " failed");
            } catch (Throwable t) {
                throw new IllegalStateException("could not set " + name, t);
            }
        }

        /** getenv(name) as native code sees it (System.getenv is a snapshot taken at JVM start). */
        static String nativeEnv(String name) {
            try (Arena arena = Arena.ofConfined()) {
                MethodHandle getenv = LINKER.downcallHandle(LINKER.defaultLookup().find("getenv").orElseThrow(),
                        FunctionDescriptor.of(java.lang.foreign.ValueLayout.ADDRESS, java.lang.foreign.ValueLayout.ADDRESS));
                return string((MemorySegment) getenv.invokeExact(arena.allocateFrom(name)));
            } catch (Throwable t) {
                throw new IllegalStateException(t);
            }
        }
    }

    private static void addPath(List<String> dirs, String path) {
        if (path == null || path.isBlank()) return;
        for (String d : path.split(java.io.File.pathSeparator)) if (!d.isBlank()) dirs.add(d);
    }

    public MethodHandle function(String symbol, FunctionDescriptor descriptor) {
        MemorySegment addr = lookup.find(symbol)
                .orElseThrow(() -> new UnsatisfiedLinkError(symbol + " not found in " + name));
        return LINKER.downcallHandle(addr, descriptor);
    }

    public static Linker linker() { return LINKER; }

    /** Reads a NUL-terminated C string, or null for NULL. */
    public static String string(MemorySegment s) {
        return s.equals(MemorySegment.NULL) ? null : s.reinterpret(Long.MAX_VALUE).getString(0);
    }
}
