package org.liberationdata.fornax;

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
import java.util.Optional;

/**
 * Finds and binds one of Fornax's native libraries.
 *
 * <p>Search order for {@code libNAME.so} (or {@code .dylib}):
 * <ol>
 *   <li>system property {@code fornax.NAME.lib}: an exact file path</li>
 *   <li>each directory in system property {@code fornax.library.path}</li>
 *   <li>each directory in env {@code FORNAX_LIBRARY_PATH}</li>
 *   <li>a Fornax natives jar on the classpath ({@code org.fornax4j:fornax-natives-<os>-<arch>-<backend>}), extracted to
 *   a per-version cache directory; with several backends there, {@code -Dfornax.backend} or {@code FORNAX_BACKEND}
 *   picks one, else Metal, CUDA, HIP, Vulkan, CPU in that order. Set {@code -Dfornax.cache.dir} (or
 *   {@code FORNAX_CACHE_DIR}) to move the cache; by default it is {@code ~/.cache/fornax} or {@code ~/Library/Caches/fornax}</li>
 *   <li>the platform loader (LD_LIBRARY_PATH, DYLD_LIBRARY_PATH, system dirs)</li>
 * </ol>
 *
 * <p>On Linux, a library built for a GPU (Vulkan, HIP, CUDA) refuses to start on a machine where that GPU runtime
 * finds no GPU, since its engine would otherwise run on the CPU without a word, and says what to install. (Not one the
 * platform loader finds: Fornax doesn't know its file, so can't tell what it was built for.) Set
 * {@code -Dfornax.allowCpu=true} (or {@code FORNAX_ALLOW_CPU=true}) to run on the CPU anyway. See {@link GpuCheck}.
 *
 * <p>Before the first library loads, Fornax sets the engines' tunables that it gives a default to, as environment
 * variables the engines read (an existing environment variable always wins):
 * <ul>
 *   <li>{@code fornax.gpu.graphs} (default {@code false}): ggml's CUDA/HIP graph capture. ROCm schedules a HIP graph
 *   recursively, one stack frame per node, which overflows a JVM thread's stack on large models (LTX), so it is off.
 *   Sets {@code GGML_CUDA_DISABLE_GRAPHS}.</li>
 *   <li>{@code fornax.vulkan.maxNodesPerSubmit} (no default): graph nodes per Vulkan submission. Lower it (8) when long
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
        Optional<Path> found = locate(name);
        if (found.isPresent()) return open(name, found.get());
        List<String> dirs = searchDirs();
        Optional<Path> bundled = BundledNatives.find(name);
        if (bundled.isPresent()) return open(name, bundled.get());
        try {
            return new NativeLibrary(name, SymbolLookup.libraryLookup(file, Arena.global()));
        } catch (IllegalArgumentException e) {
            throw new UnsatisfiedLinkError("Fornax could not find " + file + " (searched " + dirs + ", the classpath for a "
                    + "fornax-natives jar, and the platform loader). Add the org.fornax4j:fornax-natives-<os>-<arch>-<backend> "
                    + "dependency, or set -Dfornax.library.path=<dir> or -Dfornax." + name + ".lib=<file>.");
        }
    }

    /** Opens the library file at {@code path}, and on Linux makes sure a GPU build has a GPU ({@link GpuCheck}). */
    private static NativeLibrary open(String name, Path path) {
        SymbolLookup lookup;
        try {
            lookup = SymbolLookup.libraryLookup(path, Arena.global());
        } catch (IllegalArgumentException e) {
            GpuCheck.explainLoadFailure(path, e);
            throw e;
        }
        GpuCheck.requireGpu(path);
        return new NativeLibrary(name, lookup);
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
        static void apply(java.util.Properties properties) { apply(properties, Tunables::setDefault); }

        /** The variables these properties ask for, handed to set (setDefault, or a recorder in tests). */
        static void apply(java.util.Properties properties, java.util.function.BiConsumer<String, String> set) {
            if (!Boolean.parseBoolean(properties.getProperty("fornax.gpu.graphs", "false"))) set.accept("GGML_CUDA_DISABLE_GRAPHS", "1");
            String nodes = properties.getProperty("fornax.vulkan.maxNodesPerSubmit");
            if (nodes != null) set.accept("GGML_VK_MAX_NODES_PER_SUBMIT", nodes);
        }

        /**
         * setenv(name, value, 0): the process environment native code sees, unless it is already set. Windows' C
         * runtime has no setenv: there it's _putenv_s in ucrtbase, the runtime the engines' DLLs share.
         */
        static void setDefault(String name, String value) {
            if (System.getenv(name) != null) return;
            try (Arena arena = Arena.ofConfined()) {
                int rc;
                if (WINDOWS) {
                    MethodHandle putenv = LINKER.downcallHandle(crt().find("_putenv_s").orElseThrow(),
                            FunctionDescriptor.of(java.lang.foreign.ValueLayout.JAVA_INT, java.lang.foreign.ValueLayout.ADDRESS,
                                    java.lang.foreign.ValueLayout.ADDRESS));
                    rc = (int) putenv.invokeExact(arena.allocateFrom(name), arena.allocateFrom(value));
                } else {
                    MethodHandle setenv = LINKER.downcallHandle(crt().find("setenv").orElseThrow(),
                            FunctionDescriptor.of(java.lang.foreign.ValueLayout.JAVA_INT, java.lang.foreign.ValueLayout.ADDRESS,
                                    java.lang.foreign.ValueLayout.ADDRESS, java.lang.foreign.ValueLayout.JAVA_INT));
                    rc = (int) setenv.invokeExact(arena.allocateFrom(name), arena.allocateFrom(value), 0);
                }
                if (rc != 0) throw new IllegalStateException("setting " + name + " failed (" + rc + ")");
            } catch (Throwable t) {
                throw new IllegalStateException("could not set " + name, t);
            }
        }

        private static final boolean WINDOWS = System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).startsWith("windows");

        /** The C runtime's functions: the platform's default lookup, or on Windows the UCRT the engines link. */
        private static SymbolLookup crt() {
            return WINDOWS ? SymbolLookup.libraryLookup("ucrtbase", Arena.global()) : LINKER.defaultLookup();
        }

        /** getenv(name) as native code sees it (System.getenv is a snapshot taken at JVM start). */
        static String nativeEnv(String name) {
            try (Arena arena = Arena.ofConfined()) {
                MethodHandle getenv = LINKER.downcallHandle(crt().find("getenv").orElseThrow(),
                        FunctionDescriptor.of(java.lang.foreign.ValueLayout.ADDRESS, java.lang.foreign.ValueLayout.ADDRESS));
                return string((MemorySegment) getenv.invokeExact(arena.allocateFrom(name)));
            } catch (Throwable t) {
                throw new IllegalStateException(t);
            }
        }
    }

    /**
     * Where {@link #load} would find {@code libNAME} through Fornax's own settings (steps 1 to 3), without loading it.
     * Empty means {@code load} goes on to a natives jar and then the platform loader.
     */
    public static Optional<Path> locate(String name) {
        String exact = System.getProperty("fornax." + name + ".lib");
        if (exact != null) return Optional.of(Path.of(exact));
        String file = System.mapLibraryName(name);
        for (String d : searchDirs()) {
            Path p = Path.of(d, file);
            if (Files.isRegularFile(p)) return Optional.of(p);
        }
        return Optional.empty();
    }

    /** The directories from {@code fornax.library.path} and {@code FORNAX_LIBRARY_PATH}, in search order. */
    public static List<String> searchDirs() {
        List<String> dirs = new ArrayList<>();
        addPath(dirs, System.getProperty("fornax.library.path"));
        addPath(dirs, System.getenv("FORNAX_LIBRARY_PATH"));
        return dirs;
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
