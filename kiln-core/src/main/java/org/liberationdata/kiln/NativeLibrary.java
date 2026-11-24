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
