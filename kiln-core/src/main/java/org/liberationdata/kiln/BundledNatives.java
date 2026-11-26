package org.liberationdata.kiln;

import java.io.IOException;
import java.io.InputStream;
import java.net.JarURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.jar.JarEntry;

/**
 * Native libraries packaged in Kiln's natives jars ({@code org.kiln4j:kiln-natives-<os>-<arch>-<backend>}), found
 * on the classpath and extracted to a cache directory, since a shared library can't be loaded from inside a jar.
 *
 * <p>A natives jar holds its libraries under {@code org/kiln4j/natives/<os>-<arch>-<backend>/}, for example
 * {@code org/kiln4j/natives/linux-x86_64-vulkan/libkiln_llama.so}. When jars for several backends are on the
 * classpath, the one named by {@code -Dkiln.backend} or {@code KILN_BACKEND} is used, else the first of
 * {@link #BACKENDS}. Libraries are extracted to {@code <cache>/<kiln version>/<os>-<arch>-<backend>/}, where
 * {@code <cache>} is {@code -Dkiln.cache.dir} or {@code KILN_CACHE_DIR}, else the platform's user cache directory
 * ({@code ~/Library/Caches/kiln} on macOS, {@code $XDG_CACHE_HOME/kiln} or {@code ~/.cache/kiln} elsewhere).
 */
final class BundledNatives {
    static final String ROOT = "org/kiln4j/natives/";

    /** Backends in order of preference when more than one is on the classpath: Metal is macOS only. */
    static final List<String> BACKENDS = List.of("metal", "cuda", "hip", "vulkan", "cpu");

    private BundledNatives() {}

    /** The extracted file for library {@code name} from a natives jar on the classpath, or empty if there is none. */
    static Optional<Path> find(String name) {
        ClassLoader loader = loader();
        String file = System.mapLibraryName(name);
        String platform = platform(System.getProperty("os.name"), System.getProperty("os.arch"));
        List<String> available = available(loader, platform, file);
        if (available.isEmpty()) return Optional.empty();
        String backend = choose(available, requestedBackend(), file);
        String dir = platform + "-" + backend;
        try {
            Path lib = extract(loader.getResource(ROOT + dir + "/" + file), cacheRoot().resolve(version()).resolve(dir), file);
            if (verbose()) System.err.println("kiln: " + name + " from " + dir + " natives, at " + lib);
            return Optional.of(lib);
        } catch (IOException e) {
            UnsatisfiedLinkError err = new UnsatisfiedLinkError("Kiln could not extract " + file + " (" + dir + ") to "
                    + cacheRoot() + ". Set -Dkiln.cache.dir=<dir> to extract it somewhere writable.");
            err.initCause(e);
            throw err;
        }
    }

    /** {@code <os>-<arch>} as natives jars name it: {@code linux-x86_64}, {@code macos-aarch64}, ... */
    static String platform(String osName, String osArch) {
        String n = osName.toLowerCase(Locale.ROOT);
        String os = n.contains("mac") || n.contains("darwin") ? "macos" : n.contains("win") ? "windows" : n.contains("linux") ? "linux" : n.replace(' ', '_');
        String a = osArch.toLowerCase(Locale.ROOT);
        String arch = switch (a) {
            case "amd64", "x86_64", "x64" -> "x86_64";
            case "aarch64", "arm64" -> "aarch64";
            default -> a;
        };
        return os + "-" + arch;
    }

    /** The backends whose build of {@code file} is on the classpath for {@code platform}, in order of preference. */
    static List<String> available(ClassLoader loader, String platform, String file) {
        List<String> found = new ArrayList<>();
        for (String b : BACKENDS) if (loader.getResource(ROOT + platform + "-" + b + "/" + file) != null) found.add(b);
        return found;
    }

    /** The backend to load: {@code requested} if given (it must be available), else the most preferred. */
    static String choose(List<String> available, String requested, String file) {
        if (requested == null || requested.isBlank()) return available.getFirst();
        String r = requested.trim().toLowerCase(Locale.ROOT);
        if (!available.contains(r)) throw new UnsatisfiedLinkError("Kiln backend '" + r + "' was asked for (kiln.backend or KILN_BACKEND), but the natives on the classpath have "
                + file + " only for " + available + ". Add the kiln-natives jar for " + r + ", or ask for one of those.");
        return r;
    }

    /**
     * Copies {@code resource} to {@code dir/file}, unless the copy there already matches it. A copy matches when the
     * stamp written beside it ({@code file.stamp}: the jar entry's size and CRC) is the resource's. The copy is
     * written to a temporary file and moved into place, so another JVM never sees a part-written library.
     */
    static Path extract(URL resource, Path dir, String file) throws IOException {
        URLConnection c = resource.openConnection();
        c.setUseCaches(false);
        String stamp = stamp(c);
        Path target = dir.resolve(file), stampFile = dir.resolve(file + ".stamp");
        if (Files.isRegularFile(target) && Files.isRegularFile(stampFile) && Files.readString(stampFile).equals(stamp)) {
            return target;
        }
        Files.createDirectories(dir);
        Path tmp = Files.createTempFile(dir, file, ".part");
        try {
            try (InputStream in = c.getInputStream()) {
                Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            }
            move(tmp, target);
            Files.writeString(tmp, stamp);
            move(tmp, stampFile);
        } finally {
            Files.deleteIfExists(tmp);
        }
        return target;
    }

    /** What identifies a resource's content without reading it: size and CRC in a jar, size and time in a directory. */
    private static String stamp(URLConnection c) throws IOException {
        if (c instanceof JarURLConnection jar) {
            JarEntry e = jar.getJarEntry();
            return e.getSize() + " " + Long.toHexString(e.getCrc());
        }
        return c.getContentLengthLong() + " " + c.getLastModified();
    }

    private static void move(Path from, Path to) throws IOException {
        try {
            Files.move(from, to, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException e) {
            Files.move(from, to, StandardCopyOption.REPLACE_EXISTING);
        }
    }

    /** Where libraries are extracted to, before the version and platform directories. */
    static Path cacheRoot() {
        String dir = setting("kiln.cache.dir", "KILN_CACHE_DIR");
        if (dir != null) return Path.of(dir);
        String home = System.getProperty("user.home");
        String os = platform(System.getProperty("os.name"), System.getProperty("os.arch"));
        if (os.startsWith("macos-")) return Path.of(home, "Library", "Caches", "kiln");
        if (os.startsWith("windows-")) {
            String local = System.getenv("LOCALAPPDATA");
            return local != null && !local.isBlank() ? Path.of(local, "kiln", "cache") : Path.of(home, "AppData", "Local", "kiln", "cache");
        }
        String xdg = System.getenv("XDG_CACHE_HOME");
        return xdg != null && !xdg.isBlank() ? Path.of(xdg, "kiln") : Path.of(home, ".cache", "kiln");
    }

    /** Kiln's version, from kiln-core's jar manifest ("dev" when run from classes, as in Kiln's own build). */
    static String version() {
        String v = BundledNatives.class.getPackage().getImplementationVersion();
        return v != null && !v.isBlank() ? v : "dev";
    }

    static String requestedBackend() { return setting("kiln.backend", "KILN_BACKEND"); }

    private static String setting(String property, String env) {
        String v = System.getProperty(property);
        if (v == null || v.isBlank()) v = System.getenv(env);
        return v == null || v.isBlank() ? null : v;
    }

    private static boolean verbose() {
        String v = System.getenv("KILN_VERBOSE");
        return v != null && !v.isBlank() && !v.equals("0");
    }

    private static ClassLoader loader() {
        ClassLoader l = Thread.currentThread().getContextClassLoader();
        if (l == null) l = BundledNatives.class.getClassLoader();
        return l != null ? l : ClassLoader.getSystemClassLoader();
    }
}
