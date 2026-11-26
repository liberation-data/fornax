package org.liberationdata.kiln;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.SymbolLookup;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

/**
 * Linux checks for an engine built for Vulkan, which needs the Vulkan loader ({@code libvulkan.so.1}) and a driver.
 *
 * <p>Without the loader the engine library can't load at all, and {@link #explain} turns that into an error that
 * says what to install. Without a driver it loads, but ggml finds no Vulkan device and the engines run on the CPU,
 * much slower and with no error; {@link #warnIfNoDriver} logs a warning instead. A driver is found by its manifest
 * ({@code *.json}) in the Vulkan loader's search directories.
 */
final class VulkanDriver {
    static final String LOADER = "libvulkan.so.1";
    static final String DOCS = "https://kiln4j.org/docs/guides/gpu-backends/#the-vulkan-driver-on-linux";
    static final String INSTALL = String.join("\n  ",
            "Fedora:          sudo dnf install vulkan-loader mesa-vulkan-drivers",
            "Ubuntu, Debian:  sudo apt install libvulkan1 mesa-vulkan-drivers",
            "Arch:            sudo pacman -S vulkan-icd-loader vulkan-radeon   (or vulkan-intel)",
            "NVIDIA's own driver brings its Vulkan driver with it. More: " + DOCS);

    private static boolean warned;

    private VulkanDriver() {}

    /** Whether the library at {@code lib} links the Vulkan loader: its name is in the ELF's dynamic string table. */
    static boolean needsLoader(Path lib) {
        byte[] needle = LOADER.getBytes(StandardCharsets.US_ASCII);
        byte[] buf = new byte[1 << 20];
        int kept = 0;                                  // the tail of the last chunk, for a name split across chunks
        try (var in = Files.newInputStream(lib)) {
            for (int n; (n = in.read(buf, kept, buf.length - kept)) > 0; ) {
                int end = kept + n;
                outer:
                for (int i = 0; i + needle.length <= end; i++) {
                    for (int j = 0; j < needle.length; j++) if (buf[i + j] != needle[j]) continue outer;
                    return true;
                }
                kept = Math.min(needle.length - 1, end);
                System.arraycopy(buf, end - kept, buf, 0, kept);
            }
            return false;
        } catch (IOException e) {
            return false;
        }
    }

    /**
     * Throws an error that names the missing Vulkan loader when that's why the library at {@code lib} failed to open
     * ({@code failure}); returns otherwise, for the caller to rethrow {@code failure}.
     */
    static void explain(Path lib, IllegalArgumentException failure, Predicate<String> loadable) {
        if (!isLinux() || !needsLoader(lib) || loadable.test(LOADER)) return;
        UnsatisfiedLinkError e = new UnsatisfiedLinkError("Kiln could not load " + lib.getFileName() + ": it runs on Vulkan, "
                + "and this machine has no Vulkan loader (" + LOADER + "). Install it and a Vulkan driver:\n  " + INSTALL);
        e.initCause(failure);
        throw e;
    }

    static void explain(Path lib, IllegalArgumentException failure) {
        explain(lib, failure, VulkanDriver::loadable);
    }

    /** Logs a warning, once, when {@code lib} runs on Vulkan and no Vulkan driver is installed. */
    static synchronized void warnIfNoDriver(Path lib) {
        if (warned || !isLinux() || !needsLoader(lib)) return;
        warned = true;
        if (!hasDriver(System.getenv(), Path.of(System.getProperty("user.home")))) {
            System.getLogger(VulkanDriver.class.getName()).log(System.Logger.Level.WARNING, noDriverMessage(lib));
        }
    }

    static String noDriverMessage(Path lib) {
        return "Kiln found no Vulkan driver, so " + lib.getFileName() + " will run on the CPU (much slower). Install one:\n  " + INSTALL;
    }

    /** Whether a Vulkan driver is installed: an override in the environment, or a manifest where the loader looks. */
    static boolean hasDriver(Map<String, String> env, Path home) {
        for (String v : List.of("VK_DRIVER_FILES", "VK_ICD_FILENAMES", "VK_ADD_DRIVER_FILES")) {
            if (env.get(v) != null && !env.get(v).isBlank()) return true;
        }
        for (Path dir : manifestDirs(env, home)) {
            try (var files = Files.list(dir)) {
                if (files.anyMatch(f -> f.getFileName().toString().endsWith(".json"))) return true;
            } catch (IOException | RuntimeException e) {
                // not there, or not readable: the loader would skip it too
            }
        }
        return false;
    }

    /** Where the Vulkan loader looks for driver manifests on Linux, in its order. */
    static List<Path> manifestDirs(Map<String, String> env, Path home) {
        List<Path> dirs = new ArrayList<>();
        String configHome = env.getOrDefault("XDG_CONFIG_HOME", home.resolve(".config").toString());
        String configDirs = env.getOrDefault("XDG_CONFIG_DIRS", "/etc/xdg");
        String dataHome = env.getOrDefault("XDG_DATA_HOME", home.resolve(".local/share").toString());
        String dataDirs = env.getOrDefault("XDG_DATA_DIRS", "/usr/local/share:/usr/share");
        dirs.add(Path.of(configHome, "vulkan/icd.d"));
        for (String d : configDirs.split(":")) if (!d.isBlank()) dirs.add(Path.of(d, "vulkan/icd.d"));
        dirs.add(Path.of("/etc/vulkan/icd.d"));
        dirs.add(Path.of(dataHome, "vulkan/icd.d"));
        for (String d : dataDirs.split(":")) if (!d.isBlank()) dirs.add(Path.of(d, "vulkan/icd.d"));
        return dirs;
    }

    private static boolean loadable(String name) {
        try {
            SymbolLookup.libraryLookup(name, Arena.global());
            return true;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    private static boolean isLinux() {
        return System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).startsWith("linux");
    }
}
