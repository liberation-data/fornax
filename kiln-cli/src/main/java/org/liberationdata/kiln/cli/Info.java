package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.NativeLibrary;

import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/** {@code kiln info}: where Kiln looks for its native libraries and models. Loads nothing. */
final class Info {
    static final String USAGE = """
            usage: kiln info

            Shows the Java runtime, which native libraries Kiln finds (and their GPU backend), and the
            Hugging Face cache it reads and writes. Nothing is loaded onto the GPU.""";

    /** The engines, by the native library name each module loads. */
    static final List<String[]> LIBRARIES = List.of(
            new String[]{"kiln_llama", "llama.cpp (kiln chat, language models)"},
            new String[]{"stable-diffusion", "stable-diffusion.cpp (images)"},
            new String[]{"kiln_acestep", "acestep.cpp (ACE-Step songs)"},
            new String[]{"kiln_yue2", "yue2.cpp (YuE2 songs)"});

    static int run(HubCache cache, Hub hub, PrintStream out) {
        String v = Info.class.getPackage().getImplementationVersion();
        out.println("Kiln      " + (v == null ? "(development build)" : v));
        out.println("Java      " + System.getProperty("java.runtime.version") + " (" + System.getProperty("java.vendor")
                + ") " + System.getProperty("java.home"));
        out.println("Platform  " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
        List<String> dirs = NativeLibrary.searchDirs();
        out.println("Natives   searched " + (dirs.isEmpty() ? "(no kiln.library.path or KILN_LIBRARY_PATH)" : String.join(", ", dirs))
                + " then the platform loader");
        String backend = null;
        int found = 0;
        for (String[] lib : LIBRARIES) {
            Optional<Path> p = NativeLibrary.locate(lib[0]);
            if (p.isPresent() && Files.isRegularFile(p.get())) {
                found++;
                if (backend == null) backend = backend(p.get());
                out.printf("  %-18s %s (%s)%n", lib[0], p.get(), Fmt.bytes(p.get().toFile().length()));
            } else {
                out.printf("  %-18s not found in those folders: %s%n", lib[0], lib[1]);
            }
        }
        out.println("Backend   " + (backend != null ? backend : found == 0 ? "unknown (no libraries found)" : "unknown")
                + (found == 0 ? "\n          build them with native/build.sh, then set -Dkiln.library.path or KILN_LIBRARY_PATH" : ""));
        List<String> repos = cache.repos();
        long bytes = repos.stream().flatMap(r -> cache.files(r).stream()).mapToLong(HubCache.CachedFile::size).sum();
        out.println("Cache     " + cache.root() + " (" + source() + ")"
                + (Files.isDirectory(cache.root()) ? ": " + repos.size() + " repos, " + Fmt.bytes(bytes) : ": doesn't exist yet"));
        out.println("Hub       " + hub.endpoint() + ", token " + Hub.token().map(t -> "from " + t.getKey()).orElse("not set (HF_TOKEN)"));
        return 0;
    }

    /** The backend from the library folder name native/build.sh writes: build/native/&lt;os&gt;-&lt;arch&gt;-&lt;backend&gt;. */
    static String backend(Path library) {
        Path dir = library.toAbsolutePath().getParent();
        if (dir == null) return null;
        String name = dir.getFileName().toString();
        String[] parts = name.split("-");
        return parts.length >= 3 ? parts[parts.length - 1] + " (from the folder name " + name + ")" : null;
    }

    private static String source() {
        String hub = System.getenv("HF_HUB_CACHE");
        if (hub != null && !hub.isBlank()) return "HF_HUB_CACHE";
        String home = System.getenv("HF_HOME");
        if (home != null && !home.isBlank()) return "HF_HOME/hub";
        return "default";
    }

    static String names() { return LIBRARIES.stream().map(l -> l[0]).collect(Collectors.joining(", ")); }
}
