package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.hub.RepoIndex;
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.NativeLibrary;
import org.liberationdata.fornax.hub.HubToken;
import org.liberationdata.fornax.hub.FornaxHub;

import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/** {@code fornax info}: where Fornax looks for its native libraries and models. Loads nothing. */
final class Info {
    static final String USAGE = """
            usage: fornax info

            Shows the Java runtime, which native libraries Fornax finds (and their GPU backend), and the
            Hugging Face cache it reads and writes. Nothing is loaded onto the GPU.""";

    /** The engines, by the native library name each module loads. */
    static final List<String[]> LIBRARIES = List.of(
            new String[]{"fornax_llama", "llama.cpp (fornax chat and see)"},
            new String[]{"stable-diffusion", "stable-diffusion.cpp (fornax image)"},
            new String[]{"fornax_acestep", "acestep.cpp (fornax song, ACE-Step)"},
            new String[]{"fornax_yue2", "yue2.cpp (fornax song, YuE2)"},
            new String[]{"fornax_tts", "llama.cpp text to speech (fornax speak)"},
            new String[]{"fornax_whisper", "whisper.cpp (fornax transcribe)"});

    static int run(FornaxHub hub, PrintStream out) {
        String v = Info.class.getPackage().getImplementationVersion();
        out.println("Fornax      " + (v == null ? "(development build)" : v));
        out.println("Java      " + System.getProperty("java.runtime.version") + " (" + System.getProperty("java.vendor")
                + ") " + System.getProperty("java.home"));
        out.println("Platform  " + System.getProperty("os.name") + " " + System.getProperty("os.arch"));
        List<String> dirs = NativeLibrary.searchDirs();
        List<String> jars = nativesJars();
        out.println("Natives   searched " + (dirs.isEmpty() ? "(no fornax.library.path or FORNAX_LIBRARY_PATH)" : String.join(", ", dirs))
                + ", then " + (jars.isEmpty() ? "a fornax-natives jar (none on the classpath)" : String.join(", ", jars))
                + ", then the platform loader");
        String backend = null;
        int found = 0;
        for (String[] lib : LIBRARIES) {
            Optional<Path> p = NativeLibrary.locate(lib[0]);
            if (p.isPresent() && Files.isRegularFile(p.get())) {
                found++;
                if (backend == null) backend = backend(p.get());
                out.printf("  %-18s %s (%s)%n", lib[0], p.get(), Fmt.bytes(p.get().toFile().length()));
            } else {
                out.printf("  %-18s %s: %s%n", lib[0], jars.isEmpty() ? "not found in those folders" : "from the natives jar", lib[1]);
            }
        }
        if (backend == null && !jars.isEmpty()) backend = jars.get(0).replaceAll(".*fornax-natives-[^-]+-[^-]+-([^-]+)-.*", "$1");
        out.println("Backend   " + (backend != null ? backend : found == 0 ? "unknown (no libraries found)" : "unknown")
                + (found == 0 && jars.isEmpty() ? "\n          build them with native/build.sh, then set FORNAX_LIBRARY_PATH (or -Dfornax.library.path)" : ""));
        Path root = hub.cacheRoot();
        List<String> repos = HfCache.repos(root);
        long bytes = repos.stream().flatMap(r -> HfCache.files(root, r).values().stream()).mapToLong(p -> p.toFile().length()).sum();
        out.println("Cache     " + root + " (" + source() + ")"
                + (Files.isDirectory(root) ? ": " + repos.size() + " repos, " + Fmt.bytes(bytes) : ": doesn't exist yet"));
        if (!hub.modelFolders().dirs().isEmpty()) out.println("Folders   " + hub.modelFolders().dirs() + " (FORNAX_MODEL_DIRS)");
        out.println("Hub       " + RepoIndex.endpoint() + ", token " + (HubToken.find().isPresent()
                ? "set (HF_TOKEN or " + HubToken.tokenFile() + ")" : "not set (HF_TOKEN, or `hf auth login`)"));
        return 0;
    }

    /** The fornax-natives jars on the classpath, by file name (the launcher's lib folder has the platform's). */
    static List<String> nativesJars() {
        return Arrays.stream(System.getProperty("java.class.path", "").split(java.io.File.pathSeparator))
                .map(e -> Path.of(e).getFileName()).filter(n -> n != null && n.toString().startsWith("fornax-natives-"))
                .map(Path::toString).toList();
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
