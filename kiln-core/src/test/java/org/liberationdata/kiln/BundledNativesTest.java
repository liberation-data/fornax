package org.liberationdata.kiln;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.io.OutputStream;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.List;
import java.util.Map;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BundledNativesTest {
    @TempDir Path tmp;

    @Test
    void namesPlatformsTheWayNativesJarsDo() {
        assertEquals("linux-x86_64", BundledNatives.platform("Linux", "amd64"));
        assertEquals("linux-aarch64", BundledNatives.platform("Linux", "aarch64"));
        assertEquals("macos-aarch64", BundledNatives.platform("Mac OS X", "aarch64"));
        assertEquals("macos-x86_64", BundledNatives.platform("Mac OS X", "x86_64"));
        assertEquals("windows-x86_64", BundledNatives.platform("Windows 11", "amd64"));
    }

    @Test
    void findsTheBackendsOnTheClasspathInOrderOfPreference() throws IOException {
        try (var loader = loader(jar("natives.jar", Map.of(
                "org/kiln4j/natives/linux-x86_64-cpu/libkiln_llama.so", "cpu",
                "org/kiln4j/natives/linux-x86_64-vulkan/libkiln_llama.so", "vulkan",
                "org/kiln4j/natives/linux-x86_64-cuda/libkiln_llama.so", "cuda",
                "org/kiln4j/natives/linux-x86_64-hip/libstable-diffusion.so", "hip",
                "org/kiln4j/natives/linux-aarch64-vulkan/libkiln_llama.so", "arm")))) {
            assertEquals(List.of("cuda", "vulkan", "cpu"), BundledNatives.available(loader, "linux-x86_64", "libkiln_llama.so"));
            assertEquals(List.of("hip"), BundledNatives.available(loader, "linux-x86_64", "libstable-diffusion.so"));
            assertEquals(List.of(), BundledNatives.available(loader, "linux-x86_64", "libkiln_yue2.so"));
            assertEquals(List.of(), BundledNatives.available(loader, "macos-aarch64", "libkiln_llama.so"));
        }
    }

    @Test
    void findsBackendsAcrossSeveralJars() throws IOException {
        try (var loader = loader(
                jar("vulkan.jar", Map.of("org/kiln4j/natives/macos-aarch64-vulkan/libkiln_llama.dylib", "v")),
                jar("metal.jar", Map.of("org/kiln4j/natives/macos-aarch64-metal/libkiln_llama.dylib", "m")))) {
            List<String> available = BundledNatives.available(loader, "macos-aarch64", "libkiln_llama.dylib");
            assertEquals(List.of("metal", "vulkan"), available);
            assertEquals("metal", BundledNatives.choose(available, null, "libkiln_llama.dylib"));
        }
    }

    @Test
    void prefersTheFirstAvailableUnlessOneIsAskedFor() {
        List<String> available = List.of("hip", "vulkan", "cpu");
        assertEquals("hip", BundledNatives.choose(available, null, "libkiln_llama.so"));
        assertEquals("hip", BundledNatives.choose(available, " ", "libkiln_llama.so"));
        assertEquals("vulkan", BundledNatives.choose(available, "Vulkan", "libkiln_llama.so"));
        assertEquals("cpu", BundledNatives.choose(available, "cpu", "libkiln_llama.so"));
    }

    @Test
    void failsWhenTheBackendAskedForIsNotThere() {
        var e = assertThrows(UnsatisfiedLinkError.class,
                () -> BundledNatives.choose(List.of("vulkan"), "cuda", "libkiln_llama.so"));
        assertTrue(e.getMessage().contains("'cuda'") && e.getMessage().contains("[vulkan]"), e.getMessage());
    }

    @Test
    void extractsALibraryOnceAndReusesTheCopy() throws IOException {
        String entry = "org/kiln4j/natives/linux-x86_64-vulkan/libkiln_llama.so";
        try (var loader = loader(jar("natives.jar", Map.of(entry, "the library")))) {
            Path dir = tmp.resolve("cache/0.1.0/linux-x86_64-vulkan");
            Path lib = BundledNatives.extract(loader.getResource(entry), dir, "libkiln_llama.so");
            assertEquals(dir.resolve("libkiln_llama.so"), lib);
            assertArrayEquals("the library".getBytes(StandardCharsets.UTF_8), Files.readAllBytes(lib));
            Object first = Files.readAttributes(lib, BasicFileAttributes.class).fileKey();

            Path again = BundledNatives.extract(loader.getResource(entry), dir, "libkiln_llama.so");
            assertEquals(first, Files.readAttributes(again, BasicFileAttributes.class).fileKey(), "re-extracted an unchanged library");
            try (Stream<Path> files = Files.list(dir)) {
                assertEquals(List.of(lib, dir.resolve("libkiln_llama.so.stamp")), files.sorted().toList(), "left a temporary file behind");
            }
        }
    }

    @Test
    void replacesACopyThatDoesNotMatchTheJar() throws IOException {
        String entry = "org/kiln4j/natives/linux-x86_64-vulkan/libkiln_llama.so";
        Path dir = tmp.resolve("cache");
        Files.createDirectories(dir);
        Path lib = dir.resolve("libkiln_llama.so");
        Files.writeString(lib, "a copy with no stamp");
        try (var loader = loader(jar("natives.jar", Map.of(entry, "the whole library")))) {
            BundledNatives.extract(loader.getResource(entry), dir, "libkiln_llama.so");
            assertEquals("the whole library", Files.readString(lib));
        }
        // A rebuilt snapshot: the same size and the same (reproducible) entry time, but different bytes.
        try (var loader = loader(jar("rebuilt.jar", Map.of(entry, "the other library")))) {
            BundledNatives.extract(loader.getResource(entry), dir, "libkiln_llama.so");
            assertEquals("the other library", Files.readString(lib));
        }
    }

    @Test
    void cachesUnderTheDirectoryAskedFor() {
        String before = System.getProperty("kiln.cache.dir");
        try {
            System.setProperty("kiln.cache.dir", tmp.toString());
            assertEquals(tmp, BundledNatives.cacheRoot());
        } finally {
            if (before == null) System.clearProperty("kiln.cache.dir"); else System.setProperty("kiln.cache.dir", before);
        }
    }

    private Path jar(String name, Map<String, String> entries) throws IOException {
        Path jar = tmp.resolve(name);
        try (OutputStream out = Files.newOutputStream(jar); JarOutputStream j = new JarOutputStream(out)) {
            for (var e : entries.entrySet()) {
                JarEntry entry = new JarEntry(e.getKey());
                entry.setTime(1_700_000_000_000L);
                j.putNextEntry(entry);
                j.write(e.getValue().getBytes(StandardCharsets.UTF_8));
                j.closeEntry();
            }
        }
        return jar;
    }

    private static URLClassLoader loader(Path... jars) throws IOException {
        URL[] urls = new URL[jars.length];
        for (int i = 0; i < jars.length; i++) urls[i] = jars[i].toUri().toURL();
        return new URLClassLoader(urls, null);
    }
}
