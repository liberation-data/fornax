package org.liberationdata.kiln;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class VulkanDriverTest {
    @TempDir Path dir;

    private Path lib(String name, byte[]... parts) throws IOException {
        Path p = dir.resolve(name);
        try (var out = Files.newOutputStream(p)) { for (byte[] part : parts) out.write(part); }
        return p;
    }

    @Test
    void seesTheLoaderInALibrary() throws IOException {
        assertTrue(VulkanDriver.needsLoader(lib("a.so", "\0libm.so.6\0libvulkan.so.1\0".getBytes(StandardCharsets.US_ASCII))));
        assertFalse(VulkanDriver.needsLoader(lib("b.so", "\0libm.so.6\0libvulkan.so\0".getBytes(StandardCharsets.US_ASCII))));
        assertFalse(VulkanDriver.needsLoader(dir.resolve("missing.so")));
    }

    @Test
    void seesTheLoaderAcrossAChunkBoundary() throws IOException {
        byte[] pad = new byte[(1 << 20) - 5];
        assertTrue(VulkanDriver.needsLoader(lib("c.so", pad, "libvulkan.so.1".getBytes(StandardCharsets.US_ASCII))));
    }

    @Test
    void namesTheMissingLoader() throws IOException {
        assumeTrue(System.getProperty("os.name").toLowerCase().startsWith("linux"));
        Path vulkan = lib("libkiln_llama.so", "libvulkan.so.1".getBytes(StandardCharsets.US_ASCII));
        IllegalArgumentException failure = new IllegalArgumentException("Cannot open library");
        UnsatisfiedLinkError e = assertThrows(UnsatisfiedLinkError.class, () -> VulkanDriver.explain(vulkan, failure, n -> false));
        assertTrue(e.getMessage().contains("no Vulkan loader (libvulkan.so.1)"), e.getMessage());
        assertTrue(e.getMessage().contains("sudo dnf install vulkan-loader mesa-vulkan-drivers"), e.getMessage());
        assertTrue(e.getMessage().contains(VulkanDriver.DOCS), e.getMessage());
        assertSame(failure, e.getCause());
    }

    @Test
    void leavesOtherFailuresAlone() throws IOException {
        Path vulkan = lib("libkiln_sd.so", "libvulkan.so.1".getBytes(StandardCharsets.US_ASCII));
        Path cpu = lib("libkiln_tts.so", "libm.so.6".getBytes(StandardCharsets.US_ASCII));
        IllegalArgumentException failure = new IllegalArgumentException("Cannot open library");
        assertDoesNotThrow(() -> VulkanDriver.explain(vulkan, failure, n -> true));
        assertDoesNotThrow(() -> VulkanDriver.explain(cpu, failure, n -> false));
    }

    @Test
    void findsADriverManifest() throws IOException {
        Path share = dir.resolve("share");
        Map<String, String> env = Map.of("XDG_DATA_DIRS", share.toString(), "XDG_CONFIG_DIRS", dir.resolve("none").toString());
        Path home = dir.resolve("home");
        Files.createDirectories(share.resolve("vulkan/icd.d"));
        assumeTrue(!Files.isDirectory(Path.of("/etc/vulkan/icd.d")), "this machine has drivers in /etc/vulkan/icd.d");
        assertFalse(VulkanDriver.hasDriver(env, home));
        Files.writeString(share.resolve("vulkan/icd.d/radeon_icd.x86_64.json"), "{}");
        assertTrue(VulkanDriver.hasDriver(env, home));
    }

    @Test
    void trustsAnOverride() {
        assertTrue(VulkanDriver.hasDriver(Map.of("VK_DRIVER_FILES", "/opt/x/icd.json"), dir));
        assertTrue(VulkanDriver.hasDriver(Map.of("VK_ICD_FILENAMES", "/opt/x/icd.json"), dir));
    }

    @Test
    void looksWhereTheLoaderLooks() {
        Path home = Path.of("/home/u");
        assertEquals(List.of(Path.of("/home/u/.config/vulkan/icd.d"), Path.of("/etc/xdg/vulkan/icd.d"), Path.of("/etc/vulkan/icd.d"),
                        Path.of("/home/u/.local/share/vulkan/icd.d"), Path.of("/usr/local/share/vulkan/icd.d"),
                        Path.of("/usr/share/vulkan/icd.d")),
                VulkanDriver.manifestDirs(Map.of(), home));
    }

    @Test
    void warnsWithTheInstallCommands() {
        String m = VulkanDriver.noDriverMessage(Path.of("/x/libkiln_llama.so"));
        assertTrue(m.contains("libkiln_llama.so will run on the CPU"), m);
        assertTrue(m.contains("sudo apt install libvulkan1 mesa-vulkan-drivers"), m);
    }
}
