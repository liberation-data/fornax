package org.liberationdata.kiln;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class GpuCheckTest {
    @TempDir Path dir;

    private Path lib(String name, byte[]... parts) throws IOException {
        Path p = dir.resolve(name);
        try (var out = Files.newOutputStream(p)) { for (byte[] part : parts) out.write(part); }
        return p;
    }

    private static byte[] ascii(String s) { return s.getBytes(StandardCharsets.US_ASCII); }

    @Test
    void seesWhichGpuRuntimeALibraryLinks() throws IOException {
        assertEquals(GpuCheck.Gpu.VULKAN, GpuCheck.linksTo(lib("a.so", ascii("\0libm.so.6\0libvulkan.so.1\0"))));
        assertEquals(GpuCheck.Gpu.HIP, GpuCheck.linksTo(lib("b.so", ascii("\0libhipblas.so.3\0libamdhip64.so.7\0"))));
        assertEquals(GpuCheck.Gpu.CUDA, GpuCheck.linksTo(lib("c.so", ascii("\0libcudart.so.12\0"))));
        assertEquals(GpuCheck.Gpu.NONE, GpuCheck.linksTo(lib("d.so", ascii("\0libm.so.6\0libgomp.so.1\0"))));
        assertEquals(GpuCheck.Gpu.NONE, GpuCheck.linksTo(dir.resolve("missing.so")));
    }

    @Test
    void seesTheRuntimeAcrossAChunkBoundary() throws IOException {
        assertEquals(GpuCheck.Gpu.VULKAN, GpuCheck.linksTo(lib("e.so", new byte[(1 << 20) - 5], ascii("libvulkan.so.1"))));
    }

    @Test
    void refusesToRunAGpuBuildWithNoGpu() {
        UnsatisfiedLinkError e = assertThrows(UnsatisfiedLinkError.class,
                () -> GpuCheck.decide(GpuCheck.Gpu.VULKAN, "libkiln_llama.so", OptionalInt.of(0), false, m -> {}));
        assertTrue(e.getMessage().contains("Kiln found no GPU for libkiln_llama.so, which is built for Vulkan"), e.getMessage());
        assertTrue(e.getMessage().contains("sudo dnf install vulkan-loader mesa-vulkan-drivers"), e.getMessage());
        assertTrue(e.getMessage().contains("vulkaninfo --summary"), e.getMessage());
        assertTrue(e.getMessage().contains(GpuCheck.DOCS), e.getMessage());
        assertTrue(e.getMessage().contains("-Dkiln.allowCpu=true"), e.getMessage());
    }

    @Test
    void warnsLoudlyWhenTheCpuIsAllowed() {
        List<String> warnings = new ArrayList<>();
        GpuCheck.decide(GpuCheck.Gpu.HIP, "libstable-diffusion.so", OptionalInt.of(0), true, warnings::add);
        assertEquals(1, warnings.size());
        assertTrue(warnings.getFirst().contains("KILN IS RUNNING ON THE CPU"), warnings.getFirst());
        assertTrue(warnings.getFirst().contains("HIP (ROCm)"), warnings.getFirst());
    }

    @Test
    void staysQuietWithAGpuOrWhenItCantTell() {
        List<String> warnings = new ArrayList<>();
        assertDoesNotThrow(() -> GpuCheck.decide(GpuCheck.Gpu.VULKAN, "x.so", OptionalInt.of(1), false, warnings::add));
        assertDoesNotThrow(() -> GpuCheck.decide(GpuCheck.Gpu.VULKAN, "x.so", OptionalInt.empty(), false, warnings::add));
        assertEquals(List.of(), warnings);
    }

    @Test
    void namesTheMissingVulkanLoader() throws IOException {
        assumeTrue(System.getProperty("os.name").toLowerCase().startsWith("linux"));
        Path vulkan = lib("libkiln_llama.so", ascii("libvulkan.so.1"));
        IllegalArgumentException failure = new IllegalArgumentException("Cannot open library");
        UnsatisfiedLinkError e = assertThrows(UnsatisfiedLinkError.class, () -> GpuCheck.explainLoadFailure(vulkan, failure, n -> false));
        assertTrue(e.getMessage().contains("no Vulkan loader (libvulkan.so.1)"), e.getMessage());
        assertTrue(e.getMessage().contains("sudo apt install libvulkan1 mesa-vulkan-drivers"), e.getMessage());
        assertSame(failure, e.getCause());
    }

    @Test
    void leavesOtherLoadFailuresAlone() throws IOException {
        Path vulkan = lib("libkiln_sd.so", ascii("libvulkan.so.1"));
        Path cpu = lib("libkiln_tts.so", ascii("libm.so.6"));
        IllegalArgumentException failure = new IllegalArgumentException("Cannot open library");
        assertDoesNotThrow(() -> GpuCheck.explainLoadFailure(vulkan, failure, n -> true));
        assertDoesNotThrow(() -> GpuCheck.explainLoadFailure(cpu, failure, n -> false));
    }

    @Test
    void givesEachRuntimeItsOwnAdvice() {
        assertTrue(GpuCheck.install(GpuCheck.Gpu.HIP).contains("rocminfo"));
        assertTrue(GpuCheck.install(GpuCheck.Gpu.CUDA).contains("nvidia-smi"));
    }
}
