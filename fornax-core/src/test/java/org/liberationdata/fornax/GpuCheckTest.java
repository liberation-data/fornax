package org.liberationdata.fornax;

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
    void seesThatAWindowsDllRunsOnVulkan() throws IOException {
        assertEquals(GpuCheck.Gpu.VULKAN, GpuCheck.linksTo(lib("a.dll", ascii("KERNEL32.dll\0vulkan-1.dll\0"))));
        assertEquals(GpuCheck.Gpu.VULKAN, GpuCheck.linksTo(lib("b.dll", ascii("KERNEL32.dll\0VULKAN-1.DLL\0"))));
        assertEquals(GpuCheck.Gpu.NONE, GpuCheck.linksTo(lib("c.dll", ascii("KERNEL32.dll\0msvcrt.dll\0"))));
    }

    @Test
    void checksVulkanOnWindowsAndEveryRuntimeOnLinux() {
        assertTrue(GpuCheck.checked(GpuCheck.Gpu.VULKAN, true));
        assertEquals(false, GpuCheck.checked(GpuCheck.Gpu.HIP, true));
        assertEquals(false, GpuCheck.checked(GpuCheck.Gpu.CUDA, true));
        boolean linux = System.getProperty("os.name").toLowerCase().startsWith("linux");
        for (GpuCheck.Gpu gpu : List.of(GpuCheck.Gpu.VULKAN, GpuCheck.Gpu.HIP, GpuCheck.Gpu.CUDA))
            assertEquals(linux, GpuCheck.checked(gpu, false), gpu.name());
    }

    @Test
    void namesTheMissingVulkanLoaderOnWindows() throws IOException {
        Path vulkan = lib("fornax_llama.dll", ascii("vulkan-1.dll"));
        IllegalArgumentException failure = new IllegalArgumentException("Cannot open library");
        UnsatisfiedLinkError e = assertThrows(UnsatisfiedLinkError.class, () -> GpuCheck.explainLoadFailure(vulkan, failure, n -> false, true));
        assertTrue(e.getMessage().contains("no Vulkan loader (vulkan-1.dll)"), e.getMessage());
        assertTrue(e.getMessage().contains("AMD Software (Adrenalin Edition)"), e.getMessage());
        assertTrue(!e.getMessage().contains("sudo"), e.getMessage());
        assertSame(failure, e.getCause());
        assertDoesNotThrow(() -> GpuCheck.explainLoadFailure(vulkan, failure, n -> true, true));
        // A CPU build, and a runtime Windows isn't checked for, rethrow the loader's own error.
        Path cpu = lib("fornax_tts.dll", ascii("KERNEL32.dll"));
        Path hip = lib("fornax_sd.dll", ascii("libamdhip64.so"));
        assertDoesNotThrow(() -> GpuCheck.explainLoadFailure(cpu, failure, n -> false, true));
        assertDoesNotThrow(() -> GpuCheck.explainLoadFailure(hip, failure, n -> false, true));
    }

    @Test
    void givesWindowsItsOwnVulkanAdvice() {
        String advice = GpuCheck.install(GpuCheck.Gpu.VULKAN, true);
        assertTrue(advice.contains("NVIDIA") && advice.contains("Intel") && advice.contains("vulkaninfo --summary"), advice);
        assertTrue(advice.contains(GpuCheck.DOCS), advice);
        assertTrue(!advice.contains("dnf") && !advice.contains("apt"), advice);
    }

    /**
     * A Vulkan loader with no driver behind it, as on a Windows machine whose GPU driver has no Vulkan: the loader
     * reports no driver, and Fornax counts no GPU. Simulated on Linux by pointing the loader at no driver manifest, in a
     * JVM of its own (the loader reads the variables once).
     */
    @Test
    void aVulkanLoaderWithNoDriverMeansNoGpu() throws Exception {
        assumeTrue(System.getProperty("os.name").toLowerCase().startsWith("linux"));
        assumeTrue(loads("libvulkan.so.1"));
        ProcessBuilder pb = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "--enable-native-access=ALL-UNNAMED", "-cp", System.getProperty("java.class.path"), VulkanProbe.class.getName());
        pb.environment().remove("GGML_VK_VISIBLE_DEVICES");
        pb.environment().put("VK_ICD_FILENAMES", dir.resolve("none.json").toString());
        pb.environment().put("VK_DRIVER_FILES", dir.resolve("none.json").toString());
        pb.environment().remove("VK_ADD_DRIVER_FILES");   // added on top of VK_DRIVER_FILES, so it would bring a driver back
        pb.redirectErrorStream(true);
        Process p = pb.start();
        String out = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8).strip();
        assertEquals(0, p.waitFor(), out);
        assertTrue(out.endsWith("gpus=OptionalInt[0]"), out);
    }

    /** Prints what {@link GpuCheck#gpus} says about Vulkan, for {@link #aVulkanLoaderWithNoDriverMeansNoGpu}. */
    static final class VulkanProbe {
        public static void main(String[] args) {
            System.out.println("gpus=" + GpuCheck.gpus(GpuCheck.Gpu.VULKAN));
        }
    }

    @Test
    void seesTheRuntimeAcrossAChunkBoundary() throws IOException {
        assertEquals(GpuCheck.Gpu.VULKAN, GpuCheck.linksTo(lib("e.so", new byte[(1 << 20) - 5], ascii("libvulkan.so.1"))));
    }

    @Test
    void refusesToRunAGpuBuildWithNoGpu() {
        UnsatisfiedLinkError e = assertThrows(UnsatisfiedLinkError.class,
                () -> GpuCheck.decide(GpuCheck.Gpu.VULKAN, "libfornax_llama.so", OptionalInt.of(0), false, m -> {}));
        assertTrue(e.getMessage().contains("Fornax found no GPU for libfornax_llama.so, which is built for Vulkan"), e.getMessage());
        assertTrue(e.getMessage().contains("sudo dnf install vulkan-loader mesa-vulkan-drivers vulkan-tools"), e.getMessage());
        assertTrue(e.getMessage().contains("vulkaninfo --summary"), e.getMessage());
        assertTrue(e.getMessage().contains(GpuCheck.DOCS), e.getMessage());
        assertTrue(e.getMessage().contains("-Dfornax.allowCpu=true"), e.getMessage());
    }

    @Test
    void warnsLoudlyWhenTheCpuIsAllowed() {
        List<String> warnings = new ArrayList<>();
        GpuCheck.decide(GpuCheck.Gpu.HIP, "libstable-diffusion.so", OptionalInt.of(0), true, warnings::add);
        assertEquals(1, warnings.size());
        assertTrue(warnings.getFirst().contains("FORNAX IS RUNNING ON THE CPU"), warnings.getFirst());
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
        Path vulkan = lib("libfornax_llama.so", ascii("libvulkan.so.1"));
        IllegalArgumentException failure = new IllegalArgumentException("Cannot open library");
        UnsatisfiedLinkError e = assertThrows(UnsatisfiedLinkError.class, () -> GpuCheck.explainLoadFailure(vulkan, failure, n -> false, false));
        assertTrue(e.getMessage().contains("no Vulkan loader (libvulkan.so.1)"), e.getMessage());
        assertTrue(e.getMessage().contains("sudo apt install libvulkan1 mesa-vulkan-drivers"), e.getMessage());
        assertSame(failure, e.getCause());
    }

    @Test
    void leavesOtherLoadFailuresAlone() throws IOException {
        Path vulkan = lib("libfornax_sd.so", ascii("libvulkan.so.1"));
        Path cpu = lib("libfornax_tts.so", ascii("libm.so.6"));
        IllegalArgumentException failure = new IllegalArgumentException("Cannot open library");
        assertDoesNotThrow(() -> GpuCheck.explainLoadFailure(vulkan, failure, n -> true, false));
        assertDoesNotThrow(() -> GpuCheck.explainLoadFailure(cpu, failure, n -> false, false));
    }

    @Test
    void asksVulkanHowManyGpusItHas() throws IOException {
        assumeTrue(System.getProperty("os.name").toLowerCase().startsWith("linux"));
        assumeTrue(loads("libvulkan.so.1"));
        assumeTrue(System.getenv("GGML_VK_VISIBLE_DEVICES") == null);
        OptionalInt gpus = GpuCheck.gpus(GpuCheck.Gpu.VULKAN);
        assertTrue(gpus.isPresent(), "the probe couldn't tell");
        boolean renderNode;
        try (var nodes = Files.list(Path.of("/dev/dri")).filter(p -> p.getFileName().toString().startsWith("renderD"))) {
            renderNode = nodes.findAny().isPresent();
        } catch (IOException e) {
            renderNode = false;
        }
        if (!renderNode) assertEquals(0, gpus.getAsInt(), "no GPU device, yet Vulkan counted a GPU (llvmpipe?)");
    }

    @Test
    void noCudaDriverMeansNoGpu() {
        assumeTrue(System.getProperty("os.name").toLowerCase().startsWith("linux"));
        assumeTrue(!loads("libcuda.so.1"));
        assertEquals(OptionalInt.of(0), GpuCheck.gpus(GpuCheck.Gpu.CUDA));
    }

    @Test
    void allowsTheCpuOnlyWhenAskedTo() {
        assumeTrue(System.getenv("FORNAX_ALLOW_CPU") == null);
        String before = System.getProperty(GpuCheck.ALLOW_CPU);
        try {
            System.clearProperty(GpuCheck.ALLOW_CPU);
            assertEquals(false, GpuCheck.allowCpu());
            for (String yes : List.of("true", "TRUE", "1")) {
                System.setProperty(GpuCheck.ALLOW_CPU, yes);
                assertTrue(GpuCheck.allowCpu(), yes);
            }
            for (String no : List.of("false", "0", "yes")) {
                System.setProperty(GpuCheck.ALLOW_CPU, no);
                assertEquals(false, GpuCheck.allowCpu(), no);
            }
        } finally {
            if (before == null) System.clearProperty(GpuCheck.ALLOW_CPU);
            else System.setProperty(GpuCheck.ALLOW_CPU, before);
        }
    }

    @Test
    void leavesACpuBuildAlone() throws IOException {
        Path cpu = lib("libfornax_cpu.so", ascii("\0libm.so.6\0libgomp.so.1\0"));
        assertDoesNotThrow(() -> GpuCheck.requireGpu(cpu));
    }

    @Test
    void refusesAGpuBuildOnlyWhenTheRuntimeFindsNoGpu() throws IOException {
        assumeTrue(System.getProperty("os.name").toLowerCase().startsWith("linux"));
        assumeTrue(System.getenv("FORNAX_ALLOW_CPU") == null);
        Path vulkan = lib("libfornax_gpu.so", ascii("\0libvulkan.so.1\0"));
        String before = System.getProperty(GpuCheck.ALLOW_CPU);
        try {
            System.clearProperty(GpuCheck.ALLOW_CPU);
            if (GpuCheck.gpus(GpuCheck.Gpu.VULKAN).equals(OptionalInt.of(0))) {
                UnsatisfiedLinkError e = assertThrows(UnsatisfiedLinkError.class, () -> GpuCheck.requireGpu(vulkan));
                assertTrue(e.getMessage().contains("Fornax found no GPU for libfornax_gpu.so"), e.getMessage());
            } else {
                assertDoesNotThrow(() -> GpuCheck.requireGpu(vulkan));
            }
            System.setProperty(GpuCheck.ALLOW_CPU, "true");
            assertDoesNotThrow(() -> GpuCheck.requireGpu(vulkan));
            System.clearProperty(GpuCheck.ALLOW_CPU);
            assertDoesNotThrow(() -> GpuCheck.requireGpu(vulkan), "a runtime that passed once is not checked again");
        } finally {
            if (before == null) System.clearProperty(GpuCheck.ALLOW_CPU);
            else System.setProperty(GpuCheck.ALLOW_CPU, before);
        }
    }

    @Test
    void nativeLibraryExplainsOrRethrowsALoadFailure() throws IOException {
        assumeTrue(System.getProperty("os.name").toLowerCase().startsWith("linux"));
        Path broken = lib("libfornax_broken.so", ascii("not an ELF file, but it names libvulkan.so.1"));
        String property = "fornax.fornax_broken_test.lib";
        System.setProperty(property, broken.toString());
        try {
            Class<? extends Throwable> expected = loads("libvulkan.so.1") ? IllegalArgumentException.class : UnsatisfiedLinkError.class;
            assertThrows(expected, () -> NativeLibrary.load("fornax_broken_test"));
        } finally {
            System.clearProperty(property);
        }
    }

    private static boolean loads(String name) {
        try {
            java.lang.foreign.SymbolLookup.libraryLookup(name, java.lang.foreign.Arena.global());
            return true;
        } catch (IllegalArgumentException e) {
            return false;
        }
    }

    @Test
    void givesEachRuntimeItsOwnAdvice() {
        assertTrue(GpuCheck.install(GpuCheck.Gpu.HIP).contains("rocminfo"));
        assertTrue(GpuCheck.install(GpuCheck.Gpu.CUDA).contains("nvidia-smi"));
    }
}
