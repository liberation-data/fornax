package org.liberationdata.fornax;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;
import java.util.Locale;
import java.util.OptionalInt;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Predicate;

/**
 * Makes sure an engine built for a GPU has one, on Linux, so it never drops to the CPU without anyone noticing.
 *
 * <p>An engine built for Vulkan, HIP or CUDA still loads on a machine with no usable GPU (no driver, or no access to
 * the device), and ggml then runs everything on the CPU: many times slower, with no error. So after such a library
 * loads, Fornax asks the GPU runtime how many GPUs it can use, the way the engine will, and when the answer is none it
 * refuses with an error that says what to install. {@code -Dfornax.allowCpu=true} (or {@code FORNAX_ALLOW_CPU=true})
 * runs on the CPU anyway, with a loud warning. When Fornax can't tell (an unexpected error from the runtime, or a
 * device list chosen with {@code GGML_VK_VISIBLE_DEVICES}), it lets the engine decide.
 *
 * <p>A Vulkan build also can't load at all without the Vulkan loader ({@code libvulkan.so.1}); that failure gets the
 * same advice. macOS always has Metal, so none of this runs there.
 */
final class GpuCheck {
    static final String DOCS = "https://fornax4j.org/docs/gpu-drivers/";
    static final String ALLOW_CPU = "fornax.allowCpu";

    /** The GPU runtime a library links, found by the runtime's library name in the library's file. */
    enum Gpu {
        VULKAN("Vulkan", "libvulkan.so.1"), HIP("HIP (ROCm)", "libamdhip64.so"), CUDA("CUDA", "libcuda"), NONE("", null);

        final String label;
        final String soname;

        Gpu(String label, String soname) {
            this.label = label;
            this.soname = soname;
        }
    }

    private static final Set<Gpu> passed = EnumSet.noneOf(Gpu.class);

    private GpuCheck() {}

    /** The GPU runtime the library at {@code lib} links, or NONE (a CPU build, or a file Fornax can't read). */
    static Gpu linksTo(Path lib) {
        List<Gpu> gpus = List.of(Gpu.VULKAN, Gpu.HIP, Gpu.CUDA);
        byte[][] needles = gpus.stream().map(g -> g.soname.getBytes(StandardCharsets.US_ASCII)).toArray(byte[][]::new);
        int longest = 0;
        for (byte[] n : needles) longest = Math.max(longest, n.length);
        byte[] buf = new byte[1 << 20];
        int kept = 0;                                  // the tail of the last chunk, for a name split across chunks
        try (var in = Files.newInputStream(lib)) {
            for (int n; (n = in.read(buf, kept, buf.length - kept)) > 0; ) {
                int end = kept + n;
                for (int g = 0; g < needles.length; g++) if (contains(buf, end, needles[g])) return gpus.get(g);
                kept = Math.min(longest - 1, end);
                System.arraycopy(buf, end - kept, buf, 0, kept);
            }
        } catch (IOException e) {
            return Gpu.NONE;
        }
        return Gpu.NONE;
    }

    private static boolean contains(byte[] buf, int end, byte[] needle) {
        outer:
        for (int i = 0; i + needle.length <= end; i++) {
            for (int j = 0; j < needle.length; j++) if (buf[i + j] != needle[j]) continue outer;
            return true;
        }
        return false;
    }

    /**
     * Throws an error that names the missing Vulkan loader when that's why the library at {@code lib} failed to open
     * ({@code failure}); returns otherwise, for the caller to rethrow {@code failure}.
     */
    static void explainLoadFailure(Path lib, IllegalArgumentException failure, Predicate<String> loadable) {
        if (!isLinux() || linksTo(lib) != Gpu.VULKAN || loadable.test(Gpu.VULKAN.soname)) return;
        UnsatisfiedLinkError e = new UnsatisfiedLinkError("Fornax could not load " + lib.getFileName() + ": it runs on "
                + "Vulkan, and this machine has no Vulkan loader (" + Gpu.VULKAN.soname + ").\n\n" + install(Gpu.VULKAN));
        e.initCause(failure);
        throw e;
    }

    static void explainLoadFailure(Path lib, IllegalArgumentException failure) {
        explainLoadFailure(lib, failure, GpuCheck::loadable);
    }

    /** After the library at {@code lib} loads: refuses (or, if allowed, warns) when it is a GPU build with no GPU. */
    static synchronized void requireGpu(Path lib) {
        if (!isLinux()) return;
        Gpu gpu = linksTo(lib);
        if (gpu == Gpu.NONE || passed.contains(gpu)) return;
        decide(gpu, lib.getFileName().toString(), gpus(gpu), allowCpu(), message ->
                System.getLogger(GpuCheck.class.getName()).log(System.Logger.Level.WARNING, message));
        passed.add(gpu);
    }

    /** Refuses (throws) or warns, given how many usable GPUs the runtime reported; empty means it couldn't tell. */
    static void decide(Gpu gpu, String lib, OptionalInt gpus, boolean allowCpu, Consumer<String> warn) {
        if (gpus.isEmpty() || gpus.getAsInt() > 0) return;
        if (allowCpu) {
            String bar = "*".repeat(100);
            warn.accept("\n" + bar + "\n  FORNAX IS RUNNING ON THE CPU. No GPU was found for " + lib + ", which is built for "
                    + gpu.label + ".\n  Everything will be many times slower. It runs anyway because " + ALLOW_CPU
                    + " is set.\n  To use your GPU, install its driver: " + DOCS + "\n" + bar);
            return;
        }
        throw new UnsatisfiedLinkError("Fornax found no GPU for " + lib + ", which is built for " + gpu.label + ". Without "
                + "one, it would run on the CPU, many times slower, so it won't start.\n\n" + install(gpu)
                + "\n\nTo run on the CPU anyway, set -D" + ALLOW_CPU + "=true (or FORNAX_ALLOW_CPU=true).");
    }

    /** What to install, and how to check, for this runtime. */
    static String install(Gpu gpu) {
        return switch (gpu) {
            case VULKAN -> """
                    Install your GPU's Vulkan driver:
                      Fedora:          sudo dnf install vulkan-loader mesa-vulkan-drivers vulkan-tools
                      Ubuntu, Debian:  sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
                      Arch:            sudo pacman -S vulkan-icd-loader vulkan-tools vulkan-radeon   (vulkan-intel for Intel)
                      openSUSE:        sudo zypper install libvulkan1 vulkan-tools libvulkan_radeon  (libvulkan_intel for Intel)
                    NVIDIA: install NVIDIA's driver, which includes Vulkan.
                    Then check that your GPU is listed:  vulkaninfo --summary
                    """ + guide();
            case HIP -> """
                    Install ROCm for your GPU, and make sure your user can reach it (the render and video groups):
                      sudo usermod -aG render,video $USER    (then log in again)
                    Then check that your GPU is listed:  rocminfo
                    """ + guide();
            case CUDA -> """
                    Install NVIDIA's driver for your GPU.
                    Then check that your GPU is listed:  nvidia-smi
                    """ + guide();
            case NONE -> guide();
        };
    }

    private static String guide() {
        return "Step by step, for your system: " + DOCS;
    }

    static boolean allowCpu() {
        String v = System.getProperty(ALLOW_CPU, System.getenv("FORNAX_ALLOW_CPU"));
        return v != null && (v.equalsIgnoreCase("true") || v.equals("1"));
    }

    /** How many GPUs the runtime can use, as the engine will count them; empty when Fornax can't tell. */
    static OptionalInt gpus(Gpu gpu) {
        try {
            return switch (gpu) {
                case VULKAN -> vulkanGpus();
                case HIP -> hipGpus();
                case CUDA -> cudaGpus();
                case NONE -> OptionalInt.empty();
            };
        } catch (Throwable t) {
            return OptionalInt.empty();
        }
    }

    private static final Linker LINKER = Linker.nativeLinker();
    private static final int VK_ERROR_INCOMPATIBLE_DRIVER = -9;
    private static final int VK_ERROR_INITIALIZATION_FAILED = -3;
    private static final int VK_INTEGRATED_GPU = 1, VK_DISCRETE_GPU = 2;

    /** ggml's default: the integrated and discrete GPUs (not software renderers such as lavapipe). */
    private static OptionalInt vulkanGpus() throws Throwable {
        if (System.getenv("GGML_VK_VISIBLE_DEVICES") != null) return OptionalInt.empty();
        try (Arena arena = Arena.ofConfined()) {
            SymbolLookup vk = SymbolLookup.libraryLookup(Gpu.VULKAN.soname, arena);
            MethodHandle create = handle(vk, "vkCreateInstance", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            MethodHandle enumerate = handle(vk, "vkEnumeratePhysicalDevices", FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            MethodHandle properties = handle(vk, "vkGetPhysicalDeviceProperties", FunctionDescriptor.ofVoid(
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS));
            MethodHandle destroy = handle(vk, "vkDestroyInstance", FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS));

            MemorySegment info = arena.allocate(64, 8);           // VkInstanceCreateInfo, all zero but sType
            info.set(ValueLayout.JAVA_INT, 0, 1);                 // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
            MemorySegment instanceOut = arena.allocate(ValueLayout.ADDRESS);
            int rc = (int) create.invokeExact(info, MemorySegment.NULL, instanceOut);
            if (rc == VK_ERROR_INCOMPATIBLE_DRIVER || rc == VK_ERROR_INITIALIZATION_FAILED) return OptionalInt.of(0);
            if (rc != 0) return OptionalInt.empty();
            MemorySegment instance = instanceOut.get(ValueLayout.ADDRESS, 0);
            try {
                MemorySegment count = arena.allocate(ValueLayout.JAVA_INT);
                if ((int) enumerate.invokeExact(instance, count, MemorySegment.NULL) != 0) return OptionalInt.empty();
                int n = count.get(ValueLayout.JAVA_INT, 0);
                if (n == 0) return OptionalInt.of(0);
                MemorySegment devices = arena.allocate(ValueLayout.ADDRESS, n);
                int got = (int) enumerate.invokeExact(instance, count, devices);
                if (got != 0 && got != 5) return OptionalInt.empty();     // 5: VK_INCOMPLETE
                n = count.get(ValueLayout.JAVA_INT, 0);
                MemorySegment props = arena.allocate(1024, 8);    // VkPhysicalDeviceProperties is 824 bytes
                int gpus = 0;
                for (int i = 0; i < n; i++) {
                    properties.invokeExact(devices.getAtIndex(ValueLayout.ADDRESS, i), props);
                    int type = props.get(ValueLayout.JAVA_INT, 16);    // after apiVersion, driverVersion, vendorID, deviceID
                    if (type == VK_INTEGRATED_GPU || type == VK_DISCRETE_GPU) gpus++;
                }
                return OptionalInt.of(gpus);
            } finally {
                destroy.invokeExact(instance, MemorySegment.NULL);
            }
        }
    }

    /** hipGetDeviceCount; no device (100) and an old driver (35) mean none. */
    private static OptionalInt hipGpus() throws Throwable {
        return runtimeCount(List.of("libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"), null, "hipGetDeviceCount");
    }

    /** The CUDA driver API's cuInit then cuDeviceGetCount; no driver at all means none. */
    private static OptionalInt cudaGpus() throws Throwable {
        if (!loadable("libcuda.so.1")) return OptionalInt.of(0);
        return runtimeCount(List.of("libcuda.so.1"), "cuInit", "cuDeviceGetCount");
    }

    private static OptionalInt runtimeCount(List<String> names, String init, String count) throws Throwable {
        String found = names.stream().filter(GpuCheck::loadable).findFirst().orElse(null);
        if (found == null) return OptionalInt.empty();
        try (Arena arena = Arena.ofConfined()) {
            SymbolLookup lib = SymbolLookup.libraryLookup(found, arena);
            if (init != null) {
                int rc = (int) handle(lib, init, FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)).invokeExact(0);
                if (rc == 100 || rc == 35) return OptionalInt.of(0);
                if (rc != 0) return OptionalInt.empty();
            }
            MemorySegment n = arena.allocate(ValueLayout.JAVA_INT);
            int rc = (int) handle(lib, count, FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS)).invokeExact(n);
            if (rc == 100 || rc == 35) return OptionalInt.of(0);
            return rc == 0 ? OptionalInt.of(n.get(ValueLayout.JAVA_INT, 0)) : OptionalInt.empty();
        }
    }

    private static MethodHandle handle(SymbolLookup lib, String name, FunctionDescriptor fd) {
        return LINKER.downcallHandle(lib.find(name).orElseThrow(), fd);
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
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT).startsWith("linux");
    }
}
