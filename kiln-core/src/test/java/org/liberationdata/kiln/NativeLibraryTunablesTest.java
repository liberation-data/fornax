package org.liberationdata.kiln;

import org.junit.jupiter.api.Test;

import java.util.Properties;

import static org.junit.jupiter.api.Assertions.assertEquals;

class NativeLibraryTunablesTest {
    @Test
    void setsADefaultThatNativeCodeSees() {
        NativeLibrary.Tunables.setDefault("KILN_TEST_TUNABLE", "8");
        assertEquals("8", NativeLibrary.Tunables.nativeEnv("KILN_TEST_TUNABLE"));
    }

    @Test
    void turnsGpuGraphsOffUnlessAsked() {
        NativeLibrary.Tunables.apply();
        String expected = System.getenv("GGML_CUDA_DISABLE_GRAPHS") != null ? System.getenv("GGML_CUDA_DISABLE_GRAPHS") : "1";
        assertEquals(expected, NativeLibrary.Tunables.nativeEnv("GGML_CUDA_DISABLE_GRAPHS"));
    }

    @Test
    void passesVulkanNodesPerSubmitOn() {
        Properties p = new Properties();
        p.setProperty("kiln.vulkan.maxNodesPerSubmit", "8");
        NativeLibrary.Tunables.apply(p);
        String expected = System.getenv("GGML_VK_MAX_NODES_PER_SUBMIT") != null ? System.getenv("GGML_VK_MAX_NODES_PER_SUBMIT") : "8";
        assertEquals(expected, NativeLibrary.Tunables.nativeEnv("GGML_VK_MAX_NODES_PER_SUBMIT"));
    }

    @Test
    void leavesAnExistingVariableAlone() {
        NativeLibrary.Tunables.setDefault("KILN_TEST_KEPT", "first");
        NativeLibrary.Tunables.setDefault("KILN_TEST_KEPT", "second");
        assertEquals("first", NativeLibrary.Tunables.nativeEnv("KILN_TEST_KEPT"));
    }
}
