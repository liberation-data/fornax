package org.liberationdata.kiln;

import org.junit.jupiter.api.Test;

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
}
