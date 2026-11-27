package org.liberationdata.fornax;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeLibraryNotFoundTest {

    @Test
    void anIntelMacIsToldThereAreNoNativesForIt() {
        String m = NativeLibrary.notFound("fornax_llama", "libfornax_llama.dylib", List.of("/x"), "macos-x86_64");
        assertTrue(m.contains("Apple silicon only, not Intel Macs"), m);
        assertTrue(m.contains("native/build.sh metal"), m);
        assertFalse(m.contains("fornax-natives-<os>"), m);
    }

    @Test
    void elsewhereItNamesTheNativesDependency() {
        for (String platform : List.of("macos-aarch64", "linux-x86_64", "windows-x86_64")) {
            String m = NativeLibrary.notFound("fornax_llama", "libfornax_llama.so", List.of("/x"), platform);
            assertTrue(m.contains("fornax-natives-<os>-<arch>-<backend>"), m);
            assertTrue(m.contains("-Dfornax.fornax_llama.lib=<file>"), m);
            assertFalse(m.contains("Intel"), m);
        }
    }
}
