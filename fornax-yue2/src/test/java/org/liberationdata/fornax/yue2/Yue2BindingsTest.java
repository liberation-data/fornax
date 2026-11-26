package org.liberationdata.fornax.yue2;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/** The shim exports every function Yue2Model binds, fornax_yue2_set_cancel included (no model needed). */
class Yue2BindingsTest {
    @Test
    void bindsEveryFunction() {
        String dir = System.getProperty("fornax.library.path");
        assumeTrue(dir != null && new java.io.File(dir, System.mapLibraryName("fornax_yue2")).isFile(), "native library not built");
        // Binding needs no GPU, and CI runners have none: don't let Fornax's GPU check refuse the library.
        String allowCpu = System.setProperty("fornax.allowCpu", "true");
        try {
            assertDoesNotThrow(() -> Class.forName(Yue2Model.class.getName()));
        } finally {
            if (allowCpu == null) System.clearProperty("fornax.allowCpu");
            else System.setProperty("fornax.allowCpu", allowCpu);
        }
    }
}
