package org.liberationdata.fornax.cli;

import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class InfoTest {
    @Test
    void theNativesJarsLeaveOutTheNativesPomJBangPutsOnTheClasspath() {
        String m2 = "/home/u/.m2/repository/org/fornax4j/";
        String classpath = String.join(File.pathSeparator,
                m2 + "fornax-cli/0.2.0/fornax-cli-0.2.0.jar",
                m2 + "fornax-natives/0.2.0/fornax-natives-0.2.0.pom",
                m2 + "fornax-natives-linux-x86_64-vulkan/0.2.0/fornax-natives-linux-x86_64-vulkan-0.2.0.jar");
        assertEquals(List.of("fornax-natives-linux-x86_64-vulkan-0.2.0.jar"), Info.nativesJars(classpath));
    }
}
