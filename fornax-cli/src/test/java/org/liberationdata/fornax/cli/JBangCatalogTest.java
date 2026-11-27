package org.liberationdata.fornax.cli;

import org.junit.jupiter.api.Test;
import org.liberationdata.fornax.Json;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/** The repository's jbang-catalog.json: every alias runs this CLI, on Java 22+ with native access, with a real command. */
class JBangCatalogTest {
    static final Set<String> ALIAS_FIELDS = Set.of("script-ref", "description", "arguments", "java", "java-options");

    @Test
    void everyAliasRunsAFornaxCommand() throws IOException {
        Path file = Path.of(System.getProperty("fornax.jbang.catalog", "../jbang-catalog.json"));
        Map<?, ?> aliases = (Map<?, ?>) Json.get(Json.parse(Files.readString(file)), "aliases");
        assertNotNull(aliases, "no aliases in " + file);
        assertTrue(aliases.containsKey("fornax"));
        for (var e : aliases.entrySet()) {
            String name = (String) e.getKey();
            Map<?, ?> alias = (Map<?, ?>) e.getValue();
            assertTrue(ALIAS_FIELDS.containsAll(alias.keySet()), name + " has a field JBang doesn't know: " + alias.keySet());
            assertTrue(((String) alias.get("script-ref")).startsWith("org.fornax4j:fornax-cli:"), name);
            assertEquals("22+", alias.get("java"), name);  // options.release in build.gradle
            assertEquals(List.of("--enable-native-access=ALL-UNNAMED"), alias.get("java-options"), name);
            if (alias.get("arguments") instanceof List<?> args && !args.isEmpty()) {
                String command = (String) args.get(0);
                PrintStream sink = new PrintStream(new ByteArrayOutputStream(), true, StandardCharsets.UTF_8);
                assertEquals(0, Main.run(new String[] {command, "--help"}, sink, sink), name + ": no fornax " + command);
            }
        }
    }
}
