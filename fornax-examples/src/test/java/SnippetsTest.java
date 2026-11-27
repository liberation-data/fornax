import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * scripts/Snippets.java, which fills the docs' Java blocks from the snippet sources. In the default package because the
 * script is: a single-file program has no package.
 */
class SnippetsTest {
    static final String SOURCE = """
            import java.util.List;
            // --8<-- [start:a]
            import java.util.Map;
            // --8<-- [end:a]

            class A {
                void a() {
                    // --8<-- [start:a]
                    int x = 1;
                    // --8<-- [start:b]
                    int y = 2;
                    // --8<-- [end:b]
                    int hidden = 3;   // --8<-- [skip]
                    // --8<-- [end:a]
                }
            }
            """;

    final List<String> problems = new ArrayList<>();

    /** The doc filled from {@link #SOURCE} as Snippets file {@code A}. */
    String fill(String doc) throws Exception {
        Map<String, Map<String, String>> sources = new HashMap<>(Map.of("A", Snippets.regions(SOURCE.lines().toList())));
        return Snippets.fill(Path.of("doc.md"), doc, sources, problems);
    }

    @Test
    void aRegionIsItsStretchesDedentedAndJoinedWithoutMarkersOrSkippedLines() {
        Map<String, String> r = Snippets.regions(SOURCE.lines().toList());
        assertEquals("import java.util.Map;\n\nint x = 1;\nint y = 2;", r.get("a"));
        assertEquals("int y = 2;", r.get("b"));
    }

    @Test
    void aRegionThatNeverEndsIsAnError() {
        var e = assertThrows(IllegalArgumentException.class,
                () -> Snippets.regions(List.of("// --8<-- [start:a]", "int x;", "// --8<-- [end:b]", "}")));
        assertEquals("1: [start:a] has no [end:a]", e.getMessage());
        e = assertThrows(IllegalArgumentException.class,
                () -> Snippets.regions(List.of("x", "// --8<-- [start:a]", "// --8<-- [start:a]", "// --8<-- [end:a]")));
        assertEquals("2: [start:a] has no [end:a]", e.getMessage());
    }

    @Test
    void aMarkedBlockGetsItsRegion() throws Exception {
        assertEquals("<!-- snippet: A#b -->\n```java\nint y = 2;\n```\n", fill("<!-- snippet: A#b -->\n```java\nold\n```\n"));
        assertEquals("{/* snippet: A#b */}\n\n```java title=\"A.java\"\nint y = 2;\n```",
                fill("{/* snippet: A#b */}\n\n```java title=\"A.java\"\n```"));
        assertEquals(List.of(), problems);
    }

    @Test
    void anIndentedBlockKeepsItsIndentExceptOnBlankLines() throws Exception {
        assertEquals("1. Step\n   <!-- snippet: A#a -->\n   ```java\n   import java.util.Map;\n\n   int x = 1;\n   int y = 2;\n   ```\n",
                fill("1. Step\n   <!-- snippet: A#a -->\n   ```java\n   old\n   ```\n"));
        assertEquals(List.of(), problems);
    }

    @Test
    void longerBacktickAndTildeFencesAreFencesToo() throws Exception {
        assertEquals("<!-- snippet: A#b -->\n````java\nint y = 2;\n````\n", fill("<!-- snippet: A#b -->\n````java\n```\n````\n"));
        assertEquals("<!-- snippet: A#b -->\n~~~ java\nint y = 2;\n~~~\n", fill("<!-- snippet: A#b -->\n~~~ java\n~~~\n"));
        assertEquals(List.of(), problems);
    }

    @Test
    void markersAndJavaFencesInsideAnotherFenceAreText() throws Exception {
        String doc = "```md\n<!-- snippet: A#b -->\n```java\nold\n```\n\n~~~md\n```java\n~~~\n\n````md\n```java\n```\n````\n";
        assertEquals(doc, fill(doc));
        assertEquals(List.of(), problems);
    }

    @Test
    void aJavaFenceAfterALongerOtherFenceIsStillChecked() throws Exception {
        fill("````md\n```java\n```\n````\n\n```java\nunmarked\n```\n");
        assertEquals(List.of("doc.md:6: ```java block with no snippet marker before it"), problems);
    }

    @Test
    void crlfLineEndingsAreKept() throws Exception {
        assertEquals("<!-- snippet: A#b -->\r\n```java\r\nint y = 2;\r\n```\r\n", fill("<!-- snippet: A#b -->\r\n```java\r\nold\r\n```\r\n"));
        assertEquals(List.of(), problems);
    }

    @Test
    void aJavaBlockNeedsAMarker() throws Exception {
        String doc = "text\n```java\nold\n```\n";
        assertEquals(doc, fill(doc));
        assertEquals(List.of("doc.md:2: ```java block with no snippet marker before it"), problems);
    }

    @Test
    void aMarkerNeedsAJavaBlockNext() throws Exception {
        fill("<!-- snippet: A#a -->\ntext\n\n<!-- snippet: A#a -->\n<!-- snippet: A#b -->\n```java\n```\n\n"
                + "<!-- snippet: A#a -->\n```kotlin\n```\n```java\n```\n\n<!-- snippet: A#a -->");
        assertEquals(List.of(
                "doc.md:1: snippet marker with no ```java block after it",
                "doc.md:4: snippet marker with no ```java block after it",
                "doc.md:9: snippet marker with no ```java block after it",
                "doc.md:12: ```java block with no snippet marker before it",
                "doc.md:15: snippet marker with no ```java block after it"), problems);
    }

    @Test
    void aMissingRegionOrFileIsAProblemAndTheBlockIsLeftAlone() throws Exception {
        String doc = "<!-- snippet: A#nope -->\n```java\nold\n```\n<!-- snippet: Nope#a -->\n```java\nold\n```\n";
        assertEquals(doc, fill(doc));
        assertEquals(2, problems.size());
        assertTrue(problems.get(0).startsWith("doc.md:1: no region nope in "), problems.get(0));
        assertTrue(problems.get(1).startsWith("doc.md:5: no region a in "), problems.get(1));
    }

    @Test
    void anUnclosedJavaBlockIsAProblem() throws Exception {
        String doc = "<!-- snippet: A#b -->\n```java\nold\n";
        assertEquals(doc, fill(doc));
        assertEquals(List.of("doc.md:2: unclosed ```java block"), problems);
    }

    /** A copy of the repository's layout with one doc, one snippet and the script. */
    static Path repo(Path dir, String doc) throws Exception {
        Path sources = Files.createDirectories(dir.resolve("fornax-examples/src/snippets/java/org/liberationdata/fornax/snippets"));
        Files.writeString(sources.resolve("A.java"), SOURCE);
        Files.createDirectories(dir.resolve("website/src/content/docs"));
        Files.writeString(dir.resolve("website/src/content/docs/page.mdx"), "{/* snippet: A#b */}\n```java\nint y = 2;\n```\n");
        Files.writeString(dir.resolve("README.md"), doc);
        Files.copy(Path.of("../scripts/Snippets.java"), Files.createDirectories(dir.resolve("scripts")).resolve("Snippets.java"));
        return dir;
    }

    /** The script's exit status, run from {@code cwd} as {@code java scripts/Snippets.java}, but by absolute path. */
    @Test
    void anUnknownOptionIsRefusedWithoutWritingTheDoc(@TempDir Path dir) throws Exception {
        String stale = "<!-- snippet: A#b -->\n```java\nold\n```\n";
        Path repo = repo(dir, stale);
        assertEquals(2, run(repo, repo, "--chek"));
        assertEquals(stale, Files.readString(repo.resolve("README.md")));
    }

    static int run(Path repo, Path cwd, String... args) throws Exception {
        List<String> cmd = new ArrayList<>(List.of(ProcessHandle.current().info().command().orElseThrow(),
                repo.resolve("scripts/Snippets.java").toString()));
        cmd.addAll(List.of(args));
        Process p = new ProcessBuilder(cmd).directory(cwd.toFile()).inheritIO().start();
        assertTrue(p.waitFor(2, TimeUnit.MINUTES));
        return p.exitValue();
    }

    @Test
    void checkFailsOnAStaleDocWithoutWritingItAndSyncFixesItFromAnyDirectory(@TempDir Path dir) throws Exception {
        String stale = "<!-- snippet: A#b -->\n```java\nold\n```\n";
        Path repo = repo(dir, stale), elsewhere = Files.createDirectories(dir.resolve("website"));
        assertEquals(1, run(repo, elsewhere, "--check"));
        assertEquals(stale, Files.readString(repo.resolve("README.md")));
        assertEquals(0, run(repo, elsewhere));
        assertEquals("<!-- snippet: A#b -->\n```java\nint y = 2;\n```\n", Files.readString(repo.resolve("README.md")));
        assertEquals(0, run(repo, elsewhere, "--check"));
    }

    @Test
    void aSourceWithARegionThatNeverEndsFailsWithoutWritingTheDoc(@TempDir Path dir) throws Exception {
        String stale = "<!-- snippet: A#b -->\n```java\nold\n```\n";
        Path repo = repo(dir, stale);
        Path a = repo.resolve("fornax-examples/src/snippets/java/org/liberationdata/fornax/snippets/A.java");
        Files.writeString(a, SOURCE.replace("[end:b]", "[end:B]"));
        assertEquals(1, run(repo, repo));
        assertEquals(stale, Files.readString(repo.resolve("README.md")));
    }

    @Test
    void aProblemFailsEvenWithoutCheck(@TempDir Path dir) throws Exception {
        Path repo = repo(dir, "```java\nunmarked\n```\n");
        assertEquals(1, run(repo, repo));
        assertEquals(1, run(repo, repo, "--check"));
    }
}
