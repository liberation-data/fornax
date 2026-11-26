package org.liberationdata.fornax.examples;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SearchTest {
    @Test
    void aPassageStartsAtItsFirstLineOfText(@TempDir Path dir) throws Exception {
        String a = "a".repeat(900), b = "b".repeat(100);
        Files.writeString(dir.resolve("notes.md"), "\n\n" + a + "\n\n" + b + "\n");
        List<Search.Passage> p = Search.passages(dir);
        assertEquals(List.of(3, 5), p.stream().map(Search.Passage::line).toList());
        assertEquals(List.of(a, b), p.stream().map(Search.Passage::text).toList());
    }

    @Test
    void shortParagraphsMergeAndAVeryLongLineIsSplit(@TempDir Path dir) throws Exception {
        Files.writeString(dir.resolve("short.txt"), "one\n\ntwo\n");
        Files.writeString(dir.resolve("long.txt"), "x".repeat(4000) + "\n");
        Files.writeString(dir.resolve("skipped.java"), "not text");
        List<Search.Passage> p = Search.passages(dir);
        assertEquals(List.of("long.txt", "long.txt", "long.txt", "short.txt"),
                p.stream().map(x -> x.file().getFileName().toString()).toList());
        assertEquals(List.of(1600, 1600, 800), p.subList(0, 3).stream().map(x -> x.text().length()).toList());
        assertEquals("one\n\ntwo", p.get(3).text());
        assertEquals(1, p.get(3).line());
    }

    @Test
    void aFileThatIsntUtf8IsSkipped(@TempDir Path dir) throws Exception {
        Files.write(dir.resolve("latin1.txt"), new byte[] {(byte) 0xE9, (byte) 0xFF, '\n'});
        Files.writeString(dir.resolve("ok.md"), "fine\n");
        assertEquals(List.of("fine"), Search.passages(dir).stream().map(Search.Passage::text).toList());
    }
}
