package org.liberationdata.kiln.demucs;

import org.junit.jupiter.api.Test;
import org.liberationdata.kiln.Audio;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class StemsTest {
    static Audio a(float l, float r) { return new Audio(new float[] {l, l}, new float[] {r, r}, 44100); }

    static Stems fourStems() {
        return Stems.of(List.of("drums", "bass", "other", "vocals"), List.of(a(1, 2), a(10, 20), a(100, 200), a(1000, 2000)));
    }

    @Test
    void namesKeepTheModelsOrder() {
        Stems s = fourStems();
        assertEquals(List.of("drums", "bass", "other", "vocals"), s.names());
        assertEquals(44100, s.sampleRate());
        assertEquals(10f, s.get("bass").left()[0]);
        assertTrue(s.has("vocals"));
        assertFalse(s.has("guitar"));
    }

    @Test
    void backingIsEveryStemButTheVocals() {
        Stems s = fourStems();
        assertEquals(1000f, s.vocals().left()[1]);
        assertEquals(111f, s.backing().left()[0]);
        assertEquals(222f, s.backing().right()[1]);
    }

    @Test
    void mixIsTheSumOfEveryStem() {
        assertEquals(1111f, fourStems().mix().left()[0]);
        assertEquals(2222f, fourStems().mix().right()[0]);
    }

    /** A vocals/instrumental model (a RoFormer) names its backing itself. */
    @Test
    void anInstrumentalStemIsTheBacking() {
        Stems s = Stems.of(List.of("vocals", "instrumental"), List.of(a(1, 1), a(5, 6)));
        assertEquals(5f, s.backing().left()[0]);
    }

    @Test
    void aStemTheModelDoesNotHaveIsRefused() {
        var e = assertThrows(IllegalArgumentException.class, () -> fourStems().get("piano"));
        assertTrue(e.getMessage().contains("drums, bass, other, vocals"), e.getMessage());
        Stems noVocals = Stems.of(List.of("noise", "denoised"), List.of(a(1, 1), a(2, 2)));
        assertThrows(IllegalArgumentException.class, noVocals::vocals);
        assertThrows(IllegalArgumentException.class, noVocals::backing);
    }

    @Test
    void namesAndAudioMustPair() {
        assertThrows(IllegalArgumentException.class, () -> Stems.of(List.of("vocals"), List.of()));
        assertThrows(IllegalArgumentException.class, () -> Stems.of(List.of(), List.of()));
        // a repeated name would silently drop a stem from names(), backing() and mix()
        var e = assertThrows(IllegalArgumentException.class, () -> Stems.of(List.of("vocals", "vocals"), List.of(a(1, 1), a(2, 2))));
        assertTrue(e.getMessage().contains("vocals"), e.getMessage());
    }

    /** backing() and mix() add the stems sample by sample, so they must line up. */
    @Test
    void stemsMustShareOneRateAndLength() {
        Audio shorter = new Audio(new float[] {1}, new float[] {1}, 44100), slower = new Audio(new float[2], new float[2], 22050);
        var e = assertThrows(IllegalArgumentException.class, () -> Stems.of(List.of("vocals", "drums"), List.of(a(1, 1), shorter)));
        assertTrue(e.getMessage().contains("drums"), e.getMessage());
        assertThrows(IllegalArgumentException.class, () -> Stems.of(List.of("vocals", "drums"), List.of(a(1, 1), slower)));
        Audio shortRight = new Audio(new float[2], new float[1], 44100);
        assertThrows(IllegalArgumentException.class, () -> Stems.of(List.of("vocals", "drums"), List.of(a(1, 1), shortRight)));
    }
}
