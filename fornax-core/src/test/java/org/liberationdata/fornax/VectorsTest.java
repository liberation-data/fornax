package org.liberationdata.fornax;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class VectorsTest {
    @Test
    void dotAndCosine() {
        float[] a = {1, 2, 3}, b = {4, 5, 6};
        assertEquals(32, Vectors.dot(a, b), 1e-6);
        assertEquals(32 / (Math.sqrt(14) * Math.sqrt(77)), Vectors.cosine(a, b), 1e-6);
        assertEquals(1, Vectors.cosine(a, new float[] {2, 4, 6}), 1e-6);
        assertEquals(-1, Vectors.cosine(a, new float[] {-1, -2, -3}), 1e-6);
        assertEquals(0, Vectors.cosine(new float[] {1, 0}, new float[] {0, 1}), 1e-6);
    }

    @Test
    void cosineOfAZeroVectorIsZero() {
        assertEquals(0, Vectors.cosine(new float[] {0, 0}, new float[] {1, 1}));
    }

    @Test
    void lengthsMustMatch() {
        assertThrows(IllegalArgumentException.class, () -> Vectors.dot(new float[2], new float[3]));
        assertThrows(IllegalArgumentException.class, () -> Vectors.cosine(new float[2], new float[3]));
        assertThrows(IllegalArgumentException.class, () -> Vectors.topK(new float[2], new float[][] {new float[3]}, 1));
    }

    @Test
    void normalizeReturnsAUnitCopy() {
        float[] a = {3, 4};
        float[] n = Vectors.normalize(a);
        assertArrayEquals(new float[] {0.6f, 0.8f}, n, 1e-6f);
        assertArrayEquals(new float[] {3, 4}, a);  // untouched
        assertArrayEquals(new float[] {0, 0}, Vectors.normalize(new float[] {0, 0}));
    }

    @Test
    void topKRanksByCosineBestFirst() {
        float[] q = {1, 0};
        float[][] docs = {{0, 1}, {1, 0.1f}, {-1, 0}, {1, 1}, {2, 0}};
        List<Scored> top = Vectors.topK(q, docs, 3);
        assertEquals(List.of(4, 1, 3), top.stream().map(Scored::index).toList());
        assertEquals(1, top.get(0).score(), 1e-6);
        assertTrue(top.get(0).score() >= top.get(1).score() && top.get(1).score() >= top.get(2).score());
    }

    @Test
    void topKTakesAtMostWhatThereIs() {
        assertEquals(2, Vectors.topK(new float[] {1}, new float[][] {{1}, {2}}, 10).size());
        assertEquals(List.of(), Vectors.topK(new float[] {1}, new float[][] {{1}}, 0));
        assertEquals(List.of(), Vectors.topK(new float[] {1}, new float[0][], 3));
        assertThrows(IllegalArgumentException.class, () -> Vectors.topK(new float[] {1}, new float[][] {{1}}, -1));
    }

    @Test
    void topKBreaksTiesByIndex() {
        List<Scored> top = Vectors.topK(new float[] {1, 0}, new float[][] {{1, 1}, {2, 0}, {1, 1}, {3, 0}}, 4);
        assertEquals(List.of(1, 3, 0, 2), top.stream().map(Scored::index).toList());
    }

    @Test
    void rankSortsScoresBestFirst() {
        List<Scored> ranked = Vectors.rank(new float[] {0.1f, 0.9f, 0.5f, 0.9f});
        assertEquals(List.of(1, 3, 2, 0), ranked.stream().map(Scored::index).toList());
        assertEquals(0.9f, ranked.get(0).score());
    }
}
