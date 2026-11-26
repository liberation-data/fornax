package org.liberationdata.fornax;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Similarity search over embeddings in memory, for a few thousand to a few hundred thousand vectors (a linear scan;
 * no index, no vector database). With L2-normalised embeddings (what {@code LlamaModel.embed} returns by default)
 * cosine similarity and the dot product are the same number.
 */
public final class Vectors {
    private Vectors() {}

    public static double dot(float[] a, float[] b) {
        requireSameLength(a, b);
        double s = 0;
        for (int i = 0; i < a.length; i++) s += (double) a[i] * b[i];
        return s;
    }

    /** Cosine similarity, -1 to 1; 0 if either vector is all zeros. */
    public static double cosine(float[] a, float[] b) {
        requireSameLength(a, b);
        double ab = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.length; i++) {
            ab += (double) a[i] * b[i];
            aa += (double) a[i] * a[i];
            bb += (double) b[i] * b[i];
        }
        return aa == 0 || bb == 0 ? 0 : ab / Math.sqrt(aa * bb);
    }

    /** A copy scaled to length 1 (all zeros stays all zeros). */
    public static float[] normalize(float[] v) {
        double s = 0;
        for (float x : v) s += (double) x * x;
        float[] out = new float[v.length];
        if (s == 0) return out;
        double k = 1 / Math.sqrt(s);
        for (int i = 0; i < v.length; i++) out[i] = (float) (v[i] * k);
        return out;
    }

    /** The k vectors most similar to the query by cosine, best first (ties in index order). */
    public static List<Scored> topK(float[] query, float[][] vectors, int k) {
        if (k < 0) throw new IllegalArgumentException("k must be >= 0: " + k);
        if (k == 0) return List.of();
        // a min-heap of the best k so far: the worst of them on top, ready to be replaced
        PriorityQueue<Scored> best = new PriorityQueue<>(Math.min(k, Math.max(1, vectors.length)) + 1, Scored.BEST_FIRST.reversed());
        for (int i = 0; i < vectors.length; i++) {
            best.add(new Scored(i, (float) cosine(query, vectors[i])));
            if (best.size() > k) best.poll();
        }
        List<Scored> out = new ArrayList<>(best);
        out.sort(Scored.BEST_FIRST);
        return out;
    }

    /** Scores by position (scores[i] is item i's), as a list best first; ties in index order. */
    public static List<Scored> rank(float[] scores) {
        List<Scored> out = new ArrayList<>(scores.length);
        for (int i = 0; i < scores.length; i++) out.add(new Scored(i, scores[i]));
        out.sort(Scored.BEST_FIRST);
        return out;
    }

    private static void requireSameLength(float[] a, float[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("vector lengths differ: " + a.length + " and " + b.length);
    }
}
