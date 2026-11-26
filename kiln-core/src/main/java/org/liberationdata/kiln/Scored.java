package org.liberationdata.kiln;

import java.util.Comparator;

/**
 * An item's score: {@code index} is its position in the list that was scored (documents, embeddings), and a higher
 * score is a better match. {@link Vectors#topK} gives cosine similarities; a reranker gives its own relevance scores.
 */
public record Scored(int index, float score) {
    /** Best first; equal scores in index order. */
    public static final Comparator<Scored> BEST_FIRST =
            Comparator.comparingDouble((Scored s) -> s.score).reversed().thenComparingInt(Scored::index);
}
