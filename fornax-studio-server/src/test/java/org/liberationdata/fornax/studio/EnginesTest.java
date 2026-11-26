package org.liberationdata.fornax.studio;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CyclicBarrier;

import static org.junit.jupiter.api.Assertions.*;

class EnginesTest {
    /** stable-diffusion.cpp has one progress callback for the process: two runs at once each see only their own steps. */
    @Test void eachRunGetsItsOwnSteps() throws Exception {
        var barrier = new CyclicBarrier(2);
        List<String> a = new CopyOnWriteArrayList<>(), b = new CopyOnWriteArrayList<>();
        Thread ta = Thread.ofPlatform().start(() -> run(barrier, "a", a));
        Thread tb = Thread.ofPlatform().start(() -> run(barrier, "b", b));
        ta.join();
        tb.join();
        assertEquals(List.of("a1", "a2", "a3"), a);
        assertEquals(List.of("b1", "b2", "b3"), b);
        // Outside a run (a model loading, say), steps go nowhere.
        Engines.PROGRESS.progress(1, 1, 0f);
    }

    private static void run(CyclicBarrier barrier, String name, List<String> seen) {
        Engines.stepping((s, n, t) -> seen.add(name + s), () -> {
            for (int s = 1; s <= 3; s++) {
                try { barrier.await(); } catch (Exception e) { throw new IllegalStateException(e); }
                Engines.PROGRESS.progress(s, 3, 0f);  // as the native callback does, on the thread generating
            }
            return null;
        });
    }
}
