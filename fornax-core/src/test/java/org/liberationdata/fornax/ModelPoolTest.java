package org.liberationdata.fornax;

import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class ModelPoolTest {

    static final class Fake implements AutoCloseable {
        final String name;
        final List<String> log;
        boolean closed;

        Fake(String name, List<String> log) { this.name = name; this.log = log; log.add("load " + name); }

        @Override public void close() { closed = true; log.add("close " + name); }
    }

    @Test
    void keepsModelLoadedBetweenLeases() {
        List<String> log = new ArrayList<>();
        try (var pool = new ModelPool(100, null)) {
            Fake first, second;
            try (var l = pool.acquire("a", 10, () -> new Fake("a", log))) { first = l.model(); }
            try (var l = pool.acquire("a", 10, () -> new Fake("a", log))) { second = l.model(); }
            assertSame(first, second);
            assertEquals(List.of("load a"), log);
            assertEquals(10, pool.loadedBytes());
        }
        assertEquals(List.of("load a", "close a"), log);
    }

    @Test
    void evictsLeastRecentlyUsedIdleModelToFitBudget() {
        List<String> log = new ArrayList<>();
        try (var pool = new ModelPool(100, null)) {
            pool.acquire("a", 40, () -> new Fake("a", log)).close();
            pool.acquire("b", 40, () -> new Fake("b", log)).close();
            pool.acquire("a", 40, () -> new Fake("a", log)).close();  // a is now more recent than b
            pool.acquire("c", 40, () -> new Fake("c", log)).close();  // needs room: b goes
            assertEquals(List.of("load a", "load b", "close b", "load c"), log);
            assertEquals(80, pool.loadedBytes());
        }
    }

    @Test
    void neverEvictsAModelInUse() {
        List<String> log = new ArrayList<>();
        try (var pool = new ModelPool(50, null)) {
            var held = pool.acquire("a", 40, () -> new Fake("a", log));
            pool.acquire("b", 40, () -> new Fake("b", log)).close();  // over budget, but a is busy
            assertFalse(held.model().closed);
            assertEquals(80, pool.loadedBytes());
            held.close();
        }
    }

    @Test
    void oneUserAtATime() throws Exception {
        try (var pool = new ModelPool(100, null)) {
            var lease = pool.acquire("a", 1, () -> new Fake("a", new ArrayList<>()));
            var acquired = new CountDownLatch(1);
            Thread t = Thread.ofPlatform().start(() -> {
                try (var l = pool.acquire("a", 1, () -> { throw new AssertionError("reloaded"); })) { acquired.countDown(); }
            });
            assertFalse(acquired.await(200, TimeUnit.MILLISECONDS), "second user got in while the first held it");
            lease.close();
            assertTrue(acquired.await(5, TimeUnit.SECONDS));
            t.join();
        }
    }

    @Test
    void unloadsIdleModelsAfterTimeout() throws Exception {
        List<String> log = new ArrayList<>();
        try (var pool = new ModelPool(100, Duration.ofSeconds(1))) {
            pool.acquire("a", 10, () -> new Fake("a", log)).close();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (pool.loadedBytes() > 0 && System.nanoTime() < deadline) Thread.sleep(50);
            assertEquals(0, pool.loadedBytes());
            assertEquals(List.of("load a", "close a"), log);
        }
    }

    @Test
    void failedLoadReleasesTheSlot() {
        AtomicInteger loads = new AtomicInteger();
        try (var pool = new ModelPool(100, null)) {
            assertThrows(IllegalStateException.class, () -> pool.acquire("a", 1, () -> {
                loads.incrementAndGet();
                throw new IllegalStateException("bad weights");
            }));
            try (var l = pool.acquire("a", 1, () -> { loads.incrementAndGet(); return new Fake("a", new ArrayList<>()); })) {
                assertNotNull(l.model());
            }
            assertEquals(2, loads.get());
        }
    }
}
