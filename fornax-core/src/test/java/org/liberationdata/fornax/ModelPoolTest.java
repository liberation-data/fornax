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

    /** Two different models loading at once: the second counts the first's size, and so makes room for both (#68). */
    @Test
    void aModelBeingLoadedCountsAgainstTheBudget() throws Exception {
        List<String> log = java.util.Collections.synchronizedList(new ArrayList<>());
        try (var pool = new ModelPool(100, null)) {
            pool.acquire("idle", 40, () -> new Fake("idle", log)).close();
            var bLoading = new CountDownLatch(1);
            var release = new CountDownLatch(1);
            Thread b = Thread.ofPlatform().start(() -> {
                try (var l = pool.acquire("b", 40, () -> {
                    bLoading.countDown();
                    try { release.await(); } catch (InterruptedException e) { throw new IllegalStateException(e); }
                    return new Fake("b", log);
                })) { assertNotNull(l.model()); }
            });
            try {
                assertTrue(bLoading.await(10, TimeUnit.SECONDS));
                // b's 40 is on its way: 40 idle + 40 b + 40 c is over 100, so the idle model goes before c loads
                try (var c = pool.acquire("c", 40, () -> new Fake("c", log))) {
                    assertNotNull(c.model());
                    assertTrue(log.contains("close idle"), log.toString());
                }
            } finally {
                release.countDown();   // even on failure: the pool can't close while b holds its slot
                b.join(10_000);
            }
            assertEquals(80, pool.loadedBytes());
        }
    }

    @Test
    void aFailedLoadGivesBackItsReservation() {
        List<String> log = new ArrayList<>();
        try (var pool = new ModelPool(100, null)) {
            pool.acquire("idle", 40, () -> new Fake("idle", log)).close();
            assertThrows(IllegalStateException.class, () -> pool.acquire("broken", 50, () -> { throw new IllegalStateException("no"); }));
            // nothing held for the failed load: 40 idle + 60 fits
            pool.acquire("c", 60, () -> new Fake("c", log)).close();
            assertEquals(List.of("load idle", "load c"), log);
            assertEquals(100, pool.loadedBytes());
        }
    }

    /** A loader that borrows another model (budgetLock isn't held while a loader runs), which counts the outer load. */
    @Test
    void aLoaderCanBorrowAnotherModel() {
        List<String> log = new ArrayList<>();
        try (var pool = new ModelPool(100, null)) {
            pool.acquire("idle", 40, () -> new Fake("idle", log)).close();
            try (var outer = pool.acquire("outer", 40, () -> {
                // outer's 40 is on its way: 40 idle + 40 outer + 40 inner is over 100, so the idle model goes
                try (var inner = pool.acquire("inner", 40, () -> new Fake("inner", log))) { assertNotNull(inner.model()); }
                return new Fake("outer", log);
            })) {
                assertNotNull(outer.model());
            }
            assertEquals(List.of("load idle", "close idle", "load inner", "load outer"), log);
            assertEquals(80, pool.loadedBytes());
        }
    }

    // ---- concurrency: a model several requests use at once (a language model with slots) ----
    // (a timeout each: a lease a failed assertion leaves open would make the pool's close wait for ever)

    @Test @org.junit.jupiter.api.Timeout(20)
    void aModelWithConcurrencyHasThatManyUsersAtOnceAndLoadsOnce() throws Exception {
        List<String> log = java.util.Collections.synchronizedList(new ArrayList<>());
        AtomicInteger loads = new AtomicInteger();
        try (var pool = new ModelPool(100, null)) {
            CountDownLatch held = new CountDownLatch(3), release = new CountDownLatch(1);
            List<Thread> users = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                users.add(Thread.ofPlatform().start(() -> {
                    try (var l = pool.acquire("a", 10, 3, () -> { loads.incrementAndGet(); sleep(50); return new Fake("a", log); })) {
                        assertFalse(l.model().closed);
                        held.countDown();
                        release.await();
                    } catch (InterruptedException e) { throw new RuntimeException(e); }
                }));
            }
            assertTrue(held.await(5, TimeUnit.SECONDS), "three users at once");
            assertEquals(1, loads.get(), "the first lease loaded it; the others waited for that load");
            assertTrue(pool.entries().getFirst().inUse());
            var fourth = Thread.ofPlatform().start(() -> pool.acquire("a", 10, 3, () -> new Fake("a", log)).close());
            fourth.join(200);
            assertTrue(fourth.isAlive(), "a fourth waits for one of the three");
            release.countDown();
            fourth.join(5000);
            for (Thread t : users) t.join(5000);
            assertFalse(pool.entries().getFirst().inUse());
        }
        assertEquals(List.of("load a", "close a"), log);
    }

    @Test @org.junit.jupiter.api.Timeout(20)
    void aModelWithAnyUserLeftIsNeverEvictedOrUnloadedUnderIt() throws Exception {
        List<String> log = new ArrayList<>();
        try (var pool = new ModelPool(50, null)) {
            var one = pool.acquire("a", 40, 2, () -> new Fake("a", log));
            Fake a = one.model();
            pool.acquire("a", 40, 2, () -> new Fake("a", log)).close();   // the other user has gone; one remains
            pool.acquire("b", 40, () -> new Fake("b", log)).close();       // over budget, but a is still in use
            assertFalse(one.model().closed);
            pool.unloadAll();
            assertFalse(one.model().closed, "unloadAll skips it while one user remains");
            var unloading = Thread.ofPlatform().start(() -> pool.unload("a"));
            unloading.join(200);
            assertTrue(unloading.isAlive(), "unload waits for every user");
            one.close();
            unloading.join(5000);
            assertTrue(a.closed);
        }
    }

    @Test @org.junit.jupiter.api.Timeout(20)
    void theIdleReaperLeavesAModelWithAUserLeftAndUnloadsItOnceAllAreDone() throws Exception {
        List<String> log = java.util.Collections.synchronizedList(new ArrayList<>());
        try (var pool = new ModelPool(100, Duration.ofSeconds(1))) {
            var one = pool.acquire("a", 10, 2, () -> new Fake("a", log));
            Thread.sleep(2500);   // past the timeout, with the reaper running every second
            assertFalse(one.model().closed, "one of its two users still holds it");
            one.close();
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (pool.loadedBytes() > 0 && System.nanoTime() < deadline) Thread.sleep(50);
            assertEquals(List.of("load a", "close a"), log);
        }
    }

    @Test @org.junit.jupiter.api.Timeout(20)
    void aKeyKeepsItsConcurrency() {
        try (var pool = new ModelPool(100, null)) {
            pool.acquire("a", 10, 2, () -> new Fake("a", new ArrayList<>())).close();
            assertThrows(IllegalArgumentException.class, () -> pool.acquire("a", 10, () -> new Fake("a", new ArrayList<>())));
            assertThrows(IllegalArgumentException.class, () -> pool.acquire("b", 10, 0, () -> new Fake("b", new ArrayList<>())));
        }
    }

    @Test @org.junit.jupiter.api.Timeout(20)
    void aConcurrentModelsLoadStillReservesItsRoom() throws Exception {
        // #71's reservation, with concurrency: while a's load runs, b's load counts a's size and evicts what it must
        List<String> log = java.util.Collections.synchronizedList(new ArrayList<>());
        try (var pool = new ModelPool(100, null)) {
            pool.acquire("old", 40, () -> new Fake("old", log)).close();
            CountDownLatch loading = new CountDownLatch(1), finish = new CountDownLatch(1);
            var a = Thread.ofPlatform().start(() -> pool.acquire("a", 40, 4, () -> {
                loading.countDown();
                try { finish.await(); } catch (InterruptedException e) { throw new RuntimeException(e); }
                return new Fake("a", log);
            }).close());
            assertTrue(loading.await(5, TimeUnit.SECONDS));
            pool.acquire("b", 40, () -> new Fake("b", log)).close();   // 40 old + 40 a (reserved) + 40 b > 100: old goes
            finish.countDown();
            a.join(5000);
            assertTrue(log.contains("close old"), log.toString());
            assertEquals(80, pool.loadedBytes());
        }
    }

    @Test @org.junit.jupiter.api.Timeout(20)
    void aFailedLoadLetsTheLeaseWaitingOnItLoadAndGivesBackItsRoom() throws Exception {
        List<String> log = java.util.Collections.synchronizedList(new ArrayList<>());
        try (var pool = new ModelPool(100, null)) {
            CountDownLatch loading = new CountDownLatch(1), fail = new CountDownLatch(1);
            var first = Thread.ofPlatform().start(() -> assertThrows(IllegalStateException.class, () -> pool.acquire("a", 10, 2, () -> {
                loading.countDown();
                try { fail.await(); } catch (InterruptedException e) { throw new RuntimeException(e); }
                throw new IllegalStateException("no such file");
            })));
            assertTrue(loading.await(5, TimeUnit.SECONDS));
            Fake[] got = new Fake[1];
            var second = Thread.ofPlatform().start(() -> {
                try (var l = pool.acquire("a", 10, 2, () -> new Fake("a", log))) { got[0] = l.model(); }
            });
            second.join(200);
            assertTrue(second.isAlive(), "the second waits for the first's load");
            fail.countDown();
            first.join(5000);
            second.join(5000);
            assertNotNull(got[0], "the first's load failed, so the second loaded it");
            assertEquals(10, pool.loadedBytes());
            pool.acquire("b", 90, () -> new Fake("b", log)).close();   // 10 + 90 fit: the failed load's reservation is gone
            assertFalse(got[0].closed, log.toString());
            assertFalse(first.isAlive() || second.isAlive());
        }
    }

    static void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException e) { throw new RuntimeException(e); }
    }
}
