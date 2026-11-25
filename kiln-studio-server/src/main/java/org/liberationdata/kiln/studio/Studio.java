package org.liberationdata.kiln.studio;

import org.liberationdata.kiln.ModelPool;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The models the studio has loaded: Kiln's {@link ModelPool} over the {@link Catalog}, each model loaded by its own
 * engine (or a stub, for the kinds named in {@code stubs}). One request uses a model at a time; others wait for it.
 */
final class Studio implements AutoCloseable {
    private final Catalog catalog;
    private final ModelPool pool;
    private final long budgetBytes;
    private final Set<Catalog.Kind> stubs;
    private final Path outputs;
    /** The ids being loaded now, for the Models panel to say so while a load takes its time. */
    private final Set<String> loading = ConcurrentHashMap.newKeySet();
    private final Map<String, Double> loadSeconds = new ConcurrentHashMap<>();
    /** Models made from the catalogue's for a request (LTX without its audio), by id: the pool may hold them too. */
    private final Map<String, Catalog.Model> variants = new ConcurrentHashMap<>();

    Studio(Catalog catalog, long budgetBytes, Duration idleTimeout, Set<Catalog.Kind> stubs, Path outputs) throws IOException {
        this.catalog = catalog;
        this.budgetBytes = budgetBytes;
        this.pool = new ModelPool(budgetBytes, idleTimeout);
        this.stubs = Set.copyOf(stubs);
        this.outputs = Files.createDirectories(outputs);
    }

    Catalog catalog() { return catalog; }

    Path outputs() { return outputs; }

    Set<Catalog.Kind> stubs() { return stubs; }

    boolean stubbed(Catalog.Kind kind) { return stubs.contains(kind); }

    /** The model, found in the catalogue, ready to load; fails with why it can't be. */
    Catalog.Model model(String id, Catalog.Kind kind) {
        Catalog.Model m = catalog.find(id).orElseThrow(() -> new IllegalArgumentException("no model " + id + " in the cache"));
        if (m.kind() != kind) throw new IllegalArgumentException(id + " is a " + m.kind() + " model, not a " + kind + " model");
        if (!m.available()) throw new IllegalArgumentException(id + " is missing files: " + m.parts().stream()
                .filter(p -> !p.present()).map(p -> p.repo() + " " + p.file()).toList());
        return m;
    }

    /**
     * The model without its audio VAE, for a video asked for without sound: audio_vae_path is set when the model loads,
     * so this is a model of its own in the pool, beside the one with audio.
     */
    Catalog.Model silent(Catalog.Model m) {
        if (m.path("audio_vae") == null) return m;
        return variants.computeIfAbsent(m.id() + "+silent", id -> m.without("audio_vae", "+silent", " (no audio)"));
    }

    /** Whether using the model now would load it first (so a request can say it is loading). */
    boolean isLoaded(String id) {
        return pool.entries().stream().anyMatch(e -> e.key().equals(id) && e.loaded());
    }

    /** Borrows the model, loading it first if it isn't loaded. Close the lease to give it back; it stays loaded. */
    <M extends AutoCloseable> ModelPool.Lease<M> lease(Catalog.Model m, Class<M> type) {
        // A stub loads nothing, so it holds nothing of the budget.
        return pool.acquire(m.id(), stubbed(m.kind()) ? 0 : m.sizeBytes(), () -> type.cast(load(m)));
    }

    /** Seconds the model's last load took, or null if this run hasn't loaded it. */
    Double loadSeconds(String id) { return loadSeconds.get(id); }

    boolean unload(String id) { return pool.unload(id); }

    void unloadAll() { pool.unloadAll(); }

    private AutoCloseable load(Catalog.Model m) {
        loading.add(m.id());
        long t0 = System.nanoTime();
        try {
            AutoCloseable model = switch (m.kind()) {
                case chat -> stubbed(m.kind()) ? Engines.stubChat()
                        : Engines.llama(m, (int) Json.num(m.defaults(), "contextSize", 8192));
                case image -> stubbed(m.kind()) ? Engines.stubPainter() : Engines.diffusion(m);
                case song -> stubbed(m.kind()) ? Engines.stubSinger()
                        : m.engine().equals("yue2.cpp") ? Engines.yue2(m) : Engines.aceStep(m);
                case video -> stubbed(m.kind()) ? Engines.stubFilmmaker(m.path("audio_vae") != null) : Engines.ltx(m);
            };
            loadSeconds.put(m.id(), (System.nanoTime() - t0) / 1e9);
            return model;
        } finally {
            loading.remove(m.id());
        }
    }

    /** The Models panel's view: every model in the catalogue, what the pool holds of it, and the memory in use. */
    Map<String, Object> status() {
        Map<String, ModelPool.Entry> entries = new java.util.HashMap<>();
        for (var e : pool.entries()) entries.put(e.key(), e);
        List<Object> models = new ArrayList<>();
        List<Catalog.Model> all = new ArrayList<>(catalog.models());
        // A variant is listed while the pool holds it, so it can be seen, and unloaded, as any other model.
        for (var v : variants.values()) if (entries.containsKey(v.id()) && entries.get(v.id()).loaded()) all.add(v);
        for (Catalog.Model m : all) {
            var e = entries.get(m.id());
            boolean loaded = e != null && e.loaded();
            List<Object> parts = new ArrayList<>();
            for (var p : m.parts()) parts.add(Json.obj("role", p.role(), "repo", p.repo(), "file", p.file(), "present", p.present()));
            models.add(Json.obj(
                    "id", m.id(), "kind", m.kind().name(), "name", m.name(), "engine", m.engine(),
                    "available", m.available(), "sizeBytes", m.sizeBytes(), "parts", parts, "defaults", m.defaults(),
                    "loaded", loaded, "inUse", e != null && e.inUse() && loaded, "loading", loading.contains(m.id()),
                    "lastUsed", e == null ? null : e.lastUsed().toString(), "loadSeconds", loadSeconds.get(m.id()),
                    "stub", stubbed(m.kind()), "license", m.license()));
        }
        return Json.obj("cacheRoot", catalog.root().toString(), "budgetBytes", budgetBytes, "loadedBytes", pool.loadedBytes(),
                "gpu", Gpu.memory(), "stubs", stubs.stream().map(Enum::name).sorted().toList(), "models", models);
    }

    @Override public void close() { pool.close(); }
}
