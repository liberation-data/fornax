package org.liberationdata.fornax.studio;

import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.HubException;
import org.liberationdata.fornax.hub.ModelCatalog;
import org.liberationdata.fornax.hub.ModelFolders;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.hub.ModelStatus;
import org.liberationdata.fornax.hub.PullPlan;
import org.liberationdata.fornax.hub.PullProgress;
import org.liberationdata.fornax.hub.PullTask;
import org.liberationdata.fornax.hub.RepoIndex;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Function;

/**
 * Getting models from inside the studio: Fornax's catalogue ({@link ModelCatalog}), what of each set is on disk, and
 * downloads through {@link FornaxHub#pullAsync} into the cache the studio reads, so the CLI, Studio and apps share one
 * download path and one cache. The hub checks free disk before downloading anything, resumes an interrupted download
 * and verifies each file's hash.
 *
 * <p>A download belongs to the studio, not to the page that started it: closing the Models window doesn't stop a
 * 20 GB download. Asking for a set that is already downloading watches that download; {@link #cancel} stops it.
 */
final class Pulls {
    private final List<ModelSet> sets;
    /** A hub for a token (null: the hub's own, from HF_TOKEN or the token file). */
    private final Function<String, FornaxHub> hubs;
    private final URI endpoint;
    private final Map<String, Pull> active = new ConcurrentHashMap<>();
    /** The token the desktop app gave from its keychain ({@link #token(String)}); in memory only. */
    private volatile String token;

    /** One download, and the streams watching it. */
    private static final class Pull {
        final List<Sse> watchers = new CopyOnWriteArrayList<>();
        PullTask task;   // set under the lock on active, before the pull is put there
        volatile PullProgress last;
    }

    Pulls(List<ModelSet> sets, Function<String, FornaxHub> hubs, URI endpoint) {
        this.sets = List.copyOf(sets);
        this.hubs = hubs;
        this.endpoint = URI.create(endpoint.toString().replaceAll("/+$", ""));
    }

    /** Fornax's catalogue, into the catalog's cache and model folders, from HF_ENDPOINT (or huggingface.co). */
    static Pulls standard(Catalog catalog) {
        URI endpoint = RepoIndex.endpoint();
        return new Pulls(ModelCatalog.all(), token -> {
            var b = FornaxHub.builder().cacheRoot(catalog.root()).modelFolders(new ModelFolders(catalog.folders())).endpoint(endpoint);
            if (token != null && !token.isBlank()) b.token(token);
            return b.build();
        }, endpoint);
    }

    /**
     * The Hugging Face token for gated repos, from the desktop app's keychain, for every request that brings none of
     * its own; null or blank clears it, and the hub's own (HF_TOKEN, or the token file) is used again. Held in memory
     * only: never written, logged or sent anywhere but the Hub. A token no header can carry is refused, without its
     * characters, before it is kept.
     */
    void token(String t) { token = checked(t); }

    /** Whether the app has given a token ({@link #token(String)}). */
    boolean hasToken() { return token != null; }

    /** A request's own token, else the app's. */
    private String tokenFor(String requested) {
        String t = checked(requested);
        return t != null ? t : token;
    }

    /**
     * The token stripped, or null for none. One with a space or control character inside is refused here: the HTTP
     * client would refuse it too, but with the header's value, token and all, in its message, which is sent back.
     */
    static String checked(String t) {
        if (t == null || t.isBlank()) return null;
        String s = t.strip();
        if (!s.chars().allMatch(c -> c > 0x20 && c < 0x7f))
            throw new IllegalArgumentException("a Hugging Face token is printable ASCII, without spaces");
        return s;
    }

    /** The set with this id or alias ("flux2-klein" or the studio's "image:flux2-klein"). */
    ModelSet set(String name) {
        if (name == null || name.isBlank()) throw new IllegalArgumentException("id is required");
        return sets.stream().filter(s -> s.isNamed(name.strip())).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("no model set named " + name));
    }

    /** Every set: what of it is on disk (no network), its licence, whether its repo is gated, and whether it is downloading. */
    List<Object> catalogue() {
        FornaxHub hub = hubs.apply(null);
        List<Object> out = new ArrayList<>();
        for (ModelSet s : sets) {
            ModelStatus st = hub.status(s);
            List<Object> parts = new ArrayList<>();
            for (var p : st.parts())
                parts.add(Json.obj("role", p.part().role(), "repo", p.part().primary().repo(), "file", p.part().primary().file(),
                        "state", p.state().name(), "bytesOnDisk", p.bytesOnDisk(), "size", p.size()));
            Pull pull = active.get(s.id());
            out.add(Json.obj("id", s.id(), "aliases", s.aliases(), "kind", s.kind().name().toLowerCase(java.util.Locale.ROOT),
                    "title", s.title(), "engine", s.engine(), "license", s.license(), "gated", s.gated(),
                    "termsUrl", s.gated() ? termsUrl(s.parts().getFirst().primary().repo()) : null,
                    "state", st.state().name(), "bytesOnDisk", st.bytesOnDisk(), "parts", parts,
                    "downloading", pull != null, "progress", pull == null || pull.last == null ? null : progress(pull.last)));
        }
        return out;
    }

    /**
     * What a download would take, from the Hub (one HEAD request per missing file), and the free disk it would go to;
     * nothing is downloaded. The token is for a gated repo.
     */
    Map<String, Object> plan(String name, String token) throws IOException {
        ModelSet s = set(name);
        FornaxHub hub = hubs.apply(tokenFor(token));
        PullPlan plan = hub.plan(s);
        Path cacheRoot = hub.cacheRoot();
        Files.createDirectories(cacheRoot);
        return Json.obj("id", s.id(), "complete", plan.complete(), "bytesToDownload", plan.bytesToDownload(),
                "bytesTotal", plan.bytesTotal(), "freeBytes", Files.getFileStore(cacheRoot).getUsableSpace(),
                "license", s.license(), "gated", s.gated());
    }

    /**
     * Downloads the set, or watches its download if one is under way, sending {@code start}, {@code progress} (about ten a
     * second while bytes arrive), and then {@code done} (its files, by role), {@code cancelled} or {@code error}
     * ({@code reason}: a {@link HubException.Reason}, such as {@code DISK_SPACE} or {@code ACCESS_DENIED} with the
     * {@code termsUrl} to accept). Returns when the download ends; a page that leaves stops watching, not downloading.
     */
    void pull(String name, String token, Sse sse) {
        ModelSet s = set(name);
        Pull pull;
        boolean attached;
        synchronized (active) {   // one download per set: a second request watches the first's
            pull = active.get(s.id());
            attached = pull != null;
            if (!attached) {
                Pull p = pull = new Pull();
                p.task = hubs.apply(tokenFor(token)).pullAsync(s, progress -> {
                    p.last = progress;
                    Object data = progress(progress);
                    for (Sse w : p.watchers) if (!w.send("progress", data)) p.watchers.remove(w);
                });
                active.put(s.id(), p);
                p.task.result().whenComplete((files, error) -> active.remove(s.id(), p));
            }
        }
        // start first, then the latest progress, then each as it comes (progress missed meanwhile is soon superseded)
        sse.send("start", Json.obj("id", s.id(), "title", s.title(), "license", s.license(), "gated", s.gated(), "attached", attached));
        PullProgress last = pull.last;
        if (last != null) sse.send("progress", progress(last));
        pull.watchers.add(sse);
        try {
            Map<String, Path> files = pull.task.result().join();
            Map<String, Object> byRole = new LinkedHashMap<>();
            files.forEach((role, path) -> byRole.put(role, path.toString()));
            sse.send("done", Json.obj("id", s.id(), "files", byRole));
        } catch (CancellationException e) {
            sse.send("cancelled", Json.obj("id", s.id()));
        } catch (CompletionException e) {
            Throwable t = e.getCause();
            if (t instanceof CancellationException) { sse.send("cancelled", Json.obj("id", s.id())); return; }
            sse.send("error", failure(s.id(), t));
        } finally {
            pull.watchers.remove(sse);
        }
    }

    /** Stops the set's download, if one is under way; what it has downloaded stays, for the next pull to resume. */
    boolean cancel(String name) {
        Pull p = active.get(set(name).id());
        if (p == null) return false;
        p.task.cancel();
        return true;
    }

    /** Stops every download under way, as the server stops; what each has downloaded stays. */
    void cancelAll() {
        for (Pull p : active.values()) p.task.cancel();
    }

    /**
     * A failure, in words: its {@code message}, {@code reason} (a {@link HubException.Reason}, or {@code IO}) and, for
     * {@code ACCESS_DENIED}, the {@code termsUrl} to accept.
     */
    Map<String, Object> failure(String id, Throwable t) {
        var data = Json.obj("id", id, "message", t.getMessage() != null ? t.getMessage() : t.toString(), "reason",
                t instanceof HubException h ? h.reason().name() : "IO");
        if (t instanceof HubException h && h.reason() == HubException.Reason.ACCESS_DENIED && h.repo() != null)
            data.put("termsUrl", termsUrl(h.repo()));
        return data;
    }

    private String termsUrl(String repo) { return endpoint + "/" + repo; }

    private static Map<String, Object> progress(PullProgress p) {
        return Json.obj("id", p.set().id(), "role", p.part().role(), "file", p.part().primary().file(),
                "index", (long) p.index(), "count", (long) p.count(), "phase", p.phase().name(),
                "bytesDone", p.bytesDone(), "bytesTotal", p.bytesTotal(),
                "setBytesDone", p.setBytesDone(), "setBytesTotal", p.setBytesTotal(), "fraction", p.fraction(),
                "bytesPerSecond", p.bytesPerSecond(), "etaSeconds", p.eta().map(d -> d.toMillis() / 1000.0).orElse(null));
    }
}
