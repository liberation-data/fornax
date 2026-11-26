package org.liberationdata.fornax.hub;

import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.hub.HubException.Reason;
import org.liberationdata.fornax.hub.HubHttp.Cancel;
import org.liberationdata.fornax.hub.HubHttp.RemoteFile;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelSet.Part;
import org.liberationdata.fornax.hub.ModelSet.Source;
import org.liberationdata.fornax.hub.ModelStatus.PartStatus;
import org.liberationdata.fornax.hub.ModelStatus.State;
import org.liberationdata.fornax.hub.ModelStatus.Where;
import org.liberationdata.fornax.hub.PullProgress.Phase;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.DirectoryNotEmptyException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CancellationException;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Fornax's own model download and cache: one mechanism for the CLI, Fornax Studio and apps, in place of a lookup each.
 *
 * <ul>
 *   <li><b>Find</b> a {@link ModelSet}'s files: the Hugging Face cache first, then the configured
 *       {@link ModelFolders} (a ComfyUI models folder, say), so weights already on the machine are used where they
 *       are.</li>
 *   <li><b>Pull</b> what is missing into the Hugging Face cache, in the layout {@code huggingface_hub} uses, so Python
 *       tools see the files too. A pull resumes an interrupted download, checks each file's hash against the Hub's,
 *       reports progress, can be cancelled, waits for another process downloading the same file instead of
 *       downloading it twice, sends a token for gated repos, and checks free disk space before it starts.</li>
 *   <li><b>Status</b>: each part present, partly downloaded or missing, from the disk alone.</li>
 *   <li><b>Evict</b> a set's files from the cache, keeping blobs that another snapshot or set still uses.</li>
 * </ul>
 *
 * <pre>{@code
 * FornaxHub hub = FornaxHub.standard();
 * ModelSet flux = ModelCatalog.require("flux2-klein");
 * Map<String, Path> files = hub.pull(flux, p -> System.out.printf("%s %.0f%%%n", p.part().role(), p.fraction() * 100));
 * Path vae = files.get("vae");
 * }</pre>
 *
 * A FornaxHub holds no open resources and is safe to share between threads.
 */
public final class FornaxHub {
    private static final Duration LOCK_POLL = Duration.ofMillis(200);
    private static final long PROGRESS_EVERY_NANOS = 100_000_000L;

    private final CacheLayout cache;
    private final ModelFolders folders;
    private final HubHttp http;
    private final long reserveBytes;
    private final boolean offline;
    private final List<ModelSet> catalogue;
    private final RepoIndex index;

    private FornaxHub(Builder b) {
        this.cache = new CacheLayout(b.cacheRoot != null ? b.cacheRoot : HfCache.root());
        this.folders = b.folders != null ? b.folders : ModelFolders.configured();
        URI endpoint = b.endpoint != null ? b.endpoint : defaultEndpoint();
        String token = b.tokenSet ? b.token : HubToken.find().orElse(null);
        HttpClient client = b.httpClient != null ? b.httpClient : HubHttp.defaultClient();
        this.http = new HubHttp(URI.create(endpoint.toString().replaceAll("/+$", "")), token, client, b.retryDelay);
        this.reserveBytes = b.reserveBytes;
        this.offline = b.offline != null ? b.offline : envFlag("HF_HUB_OFFLINE");
        this.catalogue = b.catalogue != null ? List.copyOf(b.catalogue) : ModelCatalog.all();
        this.index = new RepoIndex(endpoint, token, offline, client);   // one client: the builder's, if given
    }

    /**
     * The usual hub: the cache {@link HfCache#root()} names, the {@link ModelFolders#configured()} folders,
     * {@code HF_ENDPOINT} (default https://huggingface.co), the token {@link HubToken#find()} finds, and
     * {@code HF_HUB_OFFLINE}.
     */
    public static FornaxHub standard() { return builder().build(); }

    public static Builder builder() { return new Builder(); }

    public Path cacheRoot() { return cache.root(); }

    public ModelFolders modelFolders() { return folders; }

    // ---- find ----

    /** The part's file: any source in the Hugging Face cache, then any in the model folders. */
    public Optional<Path> find(Part part) { return locate(part).map(Located::path); }

    /** One file: in the Hugging Face cache, then the model folders. */
    public Optional<Path> find(String repo, String file) { return find(Part.of("", repo, file)); }

    /**
     * Every part's file, by role, or an IllegalStateException naming the missing ones and how to fetch them. For an
     * app about to load the model, where a missing file is a setup error.
     */
    public Map<String, Path> require(ModelSet set) {
        Map<String, Path> out = new LinkedHashMap<>();
        List<String> missing = new ArrayList<>();
        for (Part p : set.parts()) {
            Optional<Path> f = find(p);
            if (f.isPresent()) out.put(p.role(), f.get()); else missing.add(p.role() + " (" + p.primary().repo() + " " + p.primary().file() + ")");
        }
        if (!missing.isEmpty()) throw new IllegalStateException(set.id() + " is missing " + String.join(", ", missing)
                + " in " + cache.root() + (folders.dirs().isEmpty() ? "" : " and " + folders.dirs())
                + ". Pull it with FornaxHub.pull, or `fornax pull " + set.id() + "`.");
        return out;
    }

    private record Located(Path path, Where where) {}

    private Optional<Located> locate(Part part) {
        for (Source s : part.sources()) {
            Optional<Path> p = cache.find(s);
            if (p.isPresent()) return Optional.of(new Located(p.get(), Where.HF_CACHE));
        }
        return folders.find(part).map(p -> new Located(p, Where.MODEL_FOLDER));
    }

    // ---- by reference ----

    /**
     * Whether s names a model to {@link #fetch} rather than a file: it isn't an existing path, and it is a catalogue
     * alias ({@code qwen3-4b}) or reads as a {@link ModelRef} ({@code owner/repo:Q4_K_M}, but not
     * {@code folder/model.gguf}, which is a path that isn't there).
     */
    public boolean isReference(String s) {
        if (s == null || s.isBlank()) return false;
        try { if (Files.exists(Path.of(s))) return false; } catch (java.nio.file.InvalidPathException notAPath) { /* "a:b" on Windows */ }
        return alias(s).isPresent() || ModelRef.looksLikeReference(s);
    }

    /**
     * A model by name, its files downloaded into the cache if they aren't already there, by role. The name is a
     * catalogue alias ({@code qwen3-4b}, {@code qwen3-vl-4b}: every file the set needs, such as a {@code "gguf"}
     * and its {@code "mmproj"}) or a {@link ModelRef} ({@code owner/repo[@revision][:tag]}: the file the tag picks,
     * as {@code "gguf"}, or {@code "model"} for a file that isn't a GGUF, with any further parts of a split GGUF
     * under their paths). A model already in the cache needs no network, so this works under {@code HF_HUB_OFFLINE}
     * once it has been downloaded.
     *
     * @throws IllegalArgumentException for a name that is neither (a URL says which reference to use instead), or a
     *         tag that matches several files
     * @throws HubException for a repo or tag that isn't there, and as {@link #pull} throws
     */
    public Map<String, Path> fetch(String ref, PullListener listener) throws IOException {
        Objects.requireNonNull(ref, "ref");
        Optional<ModelSet> alias = alias(ref);
        return pull(alias.isPresent() ? alias.get() : resolve(ModelRef.parse(ref)), listener);
    }

    /**
     * The set a reference names, without downloading it: from the cache when the model is there (for the default
     * revision or a commit), else from the repo's file list on the Hub, a branch or tag pinned to the commit it names.
     */
    public ModelSet resolve(ModelRef ref) throws IOException {
        boolean commit = ref.revision() != null && CacheLayout.isCommit(ref.revision());
        if (ref.revision() == null || commit) {   // a commit's snapshot never changes, so it needs no Hub either
            Map<String, Path> cached = commit ? snapshotFiles(ref.repo(), ref.revision()) : HfCache.files(cache.root(), ref.repo());
            List<String> files = ref.pick(cached.keySet(), false);
            if (!files.isEmpty() && cached.keySet().containsAll(files)) return refSet(ref, commit ? ref.revision() : "main", files);
        }
        String revision = ref.revision() == null ? "main" : ref.revision();
        RepoIndex.RepoInfo info = index.info(ref.repo(), revision);
        // a named revision is pinned to the commit it names now, so main's copy of the same file in the cache isn't taken for it
        return refSet(ref, ref.revision() == null ? revision : info.commit(),
                ref.pick(info.files().stream().map(RepoIndex.RepoFile::path).toList(), true));
    }

    /** The complete files in one snapshot of repo, by path. */
    private Map<String, Path> snapshotFiles(String repo, String commit) throws IOException {
        Path snap = cache.snapshots(repo).resolve(commit);
        Map<String, Path> out = new LinkedHashMap<>();
        if (!Files.isDirectory(snap)) return out;
        try (var w = Files.walk(snap)) {   // follows links; drops dangling ones (a download in progress)
            for (Path f : w.filter(Files::isRegularFile).sorted().toList()) out.put(snap.relativize(f).toString().replace('\\', '/'), f);
        }
        return out;
    }

    /** The set in this hub's catalogue with this id or alias. */
    Optional<ModelSet> alias(String name) { return catalogue.stream().filter(m -> m.isNamed(name.strip())).findFirst(); }

    /** One part per file: the first as "gguf" (or "model"), a split GGUF's others by path. */
    private static ModelSet refSet(ModelRef ref, String revision, List<String> files) {
        List<Part> parts = new ArrayList<>();
        for (int i = 0; i < files.size(); i++) {
            String f = files.get(i);
            String role = i > 0 ? f : GgufFiles.isGguf(f) ? "gguf" : "model";
            parts.add(new Part(role, List.of(new Source(ref.repo(), f, revision))));
        }
        return new ModelSet(ref.toString(), Kind.CHAT, ref.toString(), ref.repo(), parts, null, false, List.of());
    }

    // ---- status ----

    /** Each part present, partly downloaded or missing, from the disk alone. */
    public ModelStatus status(ModelSet set) {
        List<PartStatus> parts = new ArrayList<>();
        for (Part p : set.parts()) parts.add(status(p));
        return new ModelStatus(set, parts);
    }

    private PartStatus status(Part part) {
        Optional<Located> found = locate(part);
        if (found.isPresent()) {
            long size = sizeOf(found.get().path());
            return new PartStatus(part, State.PRESENT, found.get().path(), found.get().where(), size, size);
        }
        for (Source s : part.sources()) {
            for (CacheLayout.Pending pending : cache.pendings(s.repo())) {
                if (!pending.path().equals(s.file())) continue;
                Path incomplete = cache.incomplete(s.repo(), pending.etag());
                if (Files.isRegularFile(incomplete))   // sizeOrZero: the pull may finish, and move it, meanwhile
                    return new PartStatus(part, State.PARTIAL, incomplete, Where.HF_CACHE, sizeOrZero(incomplete), pending.size());
            }
        }
        return new PartStatus(part, State.MISSING, null, null, 0, -1);
    }

    // ---- plan and pull ----

    /**
     * What a pull would download: present parts as they are, and for each missing part its primary source's commit,
     * size and hash from the Hub (one HEAD request each), and what an interrupted pull left of it.
     */
    public PullPlan plan(ModelSet set) throws IOException { return plan(set, PullListener.NONE, new Cancel()); }

    private PullPlan plan(ModelSet set, PullListener listener, Cancel cancel) throws IOException {
        List<PullPlan.Item> items = new ArrayList<>();
        List<Part> parts = set.parts();
        for (int i = 0; i < parts.size(); i++) {
            Part p = parts.get(i);
            cancel.check();
            Optional<Path> present = find(p);
            if (present.isPresent()) {
                items.add(new PullPlan.Item(p, present.get(), null, null, null, sizeOf(present.get()), sizeOf(present.get())));
                continue;
            }
            Source s = p.primary();
            if (offline) throw new HubException(Reason.OFFLINE, s.repo(), set.id() + ": " + s.repo() + " " + s.file()
                    + " is not on disk, and HF_HUB_OFFLINE is set.");
            listener.progress(new PullProgress(set, p, i, parts.size(), Phase.CHECKING, 0, -1, 0, -1, 0));
            RemoteFile rf = http.head(s, cancel);
            Path blob = cache.blob(s.repo(), rf.etag());
            long onDisk = isComplete(blob, rf.size()) ? sizeOf(blob) : sizeOrZero(cache.incomplete(s.repo(), rf.etag()));
            items.add(new PullPlan.Item(p, null, s, rf.commit(), rf.etag(), rf.size(), onDisk));
        }
        return new PullPlan(set, items);
    }

    /**
     * Downloads the set's missing parts into the Hugging Face cache and returns every part's file, by role. Parts
     * already in the cache or a model folder are used as they are. Blocks until done; interrupting the thread
     * cancels it (with a {@link CancellationException}), and the partial file stays for the next pull to resume.
     *
     * @throws HubException for a gated repo without access, a missing file, a hash mismatch or too little disk
     * @throws IOException  when the connection fails and keeps failing after retries
     */
    public Map<String, Path> pull(ModelSet set, PullListener listener) throws IOException {
        return pull(set, listener, new Cancel());
    }

    /** {@link #pull} on a new virtual thread, with a {@link PullTask} to follow or cancel it. */
    public PullTask pullAsync(ModelSet set, PullListener listener) {
        Cancel cancel = new Cancel();
        CompletableFuture<Map<String, Path>> result = new CompletableFuture<>();
        PullTask task = new PullTask(set, result, cancel);
        Thread.ofVirtual().name("fornax-pull-" + set.id()).start(() -> {
            try { result.complete(pull(set, listener, cancel)); }
            catch (Throwable t) { result.completeExceptionally(t); }
        });
        return task;
    }

    private Map<String, Path> pull(ModelSet set, PullListener listener, Cancel cancel) throws IOException {
        PullPlan plan = plan(set, listener, cancel);
        checkDisk(plan);
        Progress progress = new Progress(set, listener, plan);
        Map<String, Path> out = new LinkedHashMap<>();
        List<PullPlan.Item> items = plan.items();
        for (int i = 0; i < items.size(); i++) {
            cancel.check();
            PullPlan.Item item = items.get(i);
            if (!item.needed()) {
                progress.event(i, Phase.PRESENT, item.size(), item.size(), true);
                out.put(item.part().role(), item.present());
                continue;
            }
            out.put(item.part().role(), fetch(item, i, progress, cancel));
        }
        return out;
    }

    /** One part: under its blob's lock, download it unless a concurrent pull already has, then link the snapshot. */
    @SuppressWarnings("try")   // the lock is held for the block, not used in it
    private Path fetch(PullPlan.Item item, int index, Progress progress, Cancel cancel) throws IOException {
        Source s = item.source();
        String repo = s.repo(), etag = item.etag();
        Path blob = cache.blob(repo, etag), part = cache.incomplete(repo, etag);
        RemoteFile rf = new RemoteFile(s, item.commit(), etag, item.size());
        try (BlobLock lock = BlobLock.acquire(cache.lock(repo, etag), cancel,
                () -> progress.event(index, Phase.WAITING, sizeOrZero(part), item.size(), false))) {
            if (!isComplete(blob, item.size())) {
                cache.writePending(repo, etag, s.file(), item.size());
                progress.started(index, sizeOrZero(part));
                try {
                    http.download(rf, part, bytes -> progress.event(index, Phase.DOWNLOADING, bytes, item.size(), false), cancel);
                } catch (HubException e) {   // a bad download is deleted: so is its note
                    if (!Files.exists(part)) Files.deleteIfExists(cache.pending(repo, etag));
                    throw e;
                }
                Files.move(part, blob, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
                Files.deleteIfExists(cache.pending(repo, etag));
            }
            Path file = cache.link(repo, item.commit(), s.file(), etag);
            if (!CacheLayout.isCommit(s.revision())) cache.writeRef(repo, s.revision(), item.commit());
            progress.event(index, Phase.DONE, item.size() >= 0 ? item.size() : sizeOf(blob), item.size(), true);
            return file;
        }
    }

    private void checkDisk(PullPlan plan) throws IOException {
        long need = plan.bytesToDownload();
        if (need == 0) return;
        Files.createDirectories(cache.root());
        var store = Files.getFileStore(cache.root());
        long free = store.getUsableSpace();
        if (free - need < reserveBytes)
            throw new HubException(Reason.DISK_SPACE, null, plan.set().id() + " needs " + bytes(need) + " more in "
                    + cache.root() + ", which has " + bytes(free) + " free" + (reserveBytes > 0 ? " (and Fornax keeps " + bytes(reserveBytes) + " free)" : "")
                    + ". Free some space, evict a model, or point HF_HUB_CACHE at a bigger disk.");
    }

    /** Builds the {@link PullProgress} events, with the set's totals and the rate. */
    private static final class Progress {
        private final ModelSet set;
        private final PullListener listener;
        private final List<PullPlan.Item> items;
        private final long[] done;
        private final long setTotal;
        private long startNanos = -1, transferred, lastEvent;

        Progress(ModelSet set, PullListener listener, PullPlan plan) {
            this.set = set;
            this.listener = listener;
            this.items = plan.items();
            this.done = new long[items.size()];
            this.setTotal = plan.bytesTotal();
            for (int i = 0; i < items.size(); i++) if (items.get(i).needed()) done[i] = items.get(i).bytesOnDisk();
        }

        /** A download of part {@code i} begins with {@code onDisk} bytes already there, which don't count toward the rate. */
        void started(int i, long onDisk) { done[i] = onDisk; }

        void event(int i, Phase phase, long bytes, long total, boolean always) {
            if (phase == Phase.DOWNLOADING) {
                if (startNanos < 0) startNanos = System.nanoTime();
                if (bytes > done[i]) transferred += bytes - done[i];   // less: the server restarted the file from 0
            }
            if (items.get(i).needed()) done[i] = bytes;
            long now = System.nanoTime();
            boolean phaseEvent = phase != Phase.DOWNLOADING && phase != Phase.WAITING;
            if (!always && !phaseEvent && now - lastEvent < PROGRESS_EVERY_NANOS) return;
            lastEvent = now;
            double rate = startNanos < 0 ? 0 : transferred / Math.max(1e-3, (now - startNanos) / 1e9);
            long setDone = 0;
            for (int k = 0; k < items.size(); k++) if (items.get(k).needed()) setDone += done[k];
            listener.progress(new PullProgress(set, items.get(i).part(), i, items.size(), phase, bytes, total, setDone, setTotal, rate));
        }
    }

    /**
     * The cross-process lock on one blob: a file lock on {@code .locks/<repo>/<etag>.lock}. Another JVM holding it, or
     * another thread of this one, means the blob is being downloaded: poll until it is free, then re-check the blob,
     * which the other pull has usually finished by then.
     *
     * <p>Threads of this JVM are kept apart by {@link #HELD} before any channel is opened, because closing a channel
     * on a file drops every lock the process holds on it (see {@link FileLock}): a thread that opened the file only to
     * find it locked would, on closing, free the downloading thread's lock for other processes.
     */
    private record BlobLock(Path key, FileChannel channel, FileLock lock) implements AutoCloseable {
        private static final Set<Path> HELD = ConcurrentHashMap.newKeySet();

        static BlobLock acquire(Path path, Cancel cancel, Runnable waiting) throws IOException {
            while (true) {
                BlobLock l = tryAcquire(path);
                if (l != null) return l;
                waiting.run();
                cancel.sleep(LOCK_POLL);
            }
        }

        /** The lock, or null when another thread or process holds it. */
        static BlobLock tryAcquire(Path path) throws IOException {
            Path key = path.toAbsolutePath().normalize();
            if (!HELD.add(key)) return null;   // another thread of this JVM has it
            FileChannel ch = null;
            try {
                Files.createDirectories(path.getParent());
                ch = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
                FileLock l = ch.tryLock();
                if (l != null) return new BlobLock(key, ch, l);
                ch.close();   // another process has it; no thread of this one does, so closing frees nothing
                HELD.remove(key);
                return null;
            } catch (IOException | RuntimeException | Error e) {
                try { if (ch != null) ch.close(); } finally { HELD.remove(key); }
                throw e;
            }
        }

        @Override public void close() throws IOException {
            try { lock.release(); } finally { try { channel.close(); } finally { HELD.remove(key); } }
        }
    }

    // ---- evict ----

    /** {@link #evict(ModelSet, Collection)}, keeping files that other sets in the catalogue on disk use. */
    public Eviction evict(ModelSet set) throws IOException { return evict(set, catalogue); }

    /**
     * Removes the set's files from the Hugging Face cache. For each part, only the source it resolves to there (the
     * one {@link #find} and {@link #status} report) is removed, not the same file published in its other sources'
     * repos, which Fornax did not pull for this set: that source's entries in every snapshot of its repo, then each blob
     * they pointed at unless another snapshot entry still points at it. Any partial download a Fornax pull left of the
     * part, from whichever source, goes too. A part
     * is kept whole when a set in {@code others} that is on disk (has a part present that this set doesn't share)
     * uses the same file: evicting ACE-Step turbo keeps the VAE that ACE-Step XL, also downloaded, needs.
     * Files in model folders are never touched. A blob another pull is downloading (its lock is held) is left.
     */
    @SuppressWarnings("try")
    public Eviction evict(ModelSet set, Collection<ModelSet> others) throws IOException {
        List<Path> removed = new ArrayList<>();
        List<Part> kept = new ArrayList<>();
        long freed = 0;
        Map<String, Set<Path>> candidates = new LinkedHashMap<>();
        for (Part part : set.parts()) {
            if (sharedWithAnother(set, part, others)) { kept.add(part); continue; }
            Optional<Source> resolved = part.sources().stream().filter(s -> cache.find(s).isPresent()).findFirst();
            if (resolved.isPresent()) {
                Source s = resolved.get();
                String repo = s.repo();
                Set<Path> blobs = candidates.computeIfAbsent(repo, r -> new LinkedHashSet<>());
                Path snaps = cache.snapshots(repo);
                if (Files.isDirectory(snaps)) {
                    try (Stream<Path> l = Files.list(snaps)) {
                        for (Path snap : l.toList()) {
                            Path entry = snap.resolve(s.file());
                            if (!Files.exists(entry, LinkOption.NOFOLLOW_LINKS)) continue;
                            Path blob = cache.blobOf(repo, entry);
                            if (blob != null) blobs.add(blob);
                            else if (Files.isRegularFile(entry, LinkOption.NOFOLLOW_LINKS)) freed += sizeOrZero(entry);
                            Files.delete(entry);
                            removed.add(entry);
                        }
                    }
                }
            }
            for (Source s : part.sources()) {   // partial downloads are Fornax's own (its .pending notes), from any source
                String repo = s.repo();
                for (CacheLayout.Pending p : cache.pendings(repo)) {
                    if (!p.path().equals(s.file())) continue;
                    Path incomplete = cache.incomplete(repo, p.etag());
                    try (BlobLock lock = BlobLock.tryAcquire(cache.lock(repo, p.etag()))) {
                        if (lock == null) continue;
                        long size = sizeOrZero(incomplete);
                        if (Files.deleteIfExists(incomplete)) { freed += size; removed.add(incomplete); }
                        Files.deleteIfExists(cache.pending(repo, p.etag()));
                        candidates.computeIfAbsent(repo, r -> new LinkedHashSet<>());   // to prune its folder
                    }
                }
            }
        }
        for (var e : candidates.entrySet()) {
            String repo = e.getKey();
            Set<Path> stillUsed = new HashSet<>();
            for (Path entry : cache.snapshotEntries(repo)) {
                Path b = cache.blobOf(repo, entry);
                if (b != null) stillUsed.add(b);
            }
            for (Path blob : e.getValue()) {
                if (stillUsed.contains(blob) || !Files.isRegularFile(blob)) continue;
                try (BlobLock lock = BlobLock.tryAcquire(cache.lock(repo, blob.getFileName().toString()))) {
                    if (lock == null) continue;
                    long size = sizeOrZero(blob);
                    if (Files.deleteIfExists(blob)) { freed += size; removed.add(blob); }
                }
            }
            prune(repo);
        }
        return new Eviction(set, freed, removed, kept);
    }

    private boolean sharedWithAnother(ModelSet set, Part part, Collection<ModelSet> others) {
        for (ModelSet o : others) {
            if (o.id().equals(set.id())) continue;
            boolean uses = o.parts().stream().anyMatch(op -> overlap(op, part));
            if (!uses) continue;
            boolean onDisk = o.parts().stream()
                    .filter(op -> set.parts().stream().noneMatch(sp -> overlap(sp, op)))
                    .anyMatch(op -> find(op).isPresent());
            if (onDisk) return true;
        }
        return false;
    }

    private static boolean overlap(Part a, Part b) {
        return a.sources().stream().anyMatch(s -> b.uses(s.repo(), s.file()));
    }

    /** Removes empty snapshot directories, and the repo's folder once it has no snapshots and no blobs left. */
    private void prune(String repo) throws IOException {
        Path snaps = cache.snapshots(repo);
        if (Files.isDirectory(snaps)) {
            try (Stream<Path> w = Files.walk(snaps)) {
                for (Path d : w.filter(p -> Files.isDirectory(p, LinkOption.NOFOLLOW_LINKS)).sorted(Comparator.reverseOrder()).toList()) {
                    if (d.equals(snaps)) continue;
                    try { Files.delete(d); } catch (DirectoryNotEmptyException ignored) {}
                }
            }
        }
        Path dir = cache.repoDir(repo);
        if (isEmptyOrMissing(snaps) && isEmptyOrMissing(dir.resolve("blobs"))) {
            deleteTree(dir);
        }
    }

    // ---- small things ----

    private static boolean isEmptyOrMissing(Path dir) throws IOException {
        if (!Files.isDirectory(dir)) return true;
        try (Stream<Path> l = Files.list(dir)) { return l.findAny().isEmpty(); }
    }

    private static void deleteTree(Path dir) throws IOException {
        if (!Files.exists(dir, LinkOption.NOFOLLOW_LINKS)) return;
        try (Stream<Path> w = Files.walk(dir)) {
            for (Path p : w.sorted(Comparator.reverseOrder()).toList()) Files.deleteIfExists(p);
        }
    }

    private static boolean isComplete(Path p, long size) {
        return Files.isRegularFile(p) && (size < 0 || sizeOrZero(p) == size);
    }

    private static long sizeOf(Path p) {
        try { return Files.size(p); } catch (IOException e) { throw new UncheckedIOException(e); }
    }

    private static long sizeOrZero(Path p) {
        try { return Files.size(p); } catch (NoSuchFileException e) { return 0; } catch (IOException e) { throw new UncheckedIOException(e); }
    }

    /** "1.4 GB", "512 MB", "12 KB", "100 B", in powers of 1000 as disks are sold. */
    static String bytes(long n) {
        if (n < 1000) return n + " B";
        String[] units = {"KB", "MB", "GB", "TB"};
        double v = n;
        int u = -1;
        while (v >= 1000 && u < units.length - 1) { v /= 1000; u++; }
        return String.format(Locale.ROOT, v < 10 ? "%.1f %s" : "%.0f %s", v, units[u]);
    }

    private static URI defaultEndpoint() {
        String ep = System.getenv("HF_ENDPOINT");
        return URI.create(ep == null || ep.isBlank() ? "https://huggingface.co" : ep.trim());
    }

    private static boolean envFlag(String name) {
        String v = System.getenv(name);
        return v != null && Set.of("1", "true", "yes", "on").contains(v.trim().toLowerCase(Locale.ROOT));
    }

    /** Settings for a {@link FornaxHub}; each defaults to what {@link #standard()} uses. */
    public static final class Builder {
        private Path cacheRoot;
        private ModelFolders folders;
        private URI endpoint;
        private String token;
        private boolean tokenSet;
        private HttpClient httpClient;
        private long reserveBytes = 1_000_000_000L;
        private Boolean offline;
        private Duration retryDelay = Duration.ofSeconds(1);
        private Collection<ModelSet> catalogue;

        private Builder() {}

        /** The Hugging Face cache to read and pull into (default {@link HfCache#root()}). */
        public Builder cacheRoot(Path root) { this.cacheRoot = root; return this; }

        /** Folders searched after the cache (default {@link ModelFolders#configured()}). */
        public Builder modelFolders(ModelFolders folders) { this.folders = folders; return this; }

        /** The Hub (default {@code HF_ENDPOINT}, else https://huggingface.co): a mirror, or a test server. */
        public Builder endpoint(URI endpoint) { this.endpoint = endpoint; return this; }

        /** The access token for gated or private repos; null for none. Default: {@link HubToken#find()}. */
        public Builder token(String token) { this.token = token == null || token.isBlank() ? null : token.trim(); this.tokenSet = true; return this; }

        /** The HTTP client; it must not follow redirects itself, since the token must not follow the Hub's to the CDN. */
        public Builder httpClient(HttpClient client) {
            if (client.followRedirects() != HttpClient.Redirect.NEVER)
                throw new IllegalArgumentException("the client must not follow redirects (HttpClient.Redirect.NEVER)");
            this.httpClient = client;
            return this;
        }

        /** Free space a pull leaves on the cache's disk; a pull that would leave less fails before it starts (default 1 GB). */
        public Builder reserveBytes(long bytes) { this.reserveBytes = bytes; return this; }

        /** No network: a pull of a missing part fails at once (default: whether {@code HF_HUB_OFFLINE} is set). */
        public Builder offline(boolean offline) { this.offline = offline; return this; }

        /** The pause before retrying a dropped connection, growing with each try in a row that adds no bytes (default 1 s). */
        public Builder retryDelay(Duration delay) { this.retryDelay = delay; return this; }

        /** The sets {@link FornaxHub#evict(ModelSet)} keeps shared files for (default {@link ModelCatalog#all()}). */
        public Builder catalogue(Collection<ModelSet> sets) { this.catalogue = sets; return this; }

        public FornaxHub build() { return new FornaxHub(this); }
    }

    @Override public String toString() {
        return "FornaxHub[" + cache.root() + (folders.dirs().isEmpty() ? "" : ", folders " + folders.dirs().stream()
                .map(Path::toString).collect(Collectors.joining(", "))) + ", " + http.endpoint() + "]";
    }
}
