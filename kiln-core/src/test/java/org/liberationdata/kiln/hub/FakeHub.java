package org.liberationdata.kiln.hub;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A small Hugging Face Hub on localhost that answers as the real one does where Kiln relies on it: a HEAD on
 * {@code /<repo>/resolve/<rev>/<path>} with {@code X-Repo-Commit}, {@code X-Linked-Etag} and {@code X-Linked-Size} on a
 * redirect (302 to the CDN for LFS files, 307 for git files), 307 for a renamed repo, 401 with {@code X-Error-Code} for
 * a gated repo without the token, and a CDN on another port that honours {@code Range}. It can corrupt a file, drop a
 * transfer part way, and slow transfers down, and it records what it was asked for.
 */
final class FakeHub implements AutoCloseable {
    record File(String repo, String path, byte[] content, boolean lfs) {
        String etag() { return lfs ? hex("SHA-256", content) : hex("SHA-1", gitBlob(content)); }
    }

    private final HttpServer hub, cdn;
    private final Map<String, File> files = new ConcurrentHashMap<>();
    private final Map<String, File> byEtag = new ConcurrentHashMap<>();
    private final Map<String, String> renamed = new ConcurrentHashMap<>();
    private final Set<String> gated = ConcurrentHashMap.newKeySet();
    private final Set<String> corrupt = ConcurrentHashMap.newKeySet();
    final String token = "hf_test_token";

    /** Every Range header the CDN saw, as "etag bytes=N-". */
    final List<String> ranges = new CopyOnWriteArrayList<>();
    /** GETs of file contents, by etag. */
    final Map<String, AtomicInteger> gets = new ConcurrentHashMap<>();
    final AtomicInteger heads = new AtomicInteger();
    final AtomicLong bytesServed = new AtomicLong();
    /** Authorization headers that reached the CDN (there should be none). */
    final List<String> cdnAuth = new CopyOnWriteArrayList<>();
    /** The next transfer stops after this many bytes (once); 0 for none. */
    final AtomicLong dropNextAfter = new AtomicLong();
    /** Every CDN response ends after this many bytes (0: never), so the file never arrives. */
    volatile long dropEveryAfter;
    /** The CDN never sends past this byte of a file (0: no limit): a transfer that stops getting anywhere. */
    volatile long stallAt;
    /** A pause after each 16 KB sent, to make transfers slow enough to overlap or cancel. */
    volatile long pauseMillis;
    /** The CDN answers a Range request with the whole file (200), as a server without range support does. */
    volatile boolean ignoreRange;
    /** Sent as X-Repo-Commit instead of the real commit, as a broken or hostile mirror might; null for the real one. */
    volatile String commitHeader;
    /**
     * The Hub answers for a git file itself, 200 with {@code ETag} and {@code Content-Length} and no {@code X-Linked-*},
     * as huggingface.co does for small files, instead of redirecting to the CDN.
     */
    volatile boolean gitFilesDirect;
    /** The CDN sends bodies chunked, without a Content-Length. */
    volatile boolean chunked;

    FakeHub() throws IOException {
        InetSocketAddress any = new InetSocketAddress(InetAddress.getLoopbackAddress(), 0);
        hub = HttpServer.create(any, 0);
        cdn = HttpServer.create(any, 0);
        hub.createContext("/", this::hub);
        cdn.createContext("/", this::cdn);
        hub.setExecutor(Executors.newCachedThreadPool());
        cdn.setExecutor(Executors.newCachedThreadPool());
        hub.start();
        cdn.start();
    }

    URI endpoint() { return URI.create("http://127.0.0.1:" + hub.getAddress().getPort()); }

    File add(String repo, String path, int size, boolean lfs) {
        byte[] b = new byte[size];
        new Random(repo.hashCode() * 31L + path.hashCode()).nextBytes(b);
        return add(repo, path, b, lfs);
    }

    File add(String repo, String path, byte[] content, boolean lfs) {
        File f = new File(repo, path, content, lfs);
        files.put(repo + "/" + path, f);
        byEtag.put(f.etag(), f);
        return f;
    }

    void gate(String repo) { gated.add(repo); }

    void rename(String from, String to) { renamed.put(from, to); }

    void corrupt(File f) { corrupt.add(f.etag()); }

    static String commit(String repo) { return hex("SHA-1", ("commit of " + repo).getBytes(StandardCharsets.UTF_8)); }

    int gets(File f) { AtomicInteger n = gets.get(f.etag()); return n == null ? 0 : n.get(); }

    // /<org>/<name>/resolve/<rev>/<path>
    private void hub(HttpExchange x) throws IOException {
        try (x) {
            String[] seg = x.getRequestURI().getPath().split("/", 6);
            if (seg.length < 6 || !seg[3].equals("resolve")) { send(x, 404, null); return; }
            String repo = seg[1] + "/" + seg[2], rev = seg[4], path = seg[5];
            if (x.getRequestMethod().equals("HEAD")) heads.incrementAndGet();
            if (renamed.containsKey(repo)) {
                x.getResponseHeaders().set("Location", "/" + renamed.get(repo) + "/resolve/" + rev + "/" + path);
                send(x, 307, null);
                return;
            }
            boolean known = files.keySet().stream().anyMatch(k -> k.startsWith(repo + "/"));
            if (!known) { x.getResponseHeaders().set("X-Error-Code", "RepoNotFound"); send(x, 401, "Repository not found"); return; }
            if (gated.contains(repo) && !("Bearer " + token).equals(x.getRequestHeaders().getFirst("Authorization"))) {
                x.getResponseHeaders().set("X-Error-Code", "GatedRepo");
                x.getResponseHeaders().set("X-Error-Message", "Access to model " + repo + " is restricted.");
                send(x, 401, "gated");
                return;
            }
            if (!rev.equals("main") && !rev.equals(commit(repo))) { x.getResponseHeaders().set("X-Error-Code", "RevisionNotFound"); send(x, 404, null); return; }
            File f = files.get(repo + "/" + path);
            if (f == null) { x.getResponseHeaders().set("X-Error-Code", "EntryNotFound"); send(x, 404, null); return; }
            var h = x.getResponseHeaders();
            h.set("X-Repo-Commit", commitHeader != null ? commitHeader : commit(repo));
            if (gitFilesDirect && !f.lfs()) {
                h.set("ETag", "\"" + f.etag() + "\"");
                if (x.getRequestMethod().equals("HEAD")) {
                    h.set("Content-Length", String.valueOf(f.content().length));
                    x.sendResponseHeaders(200, -1);
                } else {
                    gets.computeIfAbsent(f.etag(), k -> new AtomicInteger()).incrementAndGet();
                    x.sendResponseHeaders(200, f.content().length);
                    x.getResponseBody().write(f.content());
                }
                return;
            }
            h.set("X-Linked-Etag", "\"" + f.etag() + "\"");
            h.set("X-Linked-Size", String.valueOf(f.content().length));
            h.set("Location", "http://127.0.0.1:" + cdn.getAddress().getPort() + "/" + (f.lfs() ? "cdn" : "raw") + "/" + f.etag());
            send(x, f.lfs() ? 302 : 307, null);
        }
    }

    private void cdn(HttpExchange x) throws IOException {
        try (x) {
            String auth = x.getRequestHeaders().getFirst("Authorization");
            if (auth != null) cdnAuth.add(auth);
            String etag = x.getRequestURI().getPath().replaceFirst("^/(cdn|raw)/", "");
            File f = byEtag.get(etag);
            if (f == null) { send(x, 404, null); return; }
            gets.computeIfAbsent(etag, k -> new AtomicInteger()).incrementAndGet();
            byte[] content = f.content().clone();
            if (corrupt.contains(etag)) content[content.length / 2] ^= 0x5a;
            long from = 0;
            String range = x.getRequestHeaders().getFirst("Range");
            if (range != null) ranges.add(etag + " " + range);
            if (range != null && !ignoreRange) {
                from = Long.parseLong(range.replaceAll("bytes=(\\d+)-.*", "$1"));
                if (from >= content.length) { send(x, 416, null); return; }
                x.getResponseHeaders().set("Content-Range", "bytes " + from + "-" + (content.length - 1) + "/" + content.length);
            }
            x.getResponseHeaders().set("Accept-Ranges", "bytes");
            long length = content.length - from;
            x.sendResponseHeaders(range != null && !ignoreRange ? 206 : 200, chunked ? 0 : length);
            long drop = dropNextAfter.getAndSet(0);
            if (drop == 0) drop = dropEveryAfter;
            OutputStream out = x.getResponseBody();
            int pos = (int) from;
            long sent = 0;
            if (stallAt > 0) drop = drop > 0 ? Math.min(drop, Math.max(0, stallAt - from)) : Math.max(0, stallAt - from);
            if (stallAt > 0 && drop == 0) throw new IOException("stalled on purpose");
            while (pos < content.length) {
                int n = Math.min(16 * 1024, content.length - pos);
                if (drop > 0 && sent + n > drop) n = (int) (drop - sent);
                if (n > 0) {
                    out.write(content, pos, n);
                    out.flush();
                    pos += n;
                    sent += n;
                    bytesServed.addAndGet(n);
                }
                if (drop > 0 && sent >= drop) throw new IOException("dropped on purpose");   // closes the connection short
                if (pauseMillis > 0) try { Thread.sleep(pauseMillis); } catch (InterruptedException e) { return; }
            }
        } catch (IOException e) {
            // the client went away (a cancel), or the drop above: either way the exchange is over
        }
    }

    private static void send(HttpExchange x, int code, String body) throws IOException {
        if (body == null || x.getRequestMethod().equals("HEAD")) { x.sendResponseHeaders(code, -1); return; }
        byte[] b = body.getBytes(StandardCharsets.UTF_8);
        x.sendResponseHeaders(code, b.length);
        x.getResponseBody().write(b);
    }

    static byte[] gitBlob(byte[] content) {
        byte[] head = ("blob " + content.length + "\0").getBytes(StandardCharsets.US_ASCII);
        byte[] all = new byte[head.length + content.length];
        System.arraycopy(head, 0, all, 0, head.length);
        System.arraycopy(content, 0, all, head.length, content.length);
        return all;
    }

    static String hex(String algorithm, byte[] b) {
        try { return HexFormat.of().formatHex(MessageDigest.getInstance(algorithm).digest(b)); }
        catch (Exception e) { throw new IllegalStateException(e); }
    }

    @Override public void close() {
        hub.stop(0);
        cdn.stop(0);
    }
}
