package org.liberationdata.kiln.hub;

import org.liberationdata.kiln.hub.HubException.Reason;
import org.liberationdata.kiln.hub.ModelSet.Source;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpHeaders;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.HexFormat;
import java.util.Locale;
import java.util.Optional;
import java.util.concurrent.CancellationException;
import java.util.function.LongConsumer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The Hugging Face Hub's file endpoints, as {@code huggingface_hub} uses them: a HEAD on
 * {@code /<repo>/resolve/<revision>/<path>} for a file's commit, etag and size, and a GET (with {@code Range} to resume)
 * that redirects to the CDN. The token goes to the Hub's own scheme, host and port only, never to the CDN it redirects to.
 */
final class HubHttp {
    /**
     * A file as the Hub describes it. {@code etag} is its blob name in the cache: the sha256 for LFS files
     * ({@code X-Linked-Etag}), the git blob id otherwise ({@code ETag}). {@code size} is -1 when the Hub didn't say.
     */
    record RemoteFile(Source source, String commit, String etag, long size) {}

    private static final Pattern CONTENT_RANGE = Pattern.compile("bytes (\\d+)-\\d+/(\\d+|\\*)");
    private static final int ATTEMPTS = 5;

    private final URI endpoint;
    private final String token;
    private final HttpClient http;
    private final Duration retryDelay;

    HubHttp(URI endpoint, String token, HttpClient http, Duration retryDelay) {
        this.endpoint = endpoint;
        this.token = token;
        this.http = http;
        this.retryDelay = retryDelay;
    }

    static HttpClient defaultClient() {
        return HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NEVER).connectTimeout(Duration.ofSeconds(30)).build();
    }

    URI endpoint() { return endpoint; }

    boolean hasToken() { return token != null; }

    /** The file's commit, etag and size, following the Hub's own redirects (a renamed repo) but not the CDN's. */
    RemoteFile head(Source s, Cancel cancel) throws IOException {
        URI uri = URI.create(endpoint + "/" + s.repo() + "/resolve/" + encode(s.revision()) + "/" + encodePath(s.file()));
        for (int hops = 0; hops < 10; hops++) {
            cancel.check();
            HttpResponse<Void> r = send(request(uri).method("HEAD", HttpRequest.BodyPublishers.noBody()).build(),
                    HttpResponse.BodyHandlers.discarding(), cancel);
            int code = r.statusCode();
            HttpHeaders h = r.headers();
            Optional<String> commit = h.firstValue("X-Repo-Commit");
            if (code >= 200 && code < 300 || code >= 300 && code < 400 && commit.isPresent()) {
                String etag = h.firstValue("X-Linked-Etag").or(() -> h.firstValue("ETag")).map(HubHttp::unquote).orElse(null);
                long size = h.firstValue("X-Linked-Size").or(() -> code < 300 ? h.firstValue("Content-Length") : Optional.empty())
                        .map(Long::parseLong).orElse(-1L);
                // both name paths in the cache (snapshots/<commit>, blobs/<etag>), so neither may be "..", say
                if (commit.isEmpty() || !CacheLayout.isCommit(commit.get()) || etag == null || !etag.matches("\\w[\\w.-]*"))
                    throw new HubException(Reason.PROTOCOL, s.repo(), s.repo() + " " + s.file() + ": the Hub didn't give its commit and etag");
                return new RemoteFile(s, commit.get(), etag, size);
            }
            if (code >= 300 && code < 400) {
                uri = uri.resolve(h.firstValue("Location").orElseThrow(() ->
                        new HubException(Reason.PROTOCOL, s.repo(), s.repo() + ": a redirect without a Location")));
                continue;
            }
            throw failure(code, s.repo(), s.file(), h);
        }
        throw new HubException(Reason.PROTOCOL, s.repo(), s.repo() + " " + s.file() + ": too many redirects");
    }

    /**
     * Downloads {@code f} into {@code part}, resuming from what is already there, retrying a dropped connection, and
     * checking its size and hash: sha256 for an LFS etag, git's blob sha1 for a git etag. A mismatch deletes
     * {@code part}. {@code progress} receives the file's bytes on disk so far.
     */
    void download(RemoteFile f, Path part, LongConsumer progress, Cancel cancel) throws IOException {
        URI uri = URI.create(endpoint + "/" + f.source().repo() + "/resolve/" + f.commit() + "/" + encodePath(f.source().file()));
        Digest digest = Digest.of(f);
        IOException last = null;
        for (int attempt = 1; attempt <= ATTEMPTS; attempt++) {
            try {
                fetch(uri, f, part, digest, progress, cancel);
                last = null;
                break;
            } catch (HubException e) {
                throw e;
            } catch (IOException e) {
                cancel.check();   // a cancel closes the stream, which lands here
                last = e;         // a dropped connection: resume from what we have
                cancel.sleep(retryDelay.multipliedBy(attempt));
            }
        }
        if (last != null) throw last;
        long have = Files.size(part);
        if (f.size() >= 0 && have != f.size()) {
            Files.deleteIfExists(part);
            throw new HubException(Reason.PROTOCOL, f.source().repo(), f.source().file() + ": got " + have + " bytes, expected " + f.size());
        }
        if (digest != null && !digest.matches(f.etag())) {
            Files.deleteIfExists(part);
            throw new HubException(Reason.CHECKSUM_MISMATCH, f.source().repo(), f.source().repo() + " " + f.source().file()
                    + ": the download's " + digest.name + " is " + digest.hex() + ", but the Hub says " + f.etag()
                    + ". The partial file was deleted; pull again.");
        }
    }

    private void fetch(URI uri, RemoteFile f, Path part, Digest digest, LongConsumer progress, Cancel cancel) throws IOException {
        long have = Files.exists(part) ? Files.size(part) : 0;
        if (f.size() >= 0 && have > f.size()) { Files.delete(part); have = 0; }
        if (digest != null) { digest.reset(); if (have > 0) digest.hash(part, cancel); }
        progress.accept(have);
        if (f.size() >= 0 && have == f.size()) {
            if (have == 0) { Files.createDirectories(part.getParent()); Files.write(part, new byte[0]); }   // an empty file
            return;
        }

        HttpResponse<InputStream> r = null;
        for (int hops = 0; hops < 10; hops++) {
            HttpRequest.Builder req = request(uri);
            if (have > 0) req.header("Range", "bytes=" + have + "-");
            r = send(req.build(), HttpResponse.BodyHandlers.ofInputStream(), cancel);
            int code = r.statusCode();
            if (code < 300 || code >= 400) break;
            String loc = r.headers().firstValue("Location").orElse(null);
            r.body().close();
            if (loc == null) throw new HubException(Reason.PROTOCOL, f.source().repo(), "a redirect without a Location");
            uri = uri.resolve(loc);
        }
        int code = r.statusCode();
        if (code == 416 && f.size() >= 0 && have == f.size()) { r.body().close(); return; }
        if (code != 200 && code != 206) {
            r.body().close();
            throw failure(code, f.source().repo(), f.source().file(), r.headers());
        }
        boolean append = code == 206;
        if (append) {
            Matcher m = CONTENT_RANGE.matcher(r.headers().firstValue("Content-Range").orElse(""));
            if (!m.matches() || Long.parseLong(m.group(1)) != have) {   // not the bytes asked for: start over
                r.body().close();
                Files.deleteIfExists(part);
                throw new IOException("the server resumed at the wrong offset");
            }
        } else if (have > 0) {   // the server sent the whole file: start over
            have = 0;
            if (digest != null) digest.reset();
            progress.accept(0);
        }
        Files.createDirectories(part.getParent());
        try (InputStream in = cancel.track(r.body());
             OutputStream out = Files.newOutputStream(part, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                     append ? StandardOpenOption.APPEND : StandardOpenOption.TRUNCATE_EXISTING)) {
            byte[] buf = new byte[1 << 20];
            for (int n; (n = in.read(buf)) > 0; ) {
                cancel.check();
                out.write(buf, 0, n);
                if (digest != null) digest.update(buf, n);
                have += n;
                progress.accept(have);
            }
        } finally {
            cancel.untrack();
        }
        // a body that ends cleanly but short (no Content-Length to hold it to): resume it, as for a dropped connection
        if (f.size() >= 0 && have < f.size()) throw new IOException("the transfer ended at " + have + " of " + f.size() + " bytes");
    }

    private HttpRequest.Builder request(URI uri) {
        HttpRequest.Builder b = HttpRequest.newBuilder(uri).header("User-Agent", userAgent());
        if (token != null && sendsToken(uri, endpoint)) b.header("Authorization", "Bearer " + token);
        return b;
    }

    /**
     * Whether a request to {@code uri} may carry the token: only to the endpoint's own scheme, host and port, so a
     * redirect to the CDN, or from https to http on the same host, never sends it elsewhere or in the clear.
     */
    static boolean sendsToken(URI uri, URI endpoint) {
        return uri.getHost() != null && uri.getHost().equalsIgnoreCase(endpoint.getHost()) && uri.getPort() == endpoint.getPort()
                && uri.getScheme() != null && uri.getScheme().equalsIgnoreCase(endpoint.getScheme());
    }

    private <T> HttpResponse<T> send(HttpRequest req, HttpResponse.BodyHandler<T> handler, Cancel cancel) throws IOException {
        try { return http.send(req, handler); }
        catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            cancel.check();
            throw new IOException("interrupted", e);
        }
    }

    private HubException failure(int code, String repo, String path, HttpHeaders h) {
        String what = repo + " " + path;
        String errorCode = h.firstValue("X-Error-Code").orElse("");
        String page = endpoint + "/" + repo;
        if ((code == 401 || code == 404) && errorCode.equals("RepoNotFound"))
            return new HubException(Reason.NOT_FOUND, repo, repo + ": no such repo on " + endpoint
                    + (token == null ? ", or it is private (set HF_TOKEN to a token that can read it)." : " that your token can read."));
        return switch (code) {
            case 401, 403 -> new HubException(Reason.ACCESS_DENIED, repo, what + ": access denied (HTTP " + code + "). "
                    + (token == null
                       ? repo + " is gated or private. Accept its terms on " + page + ", then set HF_TOKEN to a token from "
                         + endpoint + "/settings/tokens (or run `hf auth login`)."
                       : "A token was sent. Check that you accepted " + repo + "'s terms on " + page + " and that the token can read it."));
            case 404 -> new HubException(Reason.NOT_FOUND, repo, what + ": not found on " + endpoint
                    + " (" + (errorCode.equals("RevisionNotFound") ? "no such revision" : "no such file at that revision") + ").");
            case 429 -> new HubException(Reason.RATE_LIMITED, repo, what + ": rate limited (HTTP 429). Try again later"
                    + (token == null ? ", or set HF_TOKEN." : "."));
            default -> new HubException(Reason.PROTOCOL, repo, what + ": HTTP " + code
                    + h.firstValue("X-Error-Message").map(m -> ": " + m).orElse(""));
        };
    }

    static String encode(String s) { return URLEncoder.encode(s, StandardCharsets.UTF_8).replace("+", "%20"); }

    static String encodePath(String path) {
        StringBuilder b = new StringBuilder();
        for (String seg : path.split("/")) { if (!b.isEmpty()) b.append('/'); b.append(encode(seg)); }
        return b.toString();
    }

    private static String unquote(String etag) {
        String e = etag.trim();
        if (e.startsWith("W/")) e = e.substring(2);
        if (e.length() >= 2 && e.startsWith("\"") && e.endsWith("\"")) e = e.substring(1, e.length() - 1);
        return e.toLowerCase(Locale.ROOT);
    }

    private static String userAgent() {
        String v = HubHttp.class.getPackage().getImplementationVersion();
        return "kiln/" + (v == null ? "dev" : v) + "; java/" + Runtime.version().feature();
    }

    /** The hash an etag names: sha256 of the content (LFS), or git's sha1 of "blob <size>\0" and the content. */
    static final class Digest {
        final String name;
        private final MessageDigest md;
        private final byte[] prefix;
        private String hex;

        private Digest(String name, String algorithm, byte[] prefix) {
            this.name = name;
            this.prefix = prefix;
            try { this.md = MessageDigest.getInstance(algorithm); }
            catch (NoSuchAlgorithmException e) { throw new IllegalStateException(e); }
            reset();
        }

        /** The digest to check {@code f} against, or null when its etag is neither kind (or a git blob of unknown size). */
        static Digest of(RemoteFile f) {
            if (f.etag().matches("[0-9a-f]{64}")) return new Digest("sha256", "SHA-256", new byte[0]);
            if (f.etag().matches("[0-9a-f]{40}") && f.size() >= 0)
                return new Digest("git blob sha1", "SHA-1", ("blob " + f.size() + "\0").getBytes(StandardCharsets.US_ASCII));
            return null;
        }

        void reset() { md.reset(); md.update(prefix); hex = null; }

        void update(byte[] b, int n) { md.update(b, 0, n); }

        void hash(Path file, Cancel cancel) throws IOException {
            try (InputStream in = Files.newInputStream(file)) {
                byte[] buf = new byte[1 << 20];
                for (int n; (n = in.read(buf)) > 0; ) { cancel.check(); md.update(buf, 0, n); }
            }
        }

        String hex() { if (hex == null) hex = HexFormat.of().formatHex(md.digest()); return hex; }

        boolean matches(String etag) { return hex().equalsIgnoreCase(etag); }
    }

    /** A pull's cancel switch: a flag the loops check, and the stream in flight, closed to unblock a read. */
    static final class Cancel {
        private volatile boolean cancelled;
        private volatile InputStream current;

        void cancel() {
            cancelled = true;
            InputStream in = current;
            if (in != null) try { in.close(); } catch (IOException ignored) {}
        }

        boolean cancelled() { return cancelled || Thread.currentThread().isInterrupted(); }

        void check() {
            if (cancelled()) throw new CancellationException("the pull was cancelled");
        }

        void sleep(Duration d) {
            try { Thread.sleep(d); }
            catch (InterruptedException e) { Thread.currentThread().interrupt(); }
            check();
        }

        InputStream track(InputStream in) {
            current = in;
            if (cancelled) cancel();
            return in;
        }

        void untrack() { current = null; }
    }
}
