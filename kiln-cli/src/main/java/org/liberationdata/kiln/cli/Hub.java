package org.liberationdata.kiln.cli;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.LongConsumer;

/** The Hugging Face Hub's HTTP API: repo listings and file downloads. */
final class Hub {
    /** A file in a repo at some commit. {@code blobId} names it in the cache: the sha256 for LFS files, else the git blob id. */
    record RepoFile(String path, long size, String blobId, String sha256) {}

    record RepoInfo(String repo, String commit, String gated, List<RepoFile> files) {
        Optional<RepoFile> file(String path) { return files.stream().filter(f -> f.path().equals(path)).findFirst(); }
    }

    /** An HTTP failure with a message meant for people. */
    static final class HubException extends IOException {
        private static final long serialVersionUID = 1L;
        HubException(String message) { super(message); }
    }

    private final URI endpoint;
    private final String token;
    private final HttpClient http;

    Hub(URI endpoint, String token) {
        this.endpoint = endpoint;
        this.token = token;
        this.http = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NEVER)
                .connectTimeout(Duration.ofSeconds(30)).build();
    }

    /** {@code HF_ENDPOINT} (default https://huggingface.co) and the token {@link #token()} finds. */
    static Hub standard() {
        String ep = System.getenv("HF_ENDPOINT");
        return new Hub(URI.create(ep == null || ep.isBlank() ? "https://huggingface.co" : ep.replaceAll("/+$", "")),
                token().map(Map.Entry::getValue).orElse(null));
    }

    /**
     * The access token and where it came from: {@code HF_TOKEN}, else the file {@code hf auth login} writes
     * ({@code HF_TOKEN_PATH}, else {@code $HF_HOME/token}, else {@code ~/.cache/huggingface/token}).
     */
    static Optional<Map.Entry<String, String>> token() {
        String env = System.getenv("HF_TOKEN");
        if (env != null && !env.isBlank()) return Optional.of(Map.entry("HF_TOKEN", env.trim()));
        String path = System.getenv("HF_TOKEN_PATH");
        if (path == null || path.isBlank()) {
            String home = System.getenv("HF_HOME");
            path = (home == null || home.isBlank()
                    ? Path.of(System.getProperty("user.home"), ".cache", "huggingface") : Path.of(home)).resolve("token").toString();
        }
        try {
            Path p = Path.of(path);
            if (Files.isRegularFile(p)) {
                String t = Files.readString(p).trim();
                if (!t.isEmpty()) return Optional.of(Map.entry(p.toString(), t));
            }
        } catch (IOException ignored) {}
        return Optional.empty();
    }

    URI endpoint() { return endpoint; }

    /** The repo's files at {@code revision} (a branch, tag or commit), with sizes and blob ids. */
    RepoInfo info(String repo, String revision) throws IOException {
        URI uri = URI.create(endpoint + "/api/models/" + repo + "/revision/" + encode(revision) + "?blobs=true");
        HttpResponse<String> r = send(get(uri).build(), HttpResponse.BodyHandlers.ofString());
        if (r.statusCode() != 200) throw failure(r.statusCode(), repo, null, r.body());
        Map<String, Object> json = Json.object(r.body());
        List<RepoFile> files = new ArrayList<>();
        if (json.get("siblings") instanceof List<?> siblings) {
            for (Object o : siblings) {
                if (!(o instanceof Map<?, ?> s)) continue;
                String path = (String) s.get("rfilename");
                long size = s.get("size") instanceof Number n ? n.longValue() : -1;
                String blob = (String) s.get("blobId"), sha = null;
                if (s.get("lfs") instanceof Map<?, ?> lfs && lfs.get("sha256") instanceof String h) {
                    sha = h;
                    blob = h;
                    if (lfs.get("size") instanceof Number n) size = n.longValue();
                }
                if (path != null && blob != null) files.add(new RepoFile(path, size, blob, sha));
            }
        }
        if (!(json.get("sha") instanceof String)) throw new HubException(repo + ": the Hub didn't say which commit " + revision + " is");
        Object gated = json.get("gated");
        return new RepoInfo(repo, (String) json.get("sha"), gated instanceof String g ? g : null, List.copyOf(files));
    }

    /**
     * Downloads {@code file} into {@code part}, resuming from what is already there, and checks its size and
     * (for LFS files) sha256. A mismatch deletes {@code part}. {@code progress} receives the bytes done so far.
     */
    void download(String repo, String commit, RepoFile file, Path part, LongConsumer progress) throws IOException {
        URI uri = URI.create(endpoint + "/" + repo + "/resolve/" + commit + "/" + encodePath(file.path()));
        MessageDigest sha = file.sha256() == null ? null : sha256();
        IOException last = null;
        for (int attempt = 1; attempt <= 4; attempt++) {
            try {
                fetch(uri, repo, file, part, sha, progress);
                last = null;
                break;
            } catch (HubException e) {
                throw e;
            } catch (IOException e) {
                last = e;   // dropped connection: resume from what we have
                if (sha != null) sha.reset();
                try { Thread.sleep(1000L * attempt); } catch (InterruptedException ie) { Thread.currentThread().interrupt(); throw e; }
            }
        }
        if (last != null) throw last;
        long have = Files.size(part);
        if (file.size() >= 0 && have != file.size()) {
            Files.delete(part);
            throw new HubException(file.path() + ": got " + have + " bytes, expected " + file.size());
        }
        if (sha != null) {
            String got = HexFormat.of().formatHex(sha.digest());
            if (!got.equalsIgnoreCase(file.sha256())) {
                Files.delete(part);
                throw new HubException(file.path() + ": sha256 " + got + " doesn't match " + file.sha256());
            }
        }
    }

    private void fetch(URI uri, String repo, RepoFile file, Path part, MessageDigest sha, LongConsumer progress) throws IOException {
        long have = Files.exists(part) ? Files.size(part) : 0;
        if (file.size() >= 0 && have > file.size()) { Files.delete(part); have = 0; }
        if (sha != null && have > 0) hashExisting(part, sha);
        if (file.size() >= 0 && have == file.size()) return;

        HttpResponse<InputStream> r = null;
        for (int hops = 0; hops < 10; hops++) {
            HttpRequest.Builder req = get(uri);   // the token goes to the Hub only, not to the CDN it redirects to
            if (have > 0) req.header("Range", "bytes=" + have + "-");
            r = send(req.build(), HttpResponse.BodyHandlers.ofInputStream());
            int code = r.statusCode();
            if (code < 300 || code >= 400) break;
            String loc = r.headers().firstValue("Location").orElseThrow(() -> new HubException("redirect without Location"));
            r.body().close();
            uri = uri.resolve(loc);
        }
        int code = r.statusCode();
        if (code == 416 && file.size() >= 0 && have == file.size()) { r.body().close(); return; }
        if (code != 200 && code != 206) {
            String body;
            try (InputStream in = r.body()) { body = new String(in.readNBytes(2000), StandardCharsets.UTF_8); }
            throw failure(code, repo, file.path(), body);
        }
        boolean append = code == 206;
        if (!append && have > 0) {   // the server sent the whole file: start over
            have = 0;
            if (sha != null) sha.reset();
        }
        progress.accept(have);
        Files.createDirectories(part.getParent());
        try (InputStream in = r.body();
             OutputStream out = Files.newOutputStream(part, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
                     append ? StandardOpenOption.APPEND : StandardOpenOption.TRUNCATE_EXISTING)) {
            byte[] buf = new byte[1 << 20];
            for (int n; (n = in.read(buf)) > 0; ) {
                out.write(buf, 0, n);
                if (sha != null) sha.update(buf, 0, n);
                have += n;
                progress.accept(have);
            }
        }
    }

    private static void hashExisting(Path part, MessageDigest sha) throws IOException {
        try (InputStream in = Files.newInputStream(part)) {
            byte[] buf = new byte[1 << 20];
            for (int n; (n = in.read(buf)) > 0; ) sha.update(buf, 0, n);
        }
    }

    private HttpRequest.Builder get(URI uri) {
        HttpRequest.Builder b = HttpRequest.newBuilder(uri).header("User-Agent", userAgent());
        if (token != null && sameHost(uri)) b.header("Authorization", "Bearer " + token);
        return b;
    }

    private boolean sameHost(URI uri) {
        return uri.getHost() != null && uri.getHost().equalsIgnoreCase(endpoint.getHost());
    }

    private <T> HttpResponse<T> send(HttpRequest req, HttpResponse.BodyHandler<T> handler) throws IOException {
        try { return http.send(req, handler); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IOException("interrupted", e); }
    }

    private HubException failure(int code, String repo, String path, String body) {
        String what = path == null ? repo : repo + " " + path;
        return new HubException(switch (code) {
            case 401, 403 -> what + ": access denied (HTTP " + code + "). "
                    + (token == null ? "This repo may be gated or private: accept its terms on " + endpoint + "/" + repo
                       + ", then set HF_TOKEN (or run `hf auth login`)."
                       : "Your token was sent; check that it can read " + repo + " and that you accepted its terms on "
                       + endpoint + "/" + repo + ".");
            case 404 -> what + ": not found on " + endpoint + " (check the spelling; repo names are case-sensitive).";
            case 429 -> what + ": rate limited (HTTP 429). Try again later" + (token == null ? ", or set HF_TOKEN." : ".");
            default -> what + ": HTTP " + code + (body == null || body.isBlank() ? "" : ": " + body.strip());
        });
    }

    static String encode(String s) { return URLEncoder.encode(s, StandardCharsets.UTF_8).replace("+", "%20"); }

    static String encodePath(String path) {
        StringBuilder b = new StringBuilder();
        for (String seg : path.split("/")) { if (!b.isEmpty()) b.append('/'); b.append(encode(seg)); }
        return b.toString();
    }

    private static String userAgent() {
        String v = Hub.class.getPackage().getImplementationVersion();
        return "kiln/" + (v == null ? "dev" : v) + "; java/" + Runtime.version().feature();
    }

    private static MessageDigest sha256() {
        try { return MessageDigest.getInstance("SHA-256"); }
        catch (NoSuchAlgorithmException e) { throw new IllegalStateException(e); }
    }
}
