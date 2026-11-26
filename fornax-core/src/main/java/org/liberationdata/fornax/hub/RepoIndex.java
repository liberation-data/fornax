package org.liberationdata.fornax.hub;

import org.liberationdata.fornax.Json;
import org.liberationdata.fornax.hub.HubException.Reason;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * A repo's file list from the Hub's API: what {@link FornaxHub#fetch} picks a model reference's files from, and
 * {@code fornax pull org/name}'s GGUF picker, tags and globs. Only the list: {@link FornaxHub} downloads the files.
 */
public final class RepoIndex {
    /** A file in a repo, with its size (-1 when the Hub didn't say). */
    public record RepoFile(String path, long size) {}

    /** A repo's files at a revision; {@code commit} is the commit the revision named when asked. */
    public record RepoInfo(String repo, String commit, List<RepoFile> files) {
        public Optional<RepoFile> file(String path) { return files.stream().filter(f -> f.path().equals(path)).findFirst(); }
    }

    private final URI endpoint;
    private final String token;
    private final boolean offline;
    // redirects are followed by hand, so the token goes to the endpoint only (Java 22's client forwards it anywhere)
    private final HttpClient http = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NEVER)
            .connectTimeout(Duration.ofSeconds(30)).build();

    /** The Hub at {@code endpoint}, with {@code token} (or null) for private and gated repos, online. */
    public RepoIndex(URI endpoint, String token) { this(endpoint, token, false); }

    /** {@code offline}: {@link #info} fails at once, as {@code FornaxHub} does for a missing file under HF_HUB_OFFLINE. */
    public RepoIndex(URI endpoint, String token, boolean offline) {
        this.endpoint = URI.create(endpoint.toString().replaceAll("/+$", ""));
        this.token = token;
        this.offline = offline;
    }

    /** {@code HF_ENDPOINT} (default https://huggingface.co), the token {@link HubToken#find()} finds and HF_HUB_OFFLINE, as FornaxHub uses. */
    public static RepoIndex standard() {
        String off = System.getenv("HF_HUB_OFFLINE");
        boolean offline = off != null && java.util.Set.of("1", "true", "yes", "on").contains(off.trim().toLowerCase(java.util.Locale.ROOT));
        return new RepoIndex(endpoint(), HubToken.find().orElse(null), offline);
    }

    /** {@code HF_ENDPOINT}, else https://huggingface.co. */
    public static URI endpoint() {
        String ep = System.getenv("HF_ENDPOINT");
        return URI.create(ep == null || ep.isBlank() ? "https://huggingface.co" : ep.strip());
    }

    /** The repo's files at {@code revision} (a branch, tag or commit). */
    public RepoInfo info(String repo, String revision) throws IOException {
        if (offline) throw new HubException(Reason.OFFLINE, repo, repo + ": listing its files needs the Hub, and HF_HUB_OFFLINE "
                + "is set. Unset it to download from " + repo + "; a model already in the cache is used without the Hub.");
        URI uri = URI.create(endpoint + "/api/models/" + repo + "/revision/" + encode(revision) + "?blobs=true");
        HttpResponse<String> r = null;
        for (int hops = 0; hops < 10; hops++) {   // the Hub's own redirects (a renamed repo) keep the token; others don't get it
            HttpRequest.Builder req = HttpRequest.newBuilder(uri).header("User-Agent", "fornax");
            if (token != null && HubHttp.sendsToken(uri, endpoint)) req.header("Authorization", "Bearer " + token);
            try { r = http.send(req.build(), HttpResponse.BodyHandlers.ofString()); }
            catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IOException("interrupted", e); }
            if (r.statusCode() < 300 || r.statusCode() >= 400) break;
            String loc = r.headers().firstValue("Location").orElseThrow(() ->
                    new HubException(Reason.PROTOCOL, repo, repo + ": a redirect without a Location"));
            uri = uri.resolve(loc);
        }
        if (r.statusCode() >= 300 && r.statusCode() < 400) throw new HubException(Reason.PROTOCOL, repo, repo + ": too many redirects");
        if (r.statusCode() != 200) throw failure(r.statusCode(), repo, revision);
        Object json = Json.parse(r.body());
        List<RepoFile> files = new ArrayList<>();
        if (Json.get(json, "siblings") instanceof List<?> siblings) {
            for (Object o : siblings) {
                if (!(o instanceof Map<?, ?> s) || !(s.get("rfilename") instanceof String path)) continue;
                long size = s.get("size") instanceof Number n ? n.longValue() : -1;
                if (Json.get(s, "lfs", "size") instanceof Number n) size = n.longValue();
                files.add(new RepoFile(path, size));
            }
        }
        if (!(Json.get(json, "sha") instanceof String commit))
            throw new HubException(Reason.PROTOCOL, repo, repo + ": the Hub didn't say which commit " + revision + " is");
        return new RepoInfo(repo, commit, List.copyOf(files));
    }

    private HubException failure(int code, String repo, String revision) {
        String page = endpoint + "/" + repo;
        return switch (code) {
            case 401, 403 -> new HubException(Reason.ACCESS_DENIED, repo, repo + ": access denied (HTTP " + code + "). "
                    + (token == null ? "It may be private or gated: accept its terms on " + page + ", then set HF_TOKEN "
                                       + "(or run `hf auth login`)."
                                     : "Check that the token can read " + repo + " and that you accepted its terms on " + page + "."));
            case 404 -> new HubException(Reason.NOT_FOUND, repo, repo + (revision.equals("main") ? "" : " at " + revision)
                    + ": not found on " + endpoint + " (names are case-sensitive"
                    + (token == null ? "; a private repo needs HF_TOKEN)." : ")."));
            case 429 -> new HubException(Reason.RATE_LIMITED, repo, repo + ": rate limited (HTTP 429). Try again later"
                    + (token == null ? ", or set HF_TOKEN." : "."));
            default -> new HubException(Reason.PROTOCOL, repo, repo + ": HTTP " + code + " from " + endpoint);
        };
    }

    private static String encode(String s) { return URLEncoder.encode(s, StandardCharsets.UTF_8).replace("+", "%20"); }
}
