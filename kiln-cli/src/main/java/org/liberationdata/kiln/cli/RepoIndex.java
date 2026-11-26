package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.hub.HubException;
import org.liberationdata.kiln.hub.HubException.Reason;
import org.liberationdata.kiln.hub.HubToken;
import org.liberationdata.kiln.llama.Json;

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
 * A repo's file list from the Hub's API, for {@code kiln pull org/name}'s GGUF picker, tags and globs. Only the list:
 * the files themselves are downloaded by {@link org.liberationdata.kiln.hub.KilnHub}.
 */
final class RepoIndex {
    /** A file in a repo, with its size (-1 when the Hub didn't say). */
    record RepoFile(String path, long size) {}

    /** A repo's files at a revision; {@code commit} is the commit the revision named when asked. */
    record RepoInfo(String repo, String commit, List<RepoFile> files) {
        Optional<RepoFile> file(String path) { return files.stream().filter(f -> f.path().equals(path)).findFirst(); }
    }

    private final URI endpoint;
    private final String token;
    private final HttpClient http = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL)
            .connectTimeout(Duration.ofSeconds(30)).build();

    RepoIndex(URI endpoint, String token) {
        this.endpoint = URI.create(endpoint.toString().replaceAll("/+$", ""));
        this.token = token;
    }

    /** {@code HF_ENDPOINT} (default https://huggingface.co) and the token {@link HubToken#find()} finds, as KilnHub uses. */
    static RepoIndex standard() { return new RepoIndex(endpoint(), HubToken.find().orElse(null)); }

    static URI endpoint() {
        String ep = System.getenv("HF_ENDPOINT");
        return URI.create(ep == null || ep.isBlank() ? "https://huggingface.co" : ep.strip());
    }

    /** The repo's files at {@code revision} (a branch, tag or commit). */
    RepoInfo info(String repo, String revision) throws IOException {
        URI uri = URI.create(endpoint + "/api/models/" + repo + "/revision/" + encode(revision) + "?blobs=true");
        HttpRequest.Builder req = HttpRequest.newBuilder(uri).header("User-Agent", "kiln-cli");
        if (token != null) req.header("Authorization", "Bearer " + token);
        HttpResponse<String> r;
        try { r = http.send(req.build(), HttpResponse.BodyHandlers.ofString()); }
        catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new IOException("interrupted", e); }
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
