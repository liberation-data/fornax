package org.liberationdata.kiln.hub;

import java.net.URI;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

/** Pulls one file in a JVM of its own, for the test of two processes pulling the same file. Args: cache, endpoint, repo, file. */
final class PullMain {
    public static void main(String[] args) throws Exception {
        KilnHub hub = KilnHub.builder().cacheRoot(Path.of(args[0])).endpoint(URI.create(args[1])).token(null)
                .modelFolders(ModelFolders.none()).offline(false).reserveBytes(0).retryDelay(Duration.ofMillis(10))
                .catalogue(List.of()).build();
        var set = ModelSet.of("one", ModelSet.Kind.IMAGE, "one", "test", ModelSet.Part.of("model", args[2], args[3]));
        System.out.println("pulled " + hub.pull(set, PullListener.NONE).get("model"));
    }

    private PullMain() {}
}
