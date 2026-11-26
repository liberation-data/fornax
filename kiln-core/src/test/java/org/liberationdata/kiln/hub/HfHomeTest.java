package org.liberationdata.kiln.hub;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.kiln.HfCache;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/** Where the cache and token are for each environment, as huggingface_hub has them; in a JVM of its own, to set the environment. */
class HfHomeTest {
    @TempDir Path dir;

    /** Prints the cache root and the token file. */
    static final class Print {
        public static void main(String[] args) {
            System.out.println("root=" + HfCache.root());
            System.out.println("token=" + HubToken.tokenFile());
        }
    }

    Map<String, Path> run(Map<String, String> env) throws Exception {
        var pb = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-Duser.home=" + dir.resolve("home"), "-cp", System.getProperty("java.class.path"), Print.class.getName())
                .redirectErrorStream(true);
        for (String k : List.of("HF_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME", "HF_TOKEN_PATH")) pb.environment().remove(k);
        pb.environment().putAll(env);
        Process p = pb.start();
        String out = new String(p.getInputStream().readAllBytes());
        assertTrue(p.waitFor(60, TimeUnit.SECONDS));
        assertEquals(0, p.exitValue(), out);
        Map<String, Path> m = new java.util.HashMap<>();
        for (String line : out.split("\n")) {
            int eq = line.indexOf('=');
            if (eq > 0) m.put(line.substring(0, eq), Path.of(line.substring(eq + 1).trim()));
        }
        return m;
    }

    @Test void defaultsToTheHomeCache() throws Exception {
        var m = run(Map.of());
        assertEquals(dir.resolve("home/.cache/huggingface/hub"), m.get("root"));
        assertEquals(dir.resolve("home/.cache/huggingface/token"), m.get("token"));
    }

    @Test void followsXdgCacheHomeAsHfDoes() throws Exception {
        var m = run(Map.of("XDG_CACHE_HOME", dir.resolve("xdg").toString()));
        assertEquals(dir.resolve("xdg/huggingface/hub"), m.get("root"));
        assertEquals(dir.resolve("xdg/huggingface/token"), m.get("token"));
    }

    @Test void hfHomeThenHfHubCacheWin() throws Exception {
        var m = run(Map.of("XDG_CACHE_HOME", dir.resolve("xdg").toString(), "HF_HOME", dir.resolve("hf").toString()));
        assertEquals(dir.resolve("hf/hub"), m.get("root"));
        assertEquals(dir.resolve("hf/token"), m.get("token"));
        m = run(Map.of("HF_HOME", dir.resolve("hf").toString(), "HF_HUB_CACHE", dir.resolve("cache").toString(),
                "HF_TOKEN_PATH", dir.resolve("t").toString()));
        assertEquals(dir.resolve("cache"), m.get("root"));
        assertEquals(dir.resolve("t"), m.get("token"));
    }

    @Test void aLeadingTildeIsTheHomeDirectory() throws Exception {
        var m = run(Map.of("HF_HOME", "~/hf", "HF_HUB_CACHE", "~/big/hub"));
        assertEquals(dir.resolve("home/big/hub"), m.get("root"), "not a folder named ~ in the working directory");
        assertEquals(dir.resolve("home/hf/token"), m.get("token"));
    }
}
