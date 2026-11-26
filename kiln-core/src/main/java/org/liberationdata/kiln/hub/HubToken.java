package org.liberationdata.kiln.hub;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

/**
 * The Hugging Face access token, found where {@code huggingface_hub} looks, so a machine set up for {@code hf} needs no
 * more setup for Kiln. Only gated and private repos need one; Kiln sends it to the Hub's host only.
 */
public final class HubToken {
    private HubToken() {}

    /**
     * {@code HF_TOKEN}, else the file {@code hf auth login} writes: {@code HF_TOKEN_PATH}, else {@code $HF_HOME/token},
     * else {@code ~/.cache/huggingface/token}.
     */
    public static Optional<String> find() {
        String env = System.getenv("HF_TOKEN");
        if (env != null && !env.isBlank()) return Optional.of(env.trim());
        return fromFile(tokenFile());
    }

    /** Where {@code hf auth login} keeps the token. */
    public static Path tokenFile() {
        String path = System.getenv("HF_TOKEN_PATH");
        if (path != null && !path.isBlank()) return Path.of(path);
        String home = System.getenv("HF_HOME");
        return (home == null || home.isBlank() ? Path.of(System.getProperty("user.home"), ".cache", "huggingface") : Path.of(home))
                .resolve("token");
    }

    static Optional<String> fromFile(Path p) {
        try {
            if (Files.isRegularFile(p)) {
                String t = Files.readString(p).trim();
                if (!t.isEmpty()) return Optional.of(t);
            }
        } catch (IOException ignored) {   // unreadable: as if there were none
        }
        return Optional.empty();
    }
}
