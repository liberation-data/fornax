package org.liberationdata.kiln.hub;

import org.liberationdata.kiln.HfCache;

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
     * {@code HF_TOKEN}, else the file {@code hf auth login} writes: {@code HF_TOKEN_PATH}, else {@code token} in
     * {@link HfCache#home()} ({@code $HF_HOME}, {@code $XDG_CACHE_HOME/huggingface} or {@code ~/.cache/huggingface}).
     */
    public static Optional<String> find() {
        String env = System.getenv("HF_TOKEN");
        if (env != null && !env.isBlank()) return Optional.of(env.trim());
        return fromFile(tokenFile());
    }

    /** Where {@code hf auth login} keeps the token. */
    public static Path tokenFile() {
        String path = System.getenv("HF_TOKEN_PATH");
        return path != null && !path.isBlank() ? HfCache.expandHome(path) : HfCache.home().resolve("token");
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
