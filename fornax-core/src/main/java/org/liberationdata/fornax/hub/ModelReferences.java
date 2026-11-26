package org.liberationdata.fornax.hub;

import org.liberationdata.fornax.hub.ModelSet.Kind;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * What an engine's constructor does with the model it's given. An existing file, or a string that isn't a reference,
 * is used as it is. A {@link ModelRef} ({@code owner/repo[@revision][:tag]}) or a catalogue alias
 * ({@code whisper-base.en}) is downloaded first, with {@link FornaxHub#fetch}. One place, so that every engine takes
 * references the same way and refuses the wrong kind of model before downloading anything.
 */
public final class ModelReferences {
    private ModelReferences() {}

    /**
     * What an engine loads, for its checks and messages.
     *
     * @param name  the class, for messages: {@code "WhisperModel"}
     * @param loads what it loads, for messages: {@code "speech-to-text models"}
     * @param kinds the catalogue kinds it loads; an alias of another kind is refused
     * @param roles the role the model's file must have, one of: {@code "gguf"} for a GGUF, {@code "model"} for any other file
     * @param needs the file it needs, for messages: {@code "a whisper.cpp model"}
     */
    public record Engine(String name, String loads, Set<Kind> kinds, Set<String> roles, String needs) {
        public Engine {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(loads, "loads");
            kinds = Set.copyOf(kinds);
            roles = Set.copyOf(roles);
            Objects.requireNonNull(needs, "needs");
        }
    }

    /** {@link #locate(String, Engine, FornaxHub)} with {@link FornaxHub#standard()}, built only for a reference. */
    public static Optional<Map<String, Path>> locate(String s, Engine engine) {
        Objects.requireNonNull(s, "model");
        return isPath(s) ? Optional.empty() : locate(s, engine, FornaxHub.standard());   // the usual case: no hub needed
    }

    /** Whether s is a file or folder that exists, so names no reference. */
    public static boolean isPath(String s) {
        try { return Files.exists(Path.of(s)); }
        catch (InvalidPathException notAPath) { return false; }   // a reference's ':' on Windows
    }

    /**
     * The files s names, by role as {@link FornaxHub#fetch} gives them, downloaded if they aren't in the cache yet; or
     * empty when s is a path, to use as given.
     *
     * @throws IllegalArgumentException for a Hugging Face URL (naming the reference to use instead), an alias of a
     *         kind the engine doesn't load, or a reference to a file it can't load; all before anything is downloaded
     * @throws UncheckedIOException when the download fails, as {@link FornaxHub#fetch} throws
     */
    public static Optional<Map<String, Path>> locate(String s, Engine engine, FornaxHub hub) {
        Objects.requireNonNull(s, "model");
        if (ModelRef.urlHint(s).isPresent()) ModelRef.parse(s);   // throws, naming the reference to use instead
        if (!hub.isReference(s)) return Optional.empty();
        var alias = hub.alias(s);   // refused before fetching: an image model's files are gigabytes
        if (alias.isPresent() && !engine.kinds().contains(alias.get().kind()))
            throw new IllegalArgumentException(s + " is a " + alias.get().kind().name().toLowerCase(Locale.ROOT) + " model ("
                    + alias.get().engine() + "); " + engine.name() + " loads " + engine.loads());
        try {
            ModelSet set = alias.isPresent() ? alias.get() : hub.resolve(ModelRef.parse(s));   // fetch, checked first
            if (set.parts().stream().noneMatch(p -> engine.roles().contains(p.role())))
                throw new IllegalArgumentException(s + " isn't " + engine.needs() + " (it names "
                        + set.parts().stream().map(p -> p.primary().file()).toList() + ")");
            return Optional.of(hub.pull(set, PullListener.logging()));
        } catch (IOException e) { throw new UncheckedIOException(s + ": " + e.getMessage(), e); }
    }

    /** The model's file among what {@link #locate} found: the first of the engine's roles that's there. */
    public static Path file(Map<String, Path> files, Engine engine) {
        return engine.roles().stream().sorted().map(files::get).filter(Objects::nonNull).findFirst()
                .orElseThrow(() -> new IllegalStateException("no " + engine.roles() + " in " + files.keySet()));
    }
}
