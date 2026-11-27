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
        check(s, engine, hub);
        if (!hub.isReference(s)) return Optional.empty();
        var alias = hub.alias(s);
        try {
            ModelSet set = alias.isPresent() ? alias.get() : hub.resolve(ModelRef.parse(s));   // fetch, checked first
            if (set.parts().stream().noneMatch(p -> engine.roles().contains(p.role())))
                throw new IllegalArgumentException(s + " isn't " + engine.needs() + " (it names "
                        + set.parts().stream().map(p -> p.primary().file()).toList() + ")");
            return Optional.of(hub.pull(set, PullListener.logging()));
        } catch (IOException e) { throw new UncheckedIOException(s + ": " + e.getMessage(), e); }
    }

    /**
     * The refusals {@link #locate} makes before any request: a Hugging Face URL, or an alias of a kind the engine
     * doesn't load. For an engine that takes two models, to refuse the second before the first is downloaded.
     *
     * @throws IllegalArgumentException as {@link #locate} does for these
     */
    public static void check(String s, Engine engine) {
        Objects.requireNonNull(s, "model");
        if (!isPath(s)) check(s, engine, FornaxHub.standard());
    }

    /** {@link #check(String, Engine)} with the given hub. */
    public static void check(String s, Engine engine, FornaxHub hub) {
        Objects.requireNonNull(s, "model");
        if (ModelRef.urlHint(s).isPresent()) ModelRef.parse(s);   // throws, naming the reference to use instead
        if (!hub.isReference(s)) return;
        var alias = hub.alias(s);   // refused before fetching: an image model's files are gigabytes
        if (alias.isPresent() && !engine.kinds().contains(alias.get().kind()))
            throw new IllegalArgumentException(s + " is " + aKind(alias.get()) + "; " + engine.name() + " loads " + engine.loads());
    }

    /**
     * For an engine that takes one file per argument: s as given when it is a path, else the one file the reference
     * names, downloaded if it isn't in the cache yet. With an engine whose {@link Engine#kinds} is empty, an alias
     * (which names a whole set) is refused.
     *
     * @throws IllegalArgumentException and {@link UncheckedIOException} as {@link #locate} throws
     */
    public static String path(String s, Engine engine) {
        return locate(s, engine).map(f -> file(f, engine).toString()).orElse(s);
    }

    /** {@link #path(String, Engine)} with the given hub. */
    public static String path(String s, Engine engine, FornaxHub hub) {
        return locate(s, engine, hub).map(f -> file(f, engine).toString()).orElse(s);
    }

    /**
     * For an engine that loads several files from one name: the files of the catalogue set with that name, by role,
     * downloaded if they aren't in the cache yet.
     *
     * @param runs the native engine the set must be for ({@link ModelSet#engine()}, such as {@code "acestep.cpp"}), or
     *             null for any of the engine's kinds
     * @throws IllegalArgumentException for a name that isn't a set in the catalogue (a path or a reference names one
     *         file, for the engine's per-file arguments), or a set of another kind or native engine; before any request
     * @throws UncheckedIOException when the download fails, as {@link FornaxHub#pull} throws
     */
    public static Map<String, Path> set(String name, Engine engine, String runs) {
        return set(name, engine, runs, FornaxHub.standard());
    }

    /** {@link #set(String, Engine, String)} with the given hub. */
    public static Map<String, Path> set(String name, Engine engine, String runs, FornaxHub hub) {
        Objects.requireNonNull(name, "model");
        var found = hub.alias(name);
        if (found.isEmpty())
            throw new IllegalArgumentException(name + " isn't a model set Fornax knows; " + engine.name() + " loads "
                    + engine.loads() + " by name (" + String.join(", ", hub.catalogue().stream().filter(m -> fits(m, engine, runs))
                    .map(ModelSet::id).toList()) + "), or each file by path or reference");
        ModelSet set = found.get();
        if (!fits(set, engine, runs))
            throw new IllegalArgumentException(name + " is " + aKind(set) + "; " + engine.name() + " loads " + engine.loads());
        try { return hub.pull(set, PullListener.logging()); }
        catch (IOException e) { throw new UncheckedIOException(name + ": " + e.getMessage(), e); }
    }

    /** "a song model (acestep.cpp)", "an image model (stable-diffusion.cpp)". */
    private static String aKind(ModelSet set) {
        String kind = set.kind().name().toLowerCase(Locale.ROOT);
        return ("aeiou".indexOf(kind.charAt(0)) >= 0 ? "an " : "a ") + kind + " model (" + set.engine() + ")";
    }

    private static boolean fits(ModelSet set, Engine engine, String runs) {
        return engine.kinds().contains(set.kind()) && (runs == null || runs.equals(set.engine()));
    }

    /** The model's file among what {@link #locate} found: the first of the engine's roles that's there. */
    public static Path file(Map<String, Path> files, Engine engine) {
        return engine.roles().stream().sorted().map(files::get).filter(Objects::nonNull).findFirst()
                .orElseThrow(() -> new IllegalStateException("no " + engine.roles() + " in " + files.keySet()));
    }
}
