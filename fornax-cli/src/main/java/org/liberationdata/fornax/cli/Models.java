package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.HubException;
import org.liberationdata.fornax.hub.ModelCatalog;
import org.liberationdata.fornax.hub.ModelRef;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelStatus;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Turns a {@code <model>} argument into the files to load, downloading them first when they aren't in the cache yet
 * ({@link #fetch}): a catalogue alias ({@code qwen3-4b}, {@code flux2-klein}, ...) of a kind the command runs, or, for a
 * command that loads one file (chat, see, speak, transcribe, test), also a path or an {@code org/name[@REV][:TAG]}
 * reference.
 */
final class Models {
    private Models() {}

    /** The catalogue's ids, for messages. */
    static String aliases() { return ModelCatalog.all().stream().map(ModelSet::id).collect(Collectors.joining(", ")); }

    /**
     * A model to load, by the files it needs.
     *
     * @param name  what the user typed, for messages
     * @param set   its catalogue set, or null for a file or a reference
     * @param files its files by role, as {@link FornaxHub#pull} gives them ({@code "gguf"} or {@code "model"} for one file)
     */
    record Fetched(String name, ModelSet set, Map<String, Path> files) {
        Path file(String role) {
            Path p = files.get(role);
            if (p == null) throw new IllegalStateException(name + " has no " + role + " among " + files.keySet());
            return p;
        }
    }

    /**
     * The model {@code spec} names for {@code command}, which runs {@code kind} models: a catalogue alias of that kind,
     * or, for a command that loads one file ({@code oneFile}: chat, speech, transcription, vision and test), also a path to that file
     * or an {@code org/name[@REV][:TAG]} reference. An alias or reference not yet in the cache is downloaded first,
     * with {@code fornax pull}'s progress line on {@code err}. An alias of another kind is refused before anything is
     * downloaded.
     */
    static Fetched fetch(String spec, Kind kind, String command, boolean oneFile, FornaxHub hub, PrintStream err,
                         boolean terminal) throws IOException {
        return fetch(spec, java.util.EnumSet.of(kind), command, oneFile, hub, err, terminal);
    }

    /** {@link #fetch(String, Kind, String, boolean, FornaxHub, PrintStream, boolean)} for a command that runs several kinds. */
    static Fetched fetch(String spec, java.util.Set<Kind> kinds, String command, boolean oneFile, FornaxHub hub, PrintStream err,
                         boolean terminal) throws IOException {
        String role = kinds.equals(java.util.EnumSet.of(Kind.TRANSCRIPTION)) ? "model" : "gguf";
        if (oneFile && isFile(spec)) return new Fetched(spec, null, Map.of(role, Path.of(spec)));
        checkKind(spec, kinds, command, oneFile);
        var alias = ModelCatalog.byName(spec);
        if (alias.isPresent()) {
            ModelSet m = alias.get();
            return new Fetched(spec, m, pulled(m, m.id() + ": " + m.title() + " (" + m.engine() + ")", hub, err, terminal));
        }
        var ref = ModelRef.tryParse(spec);
        if (ref.isPresent() && oneFile) {
            ModelSet m;
            try {
                m = hub.resolve(ref.get());   // from the cache when it's there, else the repo's file list
            } catch (IOException noHub) {   // the Hub couldn't answer (offline, unreachable, refused): several models in the cache is the user's to settle
                if (ref.get().revision() == null) {
                    try { ref.get().pick(HfCache.files(hub.cacheRoot(), ref.get().repo()).keySet()); }   // throws "name one" for several
                    catch (HubException noneCached) { /* nothing to name: the Hub was needed */ }
                }
                throw noHub;
            }
            if (m.part(role).isEmpty()) throw new Main.Failure(spec + " names " + m.parts().get(0).primary().file() + ", not "
                    + (role.equals("model") ? "a whisper.cpp model" : "a GGUF model"));   // checked before the download
            return new Fetched(spec, null, pulled(m, spec, hub, err, terminal));
        }
        if (ref.isPresent()) throw new Main.Failure(spec + ": " + command + " loads a model set from the catalogue (its files "
                + "come from several repos), not a reference; " + command + " takes " + kinds(kinds));
        throw new Main.Failure("No such " + (oneFile ? "file or " : "") + "model: " + spec + "\n  " + command + " takes " + kinds(kinds)
                + (oneFile ? ", org/name[:TAG] or a path" : ""));
    }

    /**
     * Refuses an alias of a kind {@code command} doesn't run, as {@link #fetch} does (a file of that name, for a
     * {@code oneFile} command, is a file): for a command that has more to check before it fetches, and whose model's
     * kind is the mistake to say first (jitllm's chat, before its SDK).
     */
    static void checkKind(String spec, java.util.Set<Kind> kinds, String command, boolean oneFile) {
        if (oneFile && isFile(spec)) return;
        var alias = ModelCatalog.byName(spec);
        if (alias.isPresent() && !kinds.contains(alias.get().kind()))
            throw new Main.Failure(alias.get().id() + " is " + aKind(alias.get().kind()) + "; " + command + " takes "
                    + kinds(kinds));
    }

    /**
     * Whether {@code spec} is a path or a reference that {@link #fetch} would take for a one-file command: one GGUF,
     * without the mmproj an alias brings along. Asked before fetching, so a missing {@code --mmproj} is said before the
     * model downloads.
     */
    static boolean withoutMmproj(String spec) {
        return ModelCatalog.byName(spec).isEmpty() && (isFile(spec) || ModelRef.tryParse(spec).isPresent());
    }

    /** Whether s is a file that exists; false for what can't be a path at all, such as a reference's ':' on Windows. */
    private static boolean isFile(String s) {
        try { return Files.isRegularFile(Path.of(s)); }
        catch (InvalidPathException notAPath) { return false; }
    }

    /** The set's files, pulled first (after {@code header}, with fornax pull's progress line) if any is missing. */
    private static Map<String, Path> pulled(ModelSet m, String header, FornaxHub hub, PrintStream err, boolean terminal) throws IOException {
        ModelStatus status = hub.status(m);
        if (status.available()) return status.paths();
        err.println("[fornax] downloading " + header);
        Pull.notes(m, err);
        return hub.pull(m, new PullLine(err, terminal));
    }

    /** "a chat model", "an image model". */
    static String aKind(Kind kind) {
        String k = kind == Kind.TRANSCRIPTION ? "speech-to-text" : kind.name().toLowerCase(Locale.ROOT);
        return ("aeiou".indexOf(k.charAt(0)) >= 0 ? "an " : "a ") + k + " model";
    }

    /** "an image model: flux2-klein, qwen-image, qwen-image-edit". */
    static String kinds(Kind kind) { return aKind(kind) + ": " + ids(kind); }

    /** "a chat model: qwen3-0.6b, qwen3-4b, or a vision model: qwen3-vl-4b, ...", for a command that runs several kinds. */
    static String kinds(java.util.Set<Kind> kinds) {
        return kinds.stream().sorted().map(Models::kinds).collect(Collectors.joining(", or "));
    }

    /** The catalogue's ids of one kind: "flux2-klein, qwen-image, qwen-image-edit". */
    static String ids(Kind kind) { return ModelCatalog.of(kind).stream().map(ModelSet::id).collect(Collectors.joining(", ")); }

}
