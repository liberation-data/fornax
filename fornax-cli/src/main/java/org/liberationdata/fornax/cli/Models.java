package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.hub.GgufFiles;
import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelCatalog;
import org.liberationdata.fornax.hub.ModelRef;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelStatus;

import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Turns a {@code <model>} argument into something to load. Accepted, in this order:
 * <ul>
 *   <li>a path to a .gguf file</li>
 *   <li>an alias from {@link ModelCatalog} ({@code qwen3-4b}, {@code ace-step}, ...), found by {@link FornaxHub}</li>
 *   <li>{@code org/name:file.gguf} or {@code org/name:TAG} (a quant such as Q4_K_M) from the cache</li>
 *   <li>{@code org/name} when the cache holds exactly one GGUF model from it</li>
 * </ul>
 * {@link #fetch} is the other way in, for the commands that make things (image, song, speak, transcribe, see): one kind
 * of model, downloaded first when it isn't in the cache yet.
 */
final class Models {
    private Models() {}

    sealed interface Target permits Llm, Set {}

    /** A GGUF language model. {@code name} is what the user typed, for messages. */
    record Llm(String name, Path path) implements Target {}

    /** A catalogue set that isn't a chat model (image, song, ...), with its files by role. */
    record Set(ModelSet set, Map<String, Path> files) implements Target {}

    /** The catalogue's ids, for messages. */
    static String aliases() { return ModelCatalog.all().stream().map(ModelSet::id).collect(Collectors.joining(", ")); }

    static Target resolve(String spec, FornaxHub hub) {
        Path p = Path.of(spec);
        if (Files.isRegularFile(p)) return new Llm(spec, p);
        var alias = ModelCatalog.byName(spec);
        if (alias.isPresent()) {
            ModelSet m = alias.get();
            ModelStatus status = hub.status(m);
            if (!status.available()) {
                String missing = status.parts().stream().filter(s -> s.state() != ModelStatus.State.PRESENT)
                        .map(s -> s.part().primary().fileName() + (s.state() == ModelStatus.State.PARTIAL ? " (partly downloaded)" : ""))
                        .collect(Collectors.joining(", "));
                throw new Main.Failure(m.id() + " isn't downloaded yet (" + m.title() + "; missing " + missing
                        + "). Fetch it with: fornax pull " + m.id());
            }
            if (m.kind() == Kind.CHAT || m.kind() == Kind.VISION) return new Llm(spec, status.paths().get("gguf"));
            return new Set(m, status.paths());
        }
        var ref = ModelRef.tryParse(spec);
        if (ref.isPresent()) {
            if (ref.get().revision() != null) throw new Main.Failure(spec + ": fornax chat uses the downloaded copy, whatever its "
                    + "revision; drop @" + ref.get().revision() + " (fornax pull " + spec + " fetches that revision)");
            return new Llm(spec, fromRepo(ref.get().repo(), ref.get().tag(), hub.cacheRoot()));
        }
        throw new Main.Failure("No such file, alias or repo: " + spec + "\n  aliases: " + aliases());
    }

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
     * or, for a command that loads one file ({@code oneFile}: speech, transcription and vision), also a path to that file
     * or an {@code org/name[@REV][:TAG]} reference. An alias or reference not yet in the cache is downloaded first,
     * with {@code fornax pull}'s progress line on {@code err}. An alias of another kind is refused before anything is
     * downloaded.
     */
    static Fetched fetch(String spec, Kind kind, String command, boolean oneFile, FornaxHub hub, PrintStream err,
                         boolean terminal) throws IOException {
        String role = kind == Kind.TRANSCRIPTION ? "model" : "gguf";
        if (oneFile && Files.isRegularFile(Path.of(spec))) return new Fetched(spec, null, Map.of(role, Path.of(spec)));
        var alias = ModelCatalog.byName(spec);
        if (alias.isPresent()) {
            ModelSet m = alias.get();
            if (m.kind() != kind) throw new Main.Failure(m.id() + " is " + aKind(m.kind()) + "; " + command + " takes " + kinds(kind));
            return new Fetched(spec, m, pulled(m, m.id() + ": " + m.title() + " (" + m.engine() + ")", hub, err, terminal));
        }
        var ref = ModelRef.tryParse(spec);
        if (ref.isPresent() && oneFile) {
            ModelSet m = hub.resolve(ref.get());   // from the cache when it's there, else the repo's file list
            if (m.part(role).isEmpty()) throw new Main.Failure(spec + " names " + m.parts().get(0).primary().file() + ", not "
                    + (kind == Kind.TRANSCRIPTION ? "a whisper.cpp model" : "a GGUF model"));   // checked before the download
            return new Fetched(spec, null, pulled(m, spec, hub, err, terminal));
        }
        if (ref.isPresent()) throw new Main.Failure(spec + ": " + command + " loads a model set from the catalogue (its files "
                + "come from several repos), not a reference; " + command + " takes " + kinds(kind));
        throw new Main.Failure("No such " + (oneFile ? "file or " : "") + "model: " + spec + "\n  " + command + " takes " + kinds(kind)
                + (oneFile ? ", org/name[:TAG] or a path" : ""));
    }

    /**
     * Whether {@code spec} is a path or a reference that {@link #fetch} would take for a one-file command: one GGUF,
     * without the mmproj an alias brings along. Asked before fetching, so a missing {@code --mmproj} is said before the
     * model downloads.
     */
    static boolean withoutMmproj(String spec) {
        return ModelCatalog.byName(spec).isEmpty() && (Files.isRegularFile(Path.of(spec)) || ModelRef.tryParse(spec).isPresent());
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

    /** The catalogue's ids of one kind: "flux2-klein, qwen-image, qwen-image-edit". */
    static String ids(Kind kind) { return ModelCatalog.of(kind).stream().map(ModelSet::id).collect(Collectors.joining(", ")); }

    private static Path fromRepo(String repo, String tag, Path cacheRoot) {
        Map<String, Path> byPath = HfCache.files(cacheRoot, repo);
        Map<String, List<String>> models = GgufFiles.models(byPath.keySet());
        List<String> hits = tag == null ? List.copyOf(models.keySet()) : GgufFiles.match(models, tag);
        String want = repo + (tag == null ? "" : ":" + tag);
        if (hits.size() == 1) {
            List<String> parts = models.get(hits.get(0));
            List<String> missing = GgufFiles.allParts(parts).stream().filter(x -> !byPath.containsKey(x)).toList();
            if (!missing.isEmpty()) throw new Main.Failure(want + " is incomplete in the cache (missing " + missing
                    + "). Fetch it with: fornax pull " + repo + ":" + hits.get(0));
            return byPath.get(hits.get(0));
        }
        if (hits.isEmpty()) {
            String have = models.isEmpty() ? "" : "\n  in the cache from " + repo + ": " + String.join(", ", models.keySet());
            throw new Main.Failure("No GGUF model " + want + " in the cache (" + cacheRoot + ")." + have
                    + "\n  fetch one with: fornax pull " + want);
        }
        throw new Main.Failure(want + " matches several models; name one:\n" + hits.stream()
                .map(h -> "  " + repo + ":" + h).collect(Collectors.joining("\n")));
    }
}
