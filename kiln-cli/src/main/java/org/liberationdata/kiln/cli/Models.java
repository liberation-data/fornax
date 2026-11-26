package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.HfCache;
import org.liberationdata.kiln.hub.KilnHub;
import org.liberationdata.kiln.hub.ModelCatalog;
import org.liberationdata.kiln.hub.ModelSet;
import org.liberationdata.kiln.hub.ModelSet.Kind;
import org.liberationdata.kiln.hub.ModelStatus;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Turns a {@code <model>} argument into something to load. Accepted, in this order:
 * <ul>
 *   <li>a path to a .gguf file</li>
 *   <li>an alias from {@link ModelCatalog} ({@code qwen3-4b}, {@code ace-step}, ...), found by {@link KilnHub}</li>
 *   <li>{@code org/name:file.gguf} or {@code org/name:TAG} (a quant such as Q4_K_M) from the cache</li>
 *   <li>{@code org/name} when the cache holds exactly one GGUF model from it</li>
 * </ul>
 */
final class Models {
    private Models() {}

    sealed interface Target permits Llm, Set {}

    /** A GGUF language model. {@code name} is what the user typed, for messages. */
    record Llm(String name, Path path) implements Target {}

    /** A catalogue set that isn't a chat model (image, song, ...), with its files by role. */
    record Set(ModelSet set, Map<String, Path> files) implements Target {}

    /** "org/name" or "org/name:tag". */
    static final Pattern REPO = Pattern.compile("([\\w.-]+/[\\w.-]+)(?::(.+))?");

    /** The catalogue's ids, for messages. */
    static String aliases() { return ModelCatalog.all().stream().map(ModelSet::id).collect(Collectors.joining(", ")); }

    static Target resolve(String spec, KilnHub hub) {
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
                        + "). Fetch it with: kiln pull " + m.id());
            }
            if (m.kind() == Kind.CHAT || m.kind() == Kind.VISION) return new Llm(spec, status.paths().get("gguf"));
            return new Set(m, status.paths());
        }
        Matcher rm = REPO.matcher(spec);
        if (rm.matches()) return new Llm(spec, fromRepo(rm.group(1), rm.group(2), hub.cacheRoot()));
        throw new Main.Failure("No such file, alias or repo: " + spec + "\n  aliases: " + aliases());
    }

    private static Path fromRepo(String repo, String tag, Path cacheRoot) {
        Map<String, Path> byPath = HfCache.files(cacheRoot, repo);
        Map<String, List<String>> models = Gguf.models(byPath.keySet());
        List<String> hits = tag == null ? List.copyOf(models.keySet()) : Gguf.match(models, tag);
        String want = repo + (tag == null ? "" : ":" + tag);
        if (hits.size() == 1) {
            List<String> parts = models.get(hits.get(0));
            List<String> missing = Gguf.allParts(parts).stream().filter(x -> !byPath.containsKey(x)).toList();
            if (!missing.isEmpty()) throw new Main.Failure(want + " is incomplete in the cache (missing " + missing
                    + "). Fetch it with: kiln pull " + repo + ":" + hits.get(0));
            return byPath.get(hits.get(0));
        }
        if (hits.isEmpty()) {
            String have = models.isEmpty() ? "" : "\n  in the cache from " + repo + ": " + String.join(", ", models.keySet());
            throw new Main.Failure("No GGUF model " + want + " in the cache (" + cacheRoot + ")." + have
                    + "\n  fetch one with: kiln pull " + want);
        }
        throw new Main.Failure(want + " matches several models; name one:\n" + hits.stream()
                .map(h -> "  " + repo + ":" + h).collect(Collectors.joining("\n")));
    }
}
