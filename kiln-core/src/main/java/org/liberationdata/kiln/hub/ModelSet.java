package org.liberationdata.kiln.hub;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * A model as Kiln loads it: a named set of files, each in a role. An image model is not one file but a diffusion model,
 * a VAE and a text encoder, often from different Hugging Face repos; LTX-2.5 is five. Naming the set, rather than each
 * file, lets {@link KilnHub} pull, report and evict the model as one thing, and lets the CLI, Kiln Studio and other
 * apps share one list of them ({@link ModelCatalog}).
 *
 * @param id      the name to ask for it by, such as {@code "flux2-klein"}
 * @param kind    what it makes, which says which engine and window it belongs to
 * @param title   a name for people, such as {@code "FLUX.2 [klein] base 4B"}
 * @param engine  the native engine that runs it, such as {@code "stable-diffusion.cpp"}
 * @param parts   its files, in the order the engine is given them
 * @param license the weights' license, where it limits what may be done with them or what they make; null otherwise
 * @param gated   whether a repo it comes from asks you to accept its terms first, so pulling it needs a token
 * @param aliases other names it is known by (Kiln Studio's ids, such as {@code "image:flux2-klein"})
 */
public record ModelSet(String id, Kind kind, String title, String engine, List<Part> parts, String license,
                       boolean gated, List<String> aliases) {

    /** What a model makes. */
    public enum Kind { CHAT, VISION, SPEECH, TRANSCRIPTION, IMAGE, SONG, VIDEO }

    /**
     * Where a file comes from: a path in a Hugging Face repo at a revision (a branch, tag or commit; usually
     * {@code "main"}). A 40-character commit pins the file; a branch follows the repo.
     */
    public record Source(String repo, String file, String revision) {
        public Source {
            Objects.requireNonNull(repo, "repo");
            Objects.requireNonNull(file, "file");
            if (revision == null || revision.isBlank()) revision = "main";
            if (!repo.matches("[\\w.-]+/[\\w.-]+")) throw new IllegalArgumentException("not an org/name repo: " + repo);
            if (file.startsWith("/") || file.contains("..")) throw new IllegalArgumentException("not a path in a repo: " + file);
            if (revision.startsWith("/") || revision.contains("..") || revision.contains("\\"))   // it names refs/<revision>
                throw new IllegalArgumentException("not a revision: " + revision);
        }

        public Source(String repo, String file) { this(repo, file, "main"); }

        /** The last segment of {@link #file}: {@code "flux2-vae.safetensors"} of {@code "split_files/vae/flux2-vae.safetensors"}. */
        public String fileName() { return file.substring(file.lastIndexOf('/') + 1); }
    }

    /**
     * One file a model needs, by role. The first source is the one {@link KilnHub#pull} downloads; the others are the
     * same file published in other repos, accepted when one of them is already on disk (Qwen-Image's text encoder is
     * also in the HunyuanVideo repo, and a machine with either need not download it again).
     *
     * @param role what the engine uses the file for: {@code diffusion_model}, {@code vae}, {@code llm}, {@code mmproj}...
     */
    public record Part(String role, List<Source> sources) {
        public Part {
            Objects.requireNonNull(role, "role");
            sources = List.copyOf(sources);
            if (sources.isEmpty()) throw new IllegalArgumentException(role + ": a part needs a source");
        }

        public static Part of(String role, String repo, String file) { return new Part(role, List.of(new Source(repo, file))); }

        public static Part anyOf(String role, Source... sources) { return new Part(role, List.of(sources)); }

        /** The source that is downloaded when none is on disk. */
        public Source primary() { return sources.get(0); }

        /** Whether this part is the file {@code repo}/{@code file} under any of its sources. */
        public boolean uses(String repo, String file) {
            return sources.stream().anyMatch(s -> s.repo().equals(repo) && s.file().equals(file));
        }
    }

    public ModelSet {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(title, "title");
        Objects.requireNonNull(engine, "engine");
        parts = List.copyOf(parts);
        aliases = List.copyOf(aliases);
        if (parts.isEmpty()) throw new IllegalArgumentException(id + " has no parts");
        if (parts.stream().map(Part::role).distinct().count() != parts.size())
            throw new IllegalArgumentException(id + " has two parts in the same role");
    }

    /** A set with no license note, not gated, and no aliases; see {@link #withLicense}, {@link #withGated} and {@link #withAliases}. */
    public static ModelSet of(String id, Kind kind, String title, String engine, Part... parts) {
        return new ModelSet(id, kind, title, engine, List.of(parts), null, false, List.of());
    }

    public ModelSet withLicense(String license) { return new ModelSet(id, kind, title, engine, parts, license, gated, aliases); }

    public ModelSet withGated(boolean gated) { return new ModelSet(id, kind, title, engine, parts, license, gated, aliases); }

    public ModelSet withAliases(String... aliases) { return new ModelSet(id, kind, title, engine, parts, license, gated, List.of(aliases)); }

    /** The part in {@code role}. */
    public Optional<Part> part(String role) { return parts.stream().filter(p -> p.role().equals(role)).findFirst(); }

    /**
     * The same model without one of its parts, as a set of its own: LTX-2.5 without its audio VAE makes silent video
     * and needs less memory. The id and title get the suffixes; aliases are dropped, since they name the whole set.
     */
    public ModelSet without(String role, String idSuffix, String titleSuffix) {
        if (part(role).isEmpty()) throw new IllegalArgumentException(id + " has no " + role);
        List<Part> rest = new ArrayList<>(parts);
        rest.removeIf(p -> p.role().equals(role));
        return new ModelSet(id + idSuffix, kind, title + titleSuffix, engine, rest, license, gated, List.of());
    }

    /** Whether {@code name} is this set's id or one of its aliases, ignoring case. */
    public boolean isNamed(String name) {
        return id.equalsIgnoreCase(name) || aliases.stream().anyMatch(a -> a.equalsIgnoreCase(name));
    }
}
