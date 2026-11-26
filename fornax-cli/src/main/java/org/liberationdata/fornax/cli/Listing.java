package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.HfCache;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelCatalog;
import org.liberationdata.fornax.hub.ModelSet;
import org.liberationdata.fornax.hub.ModelSet.Kind;
import org.liberationdata.fornax.hub.ModelStatus;
import org.liberationdata.fornax.hub.ModelStatus.State;

import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** {@code fornax list}: the catalogue's models and what of each is on disk, then the other GGUF models in the cache. */
final class Listing {
    static final String USAGE = """
            usage: fornax list

            Shows the catalogue's models (the aliases fornax pull takes), whether each is downloaded, partly
            downloaded or not yet, with sizes on disk; then any other GGUF language models in the Hugging Face
            cache, which fornax chat can run by their org/name:file.""";

    /** A GGUF language model in the cache that no catalogue alias names. */
    record LlmEntry(String repo, String file, long size, int parts) {
        String spec() { return repo + ":" + file; }
    }

    /** Cached GGUF models outside the catalogue, skipping repos whose GGUFs belong to an image, song or speech engine. */
    static List<LlmEntry> otherLlms(Path cacheRoot) {
        List<LlmEntry> out = new ArrayList<>();
        for (String repo : HfCache.repos(cacheRoot)) {
            if (isEngineRepo(repo)) continue;
            Map<String, Path> files = HfCache.files(cacheRoot, repo);
            for (var e : Gguf.models(files.keySet()).entrySet()) {
                if (!files.keySet().containsAll(Gguf.allParts(e.getValue()))) continue;   // an incomplete split model
                if (inCatalogue(repo, e.getKey())) continue;
                long size = e.getValue().stream().mapToLong(p -> size(files.get(p))).sum();
                out.add(new LlmEntry(repo, e.getKey(), size, e.getValue().size()));
            }
        }
        return out;
    }

    private static boolean inCatalogue(String repo, String file) {
        return ModelCatalog.all().stream().anyMatch(m -> m.parts().stream().anyMatch(p -> p.uses(repo, file)));
    }

    /** Repos whose files feed engines other than chat: their GGUFs (ACE-Step's LM, YuE2) aren't chat models. */
    static boolean isEngineRepo(String repo) {
        return ModelCatalog.all().stream().filter(m -> m.kind() != Kind.CHAT && m.kind() != Kind.VISION)
                .flatMap(m -> m.parts().stream()).flatMap(p -> p.sources().stream()).anyMatch(s -> s.repo().equals(repo));
    }

    static int run(FornaxHub hub, PrintStream out) {
        out.println("Cache: " + hub.cacheRoot() + (hub.modelFolders().dirs().isEmpty() ? "" : ", then " + hub.modelFolders().dirs()));
        List<String> missing = new ArrayList<>();
        for (Kind kind : Kind.values()) {
            List<ModelSet> sets = ModelCatalog.of(kind);
            List<String> lines = new ArrayList<>();
            int w = sets.stream().mapToInt(m -> m.id().length()).max().orElse(0);
            for (ModelSet m : sets) {
                ModelStatus s = hub.status(m);
                if (s.state() == State.MISSING) { missing.add(m.id()); continue; }
                String state = s.state() == State.PRESENT ? "" : "  partly downloaded (fornax pull " + m.id() + " finishes it)";
                lines.add(String.format("  %-" + w + "s  %9s  %s%s", m.id(), Fmt.bytes(s.bytesOnDisk()), m.title(), state));
            }
            if (lines.isEmpty()) continue;
            out.println();
            out.println(heading(kind) + ":");
            lines.forEach(out::println);
        }
        List<LlmEntry> others = otherLlms(hub.cacheRoot());
        if (!others.isEmpty()) {
            out.println();
            out.println("Other GGUF models in the cache (fornax chat <org/name:file>):");
            int w = others.stream().mapToInt(e -> e.spec().length()).max().orElse(0);
            for (LlmEntry e : others)
                out.printf("  %-" + w + "s  %9s%s%n", e.spec(), Fmt.bytes(e.size()), e.parts() > 1 ? "  (" + e.parts() + " parts)" : "");
        }
        if (!missing.isEmpty()) {
            out.println();
            out.println("Not downloaded (fornax pull <alias>): " + String.join(", ", missing));
        }
        return 0;
    }

    private static String heading(Kind kind) {
        return switch (kind) {
            case CHAT -> "Chat (fornax chat <alias>)";
            case VISION -> "Vision (fornax chat <alias> for text; images from Java)";
            case IMAGE -> "Image (fornax test <alias>)";
            case SONG -> "Song (fornax test <alias>)";
            default -> kind.name().charAt(0) + kind.name().substring(1).toLowerCase(Locale.ROOT) + " (from Java)";
        };
    }

    private static long size(Path p) {
        try { return Files.size(p); } catch (IOException e) { throw new UncheckedIOException(e); }
    }
}
