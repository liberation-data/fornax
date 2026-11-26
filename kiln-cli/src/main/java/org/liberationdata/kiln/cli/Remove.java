package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.cli.Args.UsageException;
import org.liberationdata.kiln.hub.Eviction;
import org.liberationdata.kiln.hub.KilnHub;
import org.liberationdata.kiln.hub.ModelCatalog;
import org.liberationdata.kiln.hub.ModelSet;

import java.io.IOException;
import java.io.PrintStream;
import java.util.stream.Collectors;

/** {@code kiln rm}: a catalogue model's files out of the Hugging Face cache, through {@link KilnHub#evict}. */
final class Remove {
    static final String USAGE = """
            usage: kiln rm <alias>

            Deletes the model's files from the Hugging Face cache, and any partial download of them. A file
            another downloaded model also uses is kept (ACE-Step's VAE, say), and files in KILN_MODEL_DIRS are
            never touched. Other files are left to `hf cache rm`.""";

    static int run(Args a, KilnHub hub, PrintStream out) throws IOException {
        if (a.positional().size() != 1 || !a.options().isEmpty()) throw new UsageException("give one alias (see kiln list)");
        String name = a.positional(0);
        ModelSet m = ModelCatalog.byName(name).orElseThrow(() -> new Main.Failure("No alias " + name + "; kiln rm takes "
                + Models.aliases() + ". For other files, use `hf cache rm`."));
        Eviction e = hub.evict(m);
        out.println(m.id() + ": freed " + Fmt.bytes(e.bytesFreed()) + " (" + e.removed().size() + " files and links)");
        if (!e.kept().isEmpty()) out.println("  kept, as another downloaded model uses them: "
                + e.kept().stream().map(p -> p.primary().fileName()).collect(Collectors.joining(", ")));
        return 0;
    }
}
