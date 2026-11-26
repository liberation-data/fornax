package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.cli.Args.UsageException;
import org.liberationdata.kiln.hub.HubException;
import org.liberationdata.kiln.hub.KilnHub;
import org.liberationdata.kiln.hub.ModelCatalog;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** The {@code kiln} command. */
public final class Main {
    private Main() {}

    /** A failure with a message meant for people: printed without a stack trace, exit code 1. */
    static final class Failure extends RuntimeException {
        private static final long serialVersionUID = 1L;
        Failure(String message) { super(message); }
    }

    static final String USAGE = """
            usage: kiln <command> [args]

              pull <model> [file...]   download from Hugging Face into the HF cache
              list                     the catalogue's models, what's downloaded, other GGUFs in the cache
              rm <alias>               delete a model's files from the cache
              chat <model> ["prompt"]  chat with a GGUF language model (interactive without a prompt)
              test <model>             load, generate a little, report speed
              info                     native libraries, Java and cache locations

            <model> is an alias, org/name[:TAG|:file.gguf], or a .gguf path. Aliases (Kiln's model catalogue):
            %s
            kiln <command> --help for more. Set KILN_VERBOSE=1 to see the engines' logs.""";

    private static final Map<String, String> COMMAND_USAGE = Map.of(
            "pull", Pull.USAGE, "list", Listing.USAGE, "rm", Remove.USAGE, "chat", Chat.USAGE, "test", SmokeTest.USAGE,
            "info", Info.USAGE);

    public static void main(String[] argv) {
        System.exit(run(argv, System.out, System.err));
    }

    static int run(String[] argv, PrintStream out, PrintStream err) {
        if (argv.length == 0) { err.println(usage()); return 2; }
        String cmd = argv[0];
        List<String> rest = Arrays.asList(argv).subList(1, argv.length);
        if (cmd.equals("-h") || cmd.equals("--help") || cmd.equals("help")) {
            String sub = rest.isEmpty() ? null : COMMAND_USAGE.get(rest.get(0));
            out.println(sub != null ? sub : usage());
            return 0;
        }
        if (cmd.equals("--version") || cmd.equals("version")) {
            String v = Main.class.getPackage().getImplementationVersion();
            out.println("kiln " + (v == null ? "dev" : v));
            return 0;
        }
        String help = COMMAND_USAGE.get(cmd);
        if (help == null) { err.println("kiln: unknown command " + cmd + "\n\n" + usage()); return 2; }
        try {
            Args a = Args.parse(rest, valueOptions(cmd), cmd.equals("chat") ? Chat.FLAGS : Set.of());
            if (a.has("help")) { out.println(help); return 0; }
            KilnHub hub = KilnHub.standard();
            BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
            return switch (cmd) {
                case "pull" -> new Pull(hub, RepoIndex.standard(), out, err, in, terminal()).run(a);
                case "list" -> { noArgs(a); yield Listing.run(hub, out); }
                case "rm" -> Remove.run(a, hub, out);
                case "chat" -> Chat.run(a, hub, in, out, err);
                case "test" -> SmokeTest.run(a, hub, out);
                case "info" -> { noArgs(a); yield Info.run(hub, out); }
                default -> throw new IllegalStateException(cmd);
            };
        } catch (UsageException e) {
            err.println("kiln " + cmd + ": " + e.getMessage() + "\n\n" + help);
            return 2;
        } catch (Failure | HubException | IllegalArgumentException e) {   // IAE: a bad --revision, say
            err.println("kiln " + cmd + ": " + e.getMessage());
            return 1;
        } catch (LinkageError e) {   // the engine's native library is missing or wouldn't load
            Throwable root = e;
            while (root.getCause() != null) root = root.getCause();
            err.println("kiln " + cmd + ": couldn't load the native engine: " + root.getMessage()
                    + "\n  run kiln info to see where Kiln looks for " + Info.names());
            return 1;
        } catch (IOException e) {
            err.println("kiln " + cmd + ": " + e);
            return 1;
        } catch (IllegalStateException e) {   // an engine reporting a failure (load or generate)
            err.println("kiln " + cmd + ": " + e.getMessage());
            return 1;
        }
    }

    private static Set<String> valueOptions(String cmd) {
        return switch (cmd) {
            case "pull" -> Pull.OPTIONS;
            case "chat" -> Chat.OPTIONS;
            case "test" -> SmokeTest.OPTIONS;
            default -> Set.of();
        };
    }

    private static void noArgs(Args a) {
        if (!a.positional().isEmpty() || !a.options().isEmpty()) throw new UsageException("takes no arguments");
    }

    static String usage() {
        StringBuilder b = new StringBuilder();
        for (var m : ModelCatalog.all())
            b.append(String.format("  %-22s %-13s %s%n", m.id(), m.kind().name().toLowerCase(java.util.Locale.ROOT), m.title()));
        return USAGE.formatted(b.toString().stripTrailing());
    }

    /** True when the user is at a terminal (for prompts, progress redrawing and interactive chat). */
    static boolean terminal() {
        var c = System.console();
        return c != null && c.isTerminal();
    }
}
