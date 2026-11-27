package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.hub.RepoIndex;
import org.liberationdata.fornax.cli.Args.UsageException;
import org.liberationdata.fornax.hub.HubException;
import org.liberationdata.fornax.hub.FornaxHub;
import org.liberationdata.fornax.hub.ModelCatalog;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** The {@code fornax} command. */
public final class Main {
    private Main() {}

    /** A failure with a message meant for people: printed without a stack trace, exit code 1. */
    static final class Failure extends RuntimeException {
        private static final long serialVersionUID = 1L;
        Failure(String message) { super(message); }
    }

    static final String USAGE = """
            usage: fornax <command> [args]

              pull <model> [file...]   download from Hugging Face into the HF cache
              list                     the catalogue's models, what's downloaded, other GGUFs in the cache
              rm <alias>               delete a model's files from the cache
              chat <model> ["prompt"]  chat with a GGUF language model (interactive without a prompt)
              image <model> "prompt"   draw an image (flux2-klein, ...)
              song <model> "about"     make a song (ace-step-turbo, ...)
              speak ["text"]           text to speech (kokoro)
              transcribe <audio.wav>   speech to text (whisper-base.en)
              see <model> <image>      ask a vision model about an image (qwen3-vl-4b, ...)
              server [-m model]...     serve models over an OpenAI-compatible HTTP API
              test <model>             load, generate a little, report speed
              info                     native libraries, Java and cache locations

            <model> is an alias, org/name[:TAG|:file.gguf], or a .gguf path. Aliases (Fornax's model catalogue):
            %s
            fornax <command> --help for more. Set FORNAX_VERBOSE=1 to see the engines' logs.""";

    private static final Map<String, String> COMMAND_USAGE = Map.ofEntries(
            Map.entry("pull", Pull.USAGE), Map.entry("list", Listing.USAGE), Map.entry("rm", Remove.USAGE),
            Map.entry("chat", Chat.USAGE), Map.entry("image", Image.USAGE), Map.entry("song", Song.USAGE),
            Map.entry("speak", Speak.USAGE), Map.entry("transcribe", Transcribe.USAGE), Map.entry("see", See.USAGE),
            Map.entry("server", Server.USAGE), Map.entry("test", SmokeTest.USAGE), Map.entry("info", Info.USAGE));

    public static void main(String[] argv) {
        Server.httpLimits();   // before any HTTP server class loads
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
            out.println("fornax " + (v == null ? "dev" : v));
            return 0;
        }
        String help = COMMAND_USAGE.get(cmd);
        if (help == null) { err.println("fornax: unknown command " + cmd + "\n\n" + usage()); return 2; }
        try {
            Args a = Args.parse(rest, valueOptions(cmd), flags(cmd), cmd.equals("server") ? Server.SHORT : Map.of());
            if (a.has("help")) { out.println(help); return 0; }
            FornaxHub hub = FornaxHub.standard();
            BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
            return switch (cmd) {
                case "pull" -> new Pull(hub, RepoIndex.standard(), out, err, in, terminal()).run(a);
                case "list" -> { noArgs(a); yield Listing.run(hub, out); }
                case "rm" -> Remove.run(a, hub, out);
                case "chat" -> Chat.run(a, hub, in, out, err);
                case "image" -> Image.run(a, hub, out, err);
                case "song" -> Song.run(a, hub, in, out, err);
                case "speak" -> Speak.run(a, hub, in, out, err);
                case "transcribe" -> Transcribe.run(a, hub, out, err);
                case "see" -> See.run(a, hub, out, err);
                case "server" -> Server.run(a, hub, out, err);
                case "test" -> SmokeTest.run(a, hub, out, err);
                case "info" -> { noArgs(a); yield Info.run(hub, out); }
                default -> throw new IllegalStateException(cmd);
            };
        } catch (UsageException e) {
            err.println("fornax " + cmd + ": " + e.getMessage() + "\n\n" + help);
            return 2;
        } catch (Failure | HubException | IllegalArgumentException e) {   // IAE: a bad --revision, say
            err.println("fornax " + cmd + ": " + e.getMessage());
            return 1;
        } catch (LinkageError e) {   // the engine's native library is missing or wouldn't load
            Throwable root = e;
            while (root.getCause() != null) root = root.getCause();
            err.println("fornax " + cmd + ": couldn't load the native engine: " + root.getMessage()
                    + "\n  run fornax info to see where Fornax looks for " + Info.names());
            return 1;
        } catch (IOException e) {
            err.println("fornax " + cmd + ": " + describe(e));
            return 1;
        } catch (IllegalStateException e) {   // an engine reporting a failure (load or generate)
            err.println("fornax " + cmd + ": " + e.getMessage());
            return 1;
        }
    }

    /** An I/O failure in words: no connection to the Hub says so, rather than a bare java.net.ConnectException. */
    static String describe(IOException e) {
        if (!(e instanceof java.net.ConnectException || e instanceof java.net.UnknownHostException
                || e instanceof java.net.http.HttpTimeoutException)) return e.toString();
        Throwable t = e;
        while (t.getMessage() == null && t.getCause() != null) t = t.getCause();
        return "couldn't reach " + RepoIndex.endpoint() + " (" + (t.getMessage() != null ? t.getMessage() : t.getClass().getSimpleName())
                + "). Check the network connection; with HF_HUB_OFFLINE=1, fornax uses only what is already downloaded.";
    }

    private static Set<String> valueOptions(String cmd) {
        return switch (cmd) {
            case "pull" -> Pull.OPTIONS;
            case "chat" -> Chat.OPTIONS;
            case "image" -> Image.OPTIONS;
            case "song" -> Song.OPTIONS;
            case "speak" -> Speak.OPTIONS;
            case "transcribe" -> Transcribe.OPTIONS;
            case "see" -> See.OPTIONS;
            case "server" -> Server.OPTIONS;
            case "test" -> SmokeTest.OPTIONS;
            default -> Set.of();
        };
    }

    private static Set<String> flags(String cmd) {
        return switch (cmd) {
            case "chat" -> Chat.FLAGS;
            case "speak" -> Speak.FLAGS;
            case "transcribe" -> Transcribe.FLAGS;
            case "see" -> See.FLAGS;
            case "server" -> Server.FLAGS;
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
