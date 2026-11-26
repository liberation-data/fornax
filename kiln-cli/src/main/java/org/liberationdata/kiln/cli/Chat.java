package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.cli.Args.UsageException;
import org.liberationdata.kiln.hub.KilnHub;
import org.liberationdata.kiln.llama.LlamaModel;
import org.liberationdata.kiln.llama.LlamaModel.Message;
import org.liberationdata.kiln.llama.LlamaModel.Options;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/** {@code kiln chat}: one-shot or interactive streaming chat with a GGUF model. */
final class Chat {
    static final Set<String> OPTIONS = Set.of("system", "max-tokens", "temperature", "seed", "ctx", "gpu-layers");
    static final Set<String> FLAGS = Set.of("think");

    static final String USAGE = """
            usage: kiln chat <model> ["prompt"] [options]

            With a prompt, prints one reply. Without one, chats interactively (/reset clears the history,
            /exit or Ctrl-D quits); with stdin redirected, the whole of stdin is the prompt.

            <model> is a .gguf path, an alias (qwen3-4b), org/name:file.gguf, org/name:TAG (a quant such as
            Q4_K_M) or org/name, looked up in the Hugging Face cache (see kiln list).

            A reasoning model's <think>...</think> blocks (Qwen3's, for one) are hidden; --think shows them.

              --system TEXT        system prompt
              --max-tokens N       per reply (default 1024; 0 = until the model stops or the context fills)
              --temperature T      default 0.7; 0 is greedy
              --seed N             default random
              --ctx N              context size in tokens (default 8192; 0 = the model's own)
              --gpu-layers N       layers to offload (default -1 = all)
              --think              show the model's reasoning (<think> blocks)""";

    static Options options(Args a) {
        return new Options(a.intOption("max-tokens", 1024), a.floatOption("temperature", 0.7f),
                Options.DEFAULT.topP(), Options.DEFAULT.minP(), a.intOption("seed", -1));
    }

    static int run(Args a, KilnHub hub, BufferedReader in, PrintStream out, PrintStream err) throws IOException {
        String spec = a.positional(0);
        if (spec == null) throw new UsageException("which model? (see kiln list)");
        if (!(Models.resolve(spec, hub) instanceof Models.Llm llm))
            throw new Main.Failure(spec + " isn't a language model; try kiln test " + spec);
        String prompt = String.join(" ", a.rest(1));
        boolean interactive = prompt.isBlank() && Main.terminal();
        if (prompt.isBlank() && !interactive) {
            prompt = readAll(in);
            if (prompt.isBlank()) throw new UsageException("no prompt (give one, or pipe it on stdin)");
        }
        Options o = options(a);
        String system = a.option("system", null);
        boolean think = a.has("think");

        err.println("[kiln] loading " + llm.path());
        long t0 = System.nanoTime();
        try (var model = new LlamaModel(llm.path().toString(), a.intOption("gpu-layers", -1), a.intOption("ctx", 8192))) {
            err.printf(Locale.ROOT, "[kiln] loaded in %s%n", Fmt.seconds((System.nanoTime() - t0) / 1e9));
            List<Message> history = new ArrayList<>();
            if (system != null) history.add(Message.system(system));
            if (!interactive) {
                history.add(Message.user(prompt));
                reply(model, history, o, think, false, out, err);
                return 0;
            }
            err.println("[kiln] chatting with " + llm.name() + ". /reset clears the history, /exit quits.");
            boolean opensInside = false;   // the model's replies start inside a think block (its <think> in the prompt)
            while (true) {
                out.print("> ");
                out.flush();
                String line = in.readLine();
                if (line == null) { out.println(); return 0; }
                line = line.strip();
                if (line.isEmpty()) continue;
                if (line.equals("/exit") || line.equals("/quit")) return 0;
                if (line.equals("/reset")) {
                    history.subList(system != null ? 1 : 0, history.size()).clear();
                    opensInside = false;   // a guess from the replies: a stray </think> mustn't hide every later one
                    err.println("[kiln] history cleared");
                    continue;
                }
                history.add(Message.user(line));
                var r = reply(model, history, o, think, opensInside, out, err);
                opensInside |= r.openedInside();
                history.add(Message.assistant(r.reply()));
            }
        }
    }

    /**
     * Streams one reply to {@code out}, without its think blocks unless {@code think}, and its token rate to
     * {@code err}. Returns the filter, which has the reply without think blocks, for the history.
     */
    private static ThinkFilter reply(LlamaModel model, List<Message> history, Options o, boolean think,
                                     boolean startsInside, PrintStream out, PrintStream err) {
        long g0 = System.nanoTime();
        var shown = new ThinkFilter(piece -> { out.print(piece); out.flush(); }, think, startsInside);
        var r = model.chat(history, o, shown);
        shown.finish();
        double s = (System.nanoTime() - g0) / 1e9;
        out.println();
        err.printf(Locale.ROOT, "[kiln] %d prompt + %d generated tokens in %s (%.1f tok/s)%n",
                r.promptTokens(), r.generatedTokens(), Fmt.seconds(s), r.generatedTokens() / s);
        if (!think && shown.reply().isEmpty() && !r.text().isBlank())
            err.println("[kiln] the reply was all reasoning, which is hidden: raise --max-tokens, or pass --think to see it");
        return shown;
    }

    private static String readAll(BufferedReader in) throws IOException {
        StringBuilder b = new StringBuilder();
        for (String line; (line = in.readLine()) != null; ) b.append(line).append('\n');
        return b.toString().strip();
    }
}
