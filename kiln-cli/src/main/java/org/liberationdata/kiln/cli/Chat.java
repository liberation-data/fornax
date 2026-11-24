package org.liberationdata.kiln.cli;

import org.liberationdata.kiln.cli.Args.UsageException;
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

    static final String USAGE = """
            usage: kiln chat <model> ["prompt"] [options]

            With a prompt, prints one reply. Without one, chats interactively (/reset clears the history,
            /exit or Ctrl-D quits); with stdin redirected, the whole of stdin is the prompt.

            <model> is a .gguf path, an alias (qwen3-4b), org/name:file.gguf, org/name:TAG (a quant such as
            Q4_K_M) or org/name, looked up in the Hugging Face cache (see kiln list).

              --system TEXT        system prompt
              --max-tokens N       per reply (default 1024; 0 = until the model stops or the context fills)
              --temperature T      default 0.7; 0 is greedy
              --seed N             default random
              --ctx N              context size in tokens (default 8192; 0 = the model's own)
              --gpu-layers N       layers to offload (default -1 = all)""";

    static Options options(Args a) {
        return new Options(a.intOption("max-tokens", 1024), a.floatOption("temperature", 0.7f),
                Options.DEFAULT.topP(), Options.DEFAULT.minP(), a.intOption("seed", -1));
    }

    static int run(Args a, HubCache cache, BufferedReader in, PrintStream out, PrintStream err) throws IOException {
        String spec = a.positional(0);
        if (spec == null) throw new UsageException("which model? (see kiln list)");
        if (!(Models.resolve(spec, cache) instanceof Models.Llm llm))
            throw new Main.Failure(spec + " isn't a language model; try kiln test " + spec);
        String prompt = String.join(" ", a.rest(1));
        boolean interactive = prompt.isBlank() && Main.terminal();
        if (prompt.isBlank() && !interactive) {
            prompt = readAll(in);
            if (prompt.isBlank()) throw new UsageException("no prompt (give one, or pipe it on stdin)");
        }
        Options o = options(a);
        String system = a.option("system", null);

        err.println("[kiln] loading " + llm.path());
        long t0 = System.nanoTime();
        try (var model = new LlamaModel(llm.path().toString(), a.intOption("gpu-layers", -1), a.intOption("ctx", 8192))) {
            err.printf(Locale.ROOT, "[kiln] loaded in %s%n", Fmt.seconds((System.nanoTime() - t0) / 1e9));
            List<Message> history = new ArrayList<>();
            if (system != null) history.add(Message.system(system));
            if (!interactive) {
                history.add(Message.user(prompt));
                reply(model, history, o, out, err);
                return 0;
            }
            err.println("[kiln] chatting with " + llm.name() + ". /reset clears the history, /exit quits.");
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
                    err.println("[kiln] history cleared");
                    continue;
                }
                history.add(Message.user(line));
                history.add(Message.assistant(reply(model, history, o, out, err)));
            }
        }
    }

    /** Streams one reply to {@code out} and its token rate to {@code err}. */
    private static String reply(LlamaModel model, List<Message> history, Options o, PrintStream out, PrintStream err) {
        long g0 = System.nanoTime();
        var r = model.chat(history, o, piece -> { out.print(piece); out.flush(); });
        double s = (System.nanoTime() - g0) / 1e9;
        out.println();
        err.printf(Locale.ROOT, "[kiln] %d prompt + %d generated tokens in %s (%.1f tok/s)%n",
                r.promptTokens(), r.generatedTokens(), Fmt.seconds(s), r.generatedTokens() / s);
        return r.text();
    }

    private static String readAll(BufferedReader in) throws IOException {
        StringBuilder b = new StringBuilder();
        for (String line; (line = in.readLine()) != null; ) b.append(line).append('\n');
        return b.toString().strip();
    }
}
