package org.liberationdata.fornax.llama;

import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Result;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * Tokens per second and short-call latency for one GGUF, in-process and then isolated (one model loaded at a time):
 * {@code ./gradlew :fornax-llama:isolationBench --args='model.gguf [tokens] [rounds]'}. Also prints how long the isolated load takes, which
 * is what a restart after a crash costs.
 */
public final class IsolationBenchmark {
    static final List<Message> COUNT = List.of(Message.user("Count from 1 to 500, one number per line, nothing else. /no_think"));
    static final List<Message> HI = List.of(Message.user("Say hi. /no_think"));
    static final Options GREEDY = Options.DEFAULT.withTemperature(0).withSeed(1);

    public static void main(String[] args) {
        String gguf = args[0];
        int tokens = args.length > 1 ? Integer.parseInt(args[1]) : 256;
        int rounds = args.length > 2 ? Integer.parseInt(args[2]) : 2;
        for (int i = 0; i < rounds; i++) {  // alternating, so GPU clocks and the like show as noise rather than bias
            run("in-process", () -> new LlamaModel(gguf, null, -1, 4096), tokens);
            run("isolated  ", () -> LlamaModel.isolated(gguf, null, -1, 4096), tokens);
        }
    }

    static void run(String what, Supplier<LlamaModel> load, int tokens) {
        long t0 = System.nanoTime();
        try (LlamaModel m = load.get()) {
            double loadMs = (System.nanoTime() - t0) / 1e6;
            m.chat(HI, GREEDY.withMaxTokens(8));  // warm-up
            double[] tps = new double[3];
            double[] streamedTps = new double[3];
            for (int i = 0; i < 3; i++) {
                long s = System.nanoTime();
                Result r = m.chat(COUNT, GREEDY.withMaxTokens(tokens));
                tps[i] = r.generatedTokens() / ((System.nanoTime() - s) / 1e9);
                s = System.nanoTime();
                r = m.chat(COUNT, GREEDY.withMaxTokens(tokens), piece -> {});
                streamedTps[i] = r.generatedTokens() / ((System.nanoTime() - s) / 1e9);
            }
            long[] shortCall = new long[20];
            for (int i = 0; i < shortCall.length; i++) {
                long s = System.nanoTime();
                m.chat(HI, GREEDY.withMaxTokens(1));
                shortCall[i] = System.nanoTime() - s;
            }
            Arrays.sort(tps);
            Arrays.sort(streamedTps);
            Arrays.sort(shortCall);
            System.out.printf("%s  load %7.0f ms   %6.1f tok/s   streamed %6.1f tok/s   1-token chat %6.2f ms (median)%n",
                    what, loadMs, tps[1], streamedTps[1], shortCall[shortCall.length / 2] / 1e6);
        }
    }

    private IsolationBenchmark() {}
}
