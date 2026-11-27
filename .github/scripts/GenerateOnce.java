// Generates text with a real model on the CPU, in process and in an isolated worker JVM, and checks the two agree:
// a check that a platform's natives run a model end to end, not just load (run by windows.yml, with
// -Dfornax.library.path). The model is ggml-org's tinyllamas stories15M (19 MB), so it takes seconds on any runner.
//   java --enable-native-access=ALL-UNNAMED -Dfornax.library.path=<dir> -cp <fornax-core.jar><sep><fornax-llama.jar> \
//        .github/scripts/GenerateOnce.java <stories15M-q4_0.gguf>
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Mode;
import org.liberationdata.fornax.llama.LlamaModel.Options;

public class GenerateOnce {
    public static void main(String[] args) {
        String gguf = args[0];
        Options greedy = Options.DEFAULT.withTemperature(0).withMaxTokens(24);
        String prompt = "Once upon a time";

        LlamaModel.Result inProcess;
        try (var m = new LlamaModel(gguf, 0, 256)) {   // gpuLayers 0: the CPU
            inProcess = m.complete(prompt, greedy, null);
        }
        System.out.println("in process: " + inProcess.generatedTokens() + " tokens: " + prompt + inProcess.text());
        check(inProcess.generatedTokens() > 0 && !inProcess.text().isBlank(), "the model generated nothing");

        LlamaModel.Result isolated;
        try (var m = LlamaModel.isolated(gguf, 0, 256, Mode.CHAT)) {
            isolated = m.complete(prompt, greedy, null);
        }
        System.out.println("isolated:   " + isolated.generatedTokens() + " tokens: " + prompt + isolated.text());
        check(isolated.text().equals(inProcess.text()), "greedy text differs between in-process and the isolated worker");
    }

    static void check(boolean ok, String failure) {
        if (!ok) { System.err.println("FAILED: " + failure); System.exit(1); }
    }
}
