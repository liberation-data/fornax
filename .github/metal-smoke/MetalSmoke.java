// Chats a few tokens from Qwen3 0.6B through Fornax's LlamaModel with the released Metal natives. The workflow then
// checks llama.cpp's log (stderr) for the model's layers offloaded to Metal, not left on the CPU.
import java.util.List;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;

public class MetalSmoke {
    public static void main(String[] args) {
        try (var model = new LlamaModel(args[0], -1, 2048)) {
            // The first call compiles ggml's Metal kernels (minutes on a fresh runner): out of the timing.
            model.chat(List.of(Message.user("Hi")), Options.DEFAULT.withMaxTokens(1).withTemperature(0));
            long t0 = System.nanoTime();
            var r = model.chat(List.of(Message.user("Say hello in five words.")), Options.DEFAULT.withMaxTokens(24).withTemperature(0));
            double s = (System.nanoTime() - t0) / 1e9;
            System.out.printf("reply: %s%n%d tokens in %.2f s (%.1f tokens/s)%n", r.text().strip(), r.generatedTokens(), s, r.generatedTokens() / s);
            if (r.generatedTokens() == 0) throw new IllegalStateException("no tokens generated");
        }
    }
}
