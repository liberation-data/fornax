package org.liberationdata.fornax.examples;

import org.liberationdata.fornax.acestep.AceStepModel;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.sd.DiffusionModel;

import javax.imageio.ImageIO;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Three engines, one JVM: an LLM writes a song, ACE-Step sings it, FLUX.2 klein paints the cover.
 * Usage: Single [out-dir] ["what the song is about"]
 */
public class Single {
    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args.length > 0 ? args[0] : "single");
        String idea = args.length > 1 ? args[1] : "a robot with a honky-tonk heart who falls for the jukebox";
        Files.createDirectories(dir);
        long t0 = System.nanoTime();

        String title, lyrics, cover;
        try (var llm = new LlamaModel("qwen3-4b")) {   // each model by its catalogue name, downloaded on first use
            String song = llm.chat(List.of(Message.system("You are a Nashville songwriter. Be concise."),
                    Message.user("Write a short country song about " + idea + ". Reply in exactly this format:\n"
                            + "TITLE: <title>\nCOVER: <one-sentence description of the single's cover art>\n"
                            + "[Verse 1]\n<4 lines>\n[Chorus]\n<4 lines>\n[Verse 2]\n<4 lines>\n[Chorus]\n<4 lines>")),
                    Options.DEFAULT.withMaxTokens(400).withSeed(5)).text();
            title = line(song, "TITLE:");
            cover = line(song, "COVER:");
            lyrics = song.substring(song.indexOf('[')).trim();
        }
        System.out.printf("[fornax] llama.cpp wrote \"%s\" (%.1fs)%n%s%n", title, secs(t0), lyrics);

        long s0 = System.nanoTime();
        try (var singer = new AceStepModel("ace-step-turbo", false)) {   // the DiT alone: caption and lyrics drive it
            String request = "{\"caption\": \"modern country, male vocal with a southern twang, telecaster, pedal steel, "
                    + "driving drums, big singalong chorus\", \"lyrics\": " + json(lyrics) + ", \"bpm\": 100, "
                    + "\"duration\": 75, \"keyscale\": \"A major\", \"timesignature\": \"4\", \"vocal_language\": \"en\", "
                    + "\"seed\": 5, \"inference_steps\": 8, \"guidance_scale\": 1.0, \"shift\": 3.0}";
            var r = singer.generate(request);
            r.audio().writeWav(dir.resolve("single.wav").toFile());
            System.out.printf("[fornax] acestep.cpp sang %.0fs of audio (%.1fs)%n", r.audio().seconds(), secs(s0));
        }

        long i0 = System.nanoTime();
        var cfg = DiffusionModel.Config.builder().model("flux2-klein").build();
        int steps = Integer.getInteger("steps", 12), size = Integer.getInteger("width", 768);
        try (var painter = new DiffusionModel(cfg)) {
            String prompt = "Country music single cover art. " + cover + " The title \"" + title + "\" in vintage lettering.";
            var img = painter.generate(new DiffusionModel.Request(prompt, size, size, steps, 4.0, 5, List.of(), Map.of())).get(0);
            ImageIO.write(img, "png", dir.resolve("cover.png").toFile());
            System.out.printf("[fornax] stable-diffusion.cpp painted the cover (%.1fs)%n", secs(i0));
        }
        Files.writeString(dir.resolve("lyrics.txt"), title + "\n\n" + lyrics + "\n");
        System.out.printf("[fornax] done in %.1fs -> %s/{single.wav,cover.png,lyrics.txt}%n", secs(t0), dir);
    }

    private static String line(String text, String prefix) {
        return text.lines().filter(l -> l.startsWith(prefix)).findFirst().map(l -> l.substring(prefix.length()).trim()).orElse("Untitled");
    }

    private static double secs(long t) { return (System.nanoTime() - t) / 1e9; }

    private static String json(String s) {
        StringBuilder b = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"' -> b.append("\\\"");
                case '\\' -> b.append("\\\\");
                case '\n' -> b.append("\\n");
                case '\r', '\t' -> b.append(' ');
                default -> { if (c < 0x20) b.append(' '); else b.append(c); }
            }
        }
        return b.append('"').toString();
    }
}
