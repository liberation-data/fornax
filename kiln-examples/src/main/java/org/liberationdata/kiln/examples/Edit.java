package org.liberationdata.kiln.examples;

import org.liberationdata.kiln.sd.DiffusionModel;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Edit an image with stable-diffusion.cpp: img2img, inpainting, or an instruction edit, optionally with LoRAs.
 * Usage: Edit <in.png> <out.png> "<prompt>"
 *   [-Dmodel=klein|qwen-edit] [-Dmask=mask.png] [-Dstrength=0.75] [-Dlora=path[:weight][,path[:weight]...]]
 *   [-Dref=true|false] [-Dsteps=N -Dcfg=C -Dwidth=W -Dheight=H -Dseed=S -Dshift=F -Dsampler=M -Dscheduler=S]
 *
 * With klein (the default) in.png is the init image: img2img at -Dstrength, or inpainting where -Dmask is white.
 * With qwen-edit it is a reference image for an instruction ("make the sign say KILN"), and it is the init image as
 * well only when -Dmask or -Dstrength is given. -Dref=true also passes it to klein as a reference.
 * The output size defaults to the input's, rounded down to a multiple of 16.
 */
public class Edit {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("usage: Edit <in.png> <out.png> \"<prompt>\" [-Pmodel=klein|qwen-edit] [-Pmask=m.png] [-Pstrength=S] [-Plora=path:weight]");
            System.exit(2);
        }
        BufferedImage in = ImageIO.read(new File(args[0]));
        File out = new File(args[1]);
        String prompt = args[2];
        String model = System.getProperty("model", "klein");
        Image.Preset preset = Image.preset(model);
        boolean instruct = model.equals("qwen-edit");
        String mask = System.getProperty("mask"), strength = System.getProperty("strength");
        boolean ref = Boolean.parseBoolean(System.getProperty("ref", String.valueOf(instruct)));
        boolean init = !instruct || mask != null || strength != null;

        int w = Integer.getInteger("width", in.getWidth() / 16 * 16);
        int h = Integer.getInteger("height", in.getHeight() / 16 * 16);
        int steps = Integer.getInteger("steps", preset.steps());
        double cfg = Double.parseDouble(System.getProperty("cfg", String.valueOf(preset.cfg())));
        Map<String, Object> extra = new HashMap<>();
        if (System.getProperty("shift") != null) extra.put("sample.flow_shift", Double.parseDouble(System.getProperty("shift")));
        if (System.getProperty("sampler") != null) extra.put("sample.method", System.getProperty("sampler"));
        if (System.getProperty("scheduler") != null) extra.put("sample.scheduler", System.getProperty("scheduler"));

        var req = new DiffusionModel.Request(prompt, w, h, steps, cfg, Long.getLong("seed", 7))
                .withExtra(extra)
                .withLoras(loras(System.getProperty("lora")));
        if (ref) req = req.withRefImages(List.of(in));
        if (init) req = strength == null ? req.withInitImage(in) : req.withInitImage(in, Double.parseDouble(strength));
        if (mask != null) req = req.withMaskImage(ImageIO.read(new File(mask)));

        DiffusionModel.onProgress((s, n, t) -> System.out.printf("  step %d/%d (%.2fs/it)%n", s, n, t));
        long t0 = System.nanoTime();
        try (var m = new DiffusionModel(preset.config().build())) {
            System.out.printf("[kiln] loaded %s in %.1fs%n", m.modelVersion(), (System.nanoTime() - t0) / 1e9);
            long g0 = System.nanoTime();
            ImageIO.write(m.generate(req).get(0), "png", out);
            System.out.printf("[kiln] %s%s%s%s: %dx%d, %d steps in %.1fs -> %s%n", model, init ? " img2img" : "",
                    mask != null ? " inpaint" : "", ref ? " ref" : "", w, h, steps, (System.nanoTime() - g0) / 1e9, out);
        }
    }

    /** "a.safetensors:0.8,b.safetensors" (the weight defaults to 1; the last colon separates it). */
    static List<DiffusionModel.Lora> loras(String spec) {
        List<DiffusionModel.Lora> loras = new ArrayList<>();
        if (spec == null || spec.isBlank()) return loras;
        for (String item : spec.split(",")) {
            int colon = item.lastIndexOf(':');
            boolean weighted = colon > 0 && item.substring(colon + 1).matches("-?[0-9.]+");
            loras.add(weighted
                    ? new DiffusionModel.Lora(Path.of(item.substring(0, colon)), Float.parseFloat(item.substring(colon + 1)))
                    : new DiffusionModel.Lora(Path.of(item)));
        }
        return loras;
    }
}
