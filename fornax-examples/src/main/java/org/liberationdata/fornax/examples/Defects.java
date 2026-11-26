package org.liberationdata.fornax.examples;

import org.liberationdata.fornax.llama.Json;
import org.liberationdata.fornax.llama.LlamaModel;
import org.liberationdata.fornax.llama.LlamaModel.Image;
import org.liberationdata.fornax.llama.LlamaModel.Message;
import org.liberationdata.fornax.llama.LlamaModel.Options;
import org.liberationdata.fornax.llama.LlamaModel.Weights;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;
import java.util.Map;

/**
 * Defect spotting with grounding: draws a tray of parts (a grid of discs) with one dented and one missing, asks
 * Qwen3-VL for the defects' bounding boxes as JSON, and draws its boxes on the picture. Usage: Defects [-Dmodel=qwen3-vl-4b]
 * Writes out/defects.png (the input) and out/defects-found.png (with the model's boxes, in red, and the truth, in green).
 */
public class Defects {
    static final int COLS = 6, ROWS = 4, CELL = 120, R = 42, W = COLS * CELL, H = ROWS * CELL;
    static final int DENTED_COL = 4, DENTED_ROW = 1, MISSING_COL = 1, MISSING_ROW = 2;

    static final String SCHEMA = """
        {"type": "object", "properties": {"defects": {"type": "array", "items": {"type": "object",
           "properties": {"what": {"type": "string"}, "kind": {"type": "string", "enum": ["dented", "missing"]},
                          "bbox_2d": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}},
           "required": ["what", "kind", "bbox_2d"]}}},
         "required": ["defects"]}""";

    public static void main(String[] args) throws Exception {
        new File("out").mkdirs();
        BufferedImage tray = draw();
        ImageIO.write(tray, "png", new File("out/defects.png"));
        String m = System.getProperty("model", "qwen3-vl-4b");
        long t0 = System.nanoTime();
        try (var llm = new LlamaModel(Weights.named(m))) {
            System.out.printf("[fornax] loaded %s in %.1fs%n", m, (System.nanoTime() - t0) / 1e9);
            long g0 = System.nanoTime();
            String json = llm.extract(List.of(Message.user("""
                    This is a tray of identical round parts in a grid. Find every defect: a part that is dented \
                    (not a full circle) or a missing part (an empty place in the grid). Give each one's bounding box \
                    as bbox_2d [x1, y1, x2, y2]; first say in a few words what is wrong there.""", Image.of(tray))),
                    SCHEMA, Options.DEFAULT.withTemperature(0).withMaxTokens(512));
            System.out.println(json);
            System.out.printf("[fornax] answered in %.1fs%n", (System.nanoTime() - g0) / 1e9);

            // Qwen3-VL gives boxes on a 0..1000 grid over the image.
            Graphics2D g = tray.createGraphics();
            g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setStroke(new BasicStroke(3));
            g.setColor(new Color(0, 160, 0));
            for (int[] cell : new int[][] {{DENTED_COL, DENTED_ROW}, {MISSING_COL, MISSING_ROW}})
                g.drawRect(cell[0] * CELL + 6, cell[1] * CELL + 6, CELL - 12, CELL - 12);
            g.setColor(Color.RED);
            for (Object d : (List<?>) Json.get(Json.parse(json), "defects")) {
                List<?> b = (List<?>) ((Map<?, ?>) d).get("bbox_2d");
                int x1 = px(b.get(0), W), y1 = px(b.get(1), H), x2 = px(b.get(2), W), y2 = px(b.get(3), H);
                g.drawRect(x1, y1, x2 - x1, y2 - y1);
                g.drawString(String.valueOf(((Map<?, ?>) d).get("kind")), x1 + 4, y1 - 4);
                System.out.printf("[fornax] %s at grid column %d, row %d%n", ((Map<?, ?>) d).get("kind"),
                        (x1 + x2) / 2 / CELL, (y1 + y2) / 2 / CELL);
            }
            g.dispose();
            ImageIO.write(tray, "png", new File("out/defects-found.png"));
            System.out.printf("[fornax] truth: dented at column %d, row %d; missing at column %d, row %d -> out/defects-found.png%n",
                    DENTED_COL, DENTED_ROW, MISSING_COL, MISSING_ROW);
        }
    }

    static int px(Object v, int size) { return (int) Math.round(((Number) v).doubleValue() * size / 1000.0); }

    /** Grey discs on a dark tray; one has a bite out of it, one place is empty. */
    static BufferedImage draw() {
        var img = new BufferedImage(W, H, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(new Color(0x2B2F36));
        g.fillRect(0, 0, W, H);
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                if (c == MISSING_COL && r == MISSING_ROW) continue;
                int cx = c * CELL + CELL / 2, cy = r * CELL + CELL / 2;
                g.setColor(new Color(0xB8BEC6));
                g.fillOval(cx - R, cy - R, 2 * R, 2 * R);
                g.setColor(new Color(0x8A9099));
                g.fillOval(cx - 10, cy - 10, 20, 20);
                if (c == DENTED_COL && r == DENTED_ROW) {  // a bite out of the upper right
                    g.setColor(new Color(0x2B2F36));
                    g.fillOval(cx + R / 3, cy - R - R / 3, R, R);
                }
            }
        }
        g.dispose();
        return img;
    }
}
