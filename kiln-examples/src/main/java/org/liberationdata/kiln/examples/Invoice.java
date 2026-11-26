package org.liberationdata.kiln.examples;

import org.liberationdata.kiln.llama.Json;
import org.liberationdata.kiln.llama.LlamaModel;
import org.liberationdata.kiln.llama.LlamaModel.Image;
import org.liberationdata.kiln.llama.LlamaModel.Message;
import org.liberationdata.kiln.llama.LlamaModel.Options;
import org.liberationdata.kiln.llama.LlamaModel.Weights;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/**
 * Structured extraction from a picture of an invoice: vendor, date, total and line items as JSON, constrained to a
 * JSON schema. Usage: Invoice [invoice.png] [-Dmodel=qwen3-vl-4b|paddleocr-vl|medgemma-4b]
 * With no image, draws a synthetic invoice (made-up vendor and items) to out/invoice.png and reads that.
 * -Dmodel=paddleocr-vl reads the text with PaddleOCR-VL, then fills the schema with a text model (-Dllm=qwen3-4b).
 */
public class Invoice {
    static final String SCHEMA = """
        {"type": "object",
         "properties": {
           "vendor": {"type": "string"},
           "invoice_number": {"type": "string"},
           "date": {"type": "string", "description": "YYYY-MM-DD"},
           "line_items": {"type": "array", "items": {"type": "object",
              "properties": {"description": {"type": "string"}, "quantity": {"type": "number"},
                             "unit_price": {"type": "number"}, "amount": {"type": "number"}},
              "required": ["description", "quantity", "unit_price", "amount"]}},
           "subtotal": {"type": "number"},
           "tax": {"type": "number"},
           "total": {"type": "number"}},
         "required": ["vendor", "invoice_number", "date", "line_items", "subtotal", "tax", "total"]}""";

    public static void main(String[] args) throws Exception {
        Path image;
        if (args.length > 0) {
            image = Path.of(args[0]);
        } else {
            image = Path.of("out/invoice.png");
            image.toFile().getParentFile().mkdirs();
            ImageIO.write(draw(), "png", image.toFile());
            System.out.println("[kiln] drew a synthetic invoice: " + image);
        }
        Weights w = Weights.named(System.getProperty("model", "qwen3-vl-4b"));
        Options o = Options.DEFAULT.withTemperature(0).withMaxTokens(1024);
        String ask = "Extract this invoice's fields. Dates as YYYY-MM-DD, amounts as plain numbers.";
        long t0 = System.nanoTime();
        String json;
        if (w == Weights.PADDLEOCR_VL) {
            // PaddleOCR-VL reads documents but takes task prompts ("OCR:"), not instructions: read the text with it,
            // then have a text model fill in the schema.
            String text;
            try (var ocr = new LlamaModel(w)) {
                text = ocr.chat(List.of(Message.user("OCR:", Image.read(image))), o).text();
            }
            System.out.printf("[kiln] PaddleOCR-VL read %d characters in %.1fs%n", text.length(), (System.nanoTime() - t0) / 1e9);
            Weights llm = Weights.named(System.getProperty("llm", "qwen3-4b"));
            try (var model = new LlamaModel(llm)) {
                json = model.extract(List.of(Message.user(ask + "\n\n" + text)), SCHEMA, o);
            }
        } else {
            try (var llm = new LlamaModel(w)) {
                System.out.printf("[kiln] loaded %s in %.1fs%n", w, (System.nanoTime() - t0) / 1e9);
                t0 = System.nanoTime();
                json = llm.extract(List.of(Message.user(ask, Image.read(image))), SCHEMA, o);
            }
        }
        System.out.println(json);
        var doc = (Map<?, ?>) Json.parse(json);
        System.out.printf("[kiln] %s, %s: %d line items, total %s, in %.1fs%n", doc.get("vendor"), doc.get("date"),
                ((List<?>) doc.get("line_items")).size(), doc.get("total"), (System.nanoTime() - t0) / 1e9);
    }

    /** A made-up invoice: fictional vendor, customer and items. */
    static BufferedImage draw() {
        int w = 1000, h = 1300;
        var img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, w, h);
        g.setColor(new Color(0x1F3A5F));
        g.fillRect(0, 0, w, 140);
        g.setColor(Color.WHITE);
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 44));
        g.drawString("Bluegill Pottery Supply Co.", 50, 80);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 20));
        g.drawString("12 Kiln Lane, Clayton, TN 37000  ·  (555) 010-4477", 50, 118);

        g.setColor(Color.BLACK);
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 36));
        g.drawString("INVOICE", 50, 220);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 22));
        g.drawString("Invoice no.: INV-20417", 600, 200);
        g.drawString("Date: 14 March 2026", 600, 232);
        g.drawString("Due: 13 April 2026", 600, 264);
        g.drawString("Bill to:", 50, 280);
        g.drawString("Hollow Creek Ceramics Studio", 50, 312);
        g.drawString("88 Glaze Street, Riverbend, KY 40000", 50, 344);

        int y = 420;
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 22));
        g.drawString("Description", 50, y);
        g.drawString("Qty", 560, y);
        g.drawString("Unit price", 660, y);
        g.drawString("Amount", 840, y);
        g.setStroke(new BasicStroke(2));
        g.drawLine(50, y + 14, 950, y + 14);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 22));
        String[][] items = {
                {"Stoneware clay, 25 lb bag", "8", "18.50", "148.00"},
                {"Celadon glaze, 1 pint", "3", "24.00", "72.00"},
                {"Kiln shelf, 16 in. round", "2", "39.95", "79.90"},
                {"Pyrometric cones, box of 50", "1", "15.25", "15.25"},
        };
        for (String[] it : items) {
            y += 50;
            g.drawString(it[0], 50, y);
            g.drawString(it[1], 570, y);
            g.drawString("$" + it[2], 660, y);
            g.drawString("$" + it[3], 840, y);
        }
        y += 30;
        g.drawLine(560, y, 950, y);
        y += 40;
        g.drawString("Subtotal", 660, y);
        g.drawString("$315.15", 840, y);
        y += 40;
        g.drawString("Sales tax (9.25%)", 600, y);
        g.drawString("$29.15", 840, y);
        y += 44;
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 26));
        g.drawString("Total due", 660, y);
        g.drawString("$344.30", 840, y);

        g.setFont(new Font(Font.SANS_SERIF, Font.ITALIC, 18));
        g.setColor(Color.DARK_GRAY);
        g.drawString("Thank you for your business. Payment within 30 days. (Sample invoice: all names are fictional.)", 50, h - 60);
        g.dispose();
        return img;
    }
}
