package org.liberationdata.fornax.sd;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static java.lang.foreign.ValueLayout.*;
import static org.junit.jupiter.api.Assertions.*;

/** Request validation and option mapping for img2img, inpainting, ControlNet and LoRA. No native library needed. */
class RequestOptionsTest {

    private static final BufferedImage IMG = new BufferedImage(64, 48, BufferedImage.TYPE_INT_RGB);
    private static final DiffusionModel.Request BASE = new DiffusionModel.Request("a fornax", 512, 384, 8, 4.0, 7);

    private static String refused(DiffusionModel.Request r) {
        return assertThrows(IllegalArgumentException.class, r::options).getMessage();
    }

    @Test
    void aPlainRequestMapsToTheSameKeysAsBefore() {
        assertEquals(Map.of("prompt", "a fornax", "width", 512, "height", 384, "sample.steps", 8,
                "sample.guidance.txt_cfg", 4.0, "seed", 7L), BASE.options());
    }

    @Test
    void img2imgSetsStrength() {
        var o = BASE.withInitImage(IMG, 0.6).options();
        assertEquals(0.6, o.get("strength"));
        assertFalse(o.containsKey("init_image"), "images are tensors, not scalar options");
    }

    @Test
    void anInitImageWithoutStrengthLeavesTheEngineDefault() {
        assertFalse(BASE.withInitImage(IMG).options().containsKey("strength"));
    }

    @Test
    void strengthWithoutAnInitImageIsRefused() {
        var r = new DiffusionModel.Request("p", 64, 64, 4, 1, 1, List.of(), Map.of(), null, 0.5, null, null, null, List.of());
        assertTrue(refused(r).contains("img2img"));
    }

    @Test
    void strengthInExtraIsRefused() {
        assertTrue(refused(BASE.withExtra(Map.of("strength", 0.5))).contains("own Request field"));
        assertTrue(refused(BASE.withInitImage(IMG).withExtra(Map.of("strength", 0.5))).contains("own Request field"));
    }

    @Test
    void strengthOutsideZeroToOneIsRefusedAtOnce() {
        assertThrows(IllegalArgumentException.class, () -> BASE.withInitImage(IMG, 1.5));
        assertThrows(IllegalArgumentException.class, () -> BASE.withInitImage(IMG, -0.1));
        assertThrows(IllegalArgumentException.class, () -> BASE.withInitImage(IMG, Double.NaN));
    }

    @Test
    void aMaskNeedsAnInitImage() {
        assertTrue(refused(BASE.withMaskImage(IMG)).contains("initImage"));
    }

    @Test
    void aMaskMustMatchTheInitImageSize() {
        var mask = new BufferedImage(32, 48, BufferedImage.TYPE_BYTE_GRAY);
        assertTrue(refused(BASE.withInitImage(IMG).withMaskImage(mask)).contains("must match"));
    }

    @Test
    void inpaintingWithAMatchingMaskIsAccepted() {
        var mask = new BufferedImage(64, 48, BufferedImage.TYPE_BYTE_GRAY);
        var o = BASE.withInitImage(IMG, 1.0).withMaskImage(mask).options();
        assertEquals(1.0, o.get("strength"));
    }

    @Test
    void witherOrderDoesNotMatter() {
        var mask = new BufferedImage(64, 48, BufferedImage.TYPE_BYTE_GRAY);
        assertDoesNotThrow(() -> BASE.withMaskImage(mask).withInitImage(IMG, 0.8).options());
    }

    @Test
    void controlStrengthNeedsAControlImage() {
        var r = new DiffusionModel.Request("p", 64, 64, 4, 1, 1, List.of(), Map.of(), null, null, null, null, 0.5, List.of());
        assertTrue(refused(r).contains("controlImage"));
        assertEquals(0.5, BASE.withControlImage(IMG, 0.5).options().get("control_strength"));
        assertThrows(IllegalArgumentException.class, () -> BASE.withControlImage(IMG, -1));
    }

    @Test
    void imagesAndLorasInExtraAreRefused() {
        for (String k : List.of("init_image", "mask_image", "control_image", "ref_images", "loras", "control_strength"))
            assertTrue(refused(BASE.withExtra(Map.of(k, "x"))).contains(k), k);
    }

    @Test
    void aMissingLoraFileIsRefused(@TempDir Path dir) {
        var r = BASE.withLoras(new DiffusionModel.Lora(dir.resolve("nope.safetensors"), 1f));
        assertTrue(refused(r).contains("not found"));
    }

    @Test
    void aLoraWeightMustBeFinite(@TempDir Path dir) {
        assertThrows(IllegalArgumentException.class, () -> new DiffusionModel.Lora(dir, Float.NaN));
        assertEquals(1f, new DiffusionModel.Lora(dir).weight());
    }

    @Test
    void lorasPackAsSdLoraStructs(@TempDir Path dir) throws IOException {
        Path a = Files.createFile(dir.resolve("a.safetensors")), b = Files.createFile(dir.resolve("b.safetensors"));
        var loras = List.of(new DiffusionModel.Lora(a, 0.8f), new DiffusionModel.Lora(b, -0.25f));
        assertDoesNotThrow(() -> BASE.withLoras(loras).options());
        assertEquals(16, DiffusionModel.LORA.byteSize(), "sd_lora_t is { bool; float; const char* } on 64-bit");
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment arr = DiffusionModel.loraArray(loras, arena);
            for (int i = 0; i < 2; i++) {
                assertFalse(arr.get(JAVA_BOOLEAN, i * 16L));
                assertEquals(loras.get(i).weight(), arr.get(JAVA_FLOAT, i * 16L + 4));
                String path = arr.get(ADDRESS, i * 16L + 8).reinterpret(Long.MAX_VALUE).getString(0);
                assertEquals(loras.get(i).file().toAbsolutePath().toString(), path);
            }
        }
    }

    @Test
    void theHiresCheckStillApplies() {
        var o = BASE.withExtra(Map.of("hires.enabled", true, "hires.model_path", "/m/u.safetensors")).options();
        assertEquals("model", o.get("hires.upscaler"));
    }

    @Test
    void theOldConstructorsStillWork() {
        var r = new DiffusionModel.Request("p", 64, 64, 4, 1, 1, List.of(IMG), Map.of("sample.method", "euler"));
        assertNull(r.initImage());
        assertEquals(List.of(), r.loras());
        assertEquals("euler", r.options().get("sample.method"));
    }

    @Test
    void maskTensorsUseStbLuma() {
        var img = new BufferedImage(2, 1, BufferedImage.TYPE_INT_RGB);
        img.setRGB(0, 0, 0xffffff);
        img.setRGB(1, 0, 0x808080);
        byte[] g = DiffusionModel.grey(img);
        assertEquals(255, g[0] & 0xff);
        assertEquals(128, g[1] & 0xff);
    }

    /**
     * ImageIO reads a greyscale PNG as a linear-grey image, and getRGB converts that to sRGB (100 becomes 168), which
     * would move a soft mask's 0.5 threshold and brighten a depth hint. sd-cli uses the stored levels.
     */
    @Test
    void greyImagesKeepTheirStoredLevels() throws IOException {
        var g = new BufferedImage(2, 1, BufferedImage.TYPE_BYTE_GRAY);
        g.getRaster().setSample(0, 0, 0, 100);
        g.getRaster().setSample(1, 0, 0, 200);
        var png = new java.io.ByteArrayOutputStream();
        javax.imageio.ImageIO.write(g, "png", png);
        var read = javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(png.toByteArray()));
        for (BufferedImage img : List.of(g, read)) {
            byte[] grey = DiffusionModel.grey(img), rgb = DiffusionModel.rgb(img);
            assertEquals(100, grey[0] & 0xff);
            assertEquals(200, grey[1] & 0xff);
            for (int c = 0; c < 3; c++) {
                assertEquals(100, rgb[c] & 0xff);
                assertEquals(200, rgb[3 + c] & 0xff);
            }
        }
        var wide = new BufferedImage(1, 1, BufferedImage.TYPE_USHORT_GRAY);
        wide.getRaster().setSample(0, 0, 0, 100 << 8);
        assertEquals(100, DiffusionModel.grey(wide)[0] & 0xff, "16-bit grey drops its low byte, as stb_image does");
    }
}
