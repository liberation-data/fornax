package org.liberationdata.fornax.studio;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * The GPU memory the whole system is using, where the driver tells it (amdgpu's {@code mem_info_*} in sysfs): VRAM, and
 * GTT, the system memory the GPU maps, which is most of it on an APU such as Strix Halo. Elsewhere, an empty list.
 * This is every process's use, not only the studio's: the pool's own estimate is beside it in the status.
 */
final class Gpu {
    private Gpu() {}

    static List<Object> memory() {
        List<Object> out = new ArrayList<>();
        Path drm = Path.of("/sys/class/drm");
        if (!Files.isDirectory(drm)) return out;
        try (Stream<Path> cards = Files.list(drm)) {
            for (Path card : cards.filter(p -> p.getFileName().toString().matches("card\\d+")).sorted().toList()) {
                Path dev = card.resolve("device");
                Long vramUsed = read(dev.resolve("mem_info_vram_used")), vramTotal = read(dev.resolve("mem_info_vram_total"));
                Long gttUsed = read(dev.resolve("mem_info_gtt_used")), gttTotal = read(dev.resolve("mem_info_gtt_total"));
                if (vramTotal == null && gttTotal == null) continue;
                out.add(Json.obj("card", card.getFileName().toString(),
                        "vramUsed", vramUsed, "vramTotal", vramTotal, "gttUsed", gttUsed, "gttTotal", gttTotal));
            }
        } catch (IOException ignored) {
            // Not told, then.
        }
        return out;
    }

    private static Long read(Path p) {
        try { return Long.parseLong(Files.readString(p).trim()); } catch (IOException | NumberFormatException e) { return null; }
    }
}
