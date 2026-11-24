package org.liberationdata.kiln.studio;

import org.liberationdata.kiln.HfCache;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * What in the Hugging Face cache Kiln can run, as the studio offers it:
 * <ul>
 *   <li><b>chat</b>: every GGUF language model in the cache (llama.cpp), found by scanning it;</li>
 *   <li><b>image</b> and <b>song</b>: the presets the Kiln examples use (stable-diffusion.cpp, ACE-Step), each a set of
 *       files, available when all of them are downloaded.</li>
 * </ul>
 * Read afresh on each call, so a model downloaded while the studio runs shows up on the next refresh.
 */
final class Catalog {

    enum Kind { chat, image, song }

    /** One file a model needs: which part it is, where on the Hub it comes from, and where it is now (or null). */
    record Part(String role, String repo, String file, Path path) {
        boolean present() { return path != null; }
    }

    /**
     * A model the studio can load. {@code defaults} are the generation settings the preset suggests (steps, guidance,
     * size, ...), which the page fills its form with.
     */
    record Model(String id, Kind kind, String name, String engine, List<Part> parts, Map<String, Object> defaults) {
        boolean available() { return parts.stream().allMatch(Part::present); }

        long sizeBytes() {
            long n = 0;
            for (Part p : parts) {
                if (!p.present()) continue;
                try { n += Files.size(p.path()); } catch (IOException e) { throw new UncheckedIOException(e); }
            }
            return n;
        }

        Path path(String role) {
            return parts.stream().filter(p -> p.role().equals(role)).findFirst().map(Part::path).orElse(null);
        }
    }

    // GGUFs in the cache that aren't chat models: the song engines' own weights, embedders, VAEs and vision projectors.
    private static final Set<String> NOT_CHAT_REPOS = Set.of("Serveurperso/ACE-Step-1.5-GGUF", "Serveurperso/YuE2-GGUF");
    private static final Pattern NOT_CHAT_FILE = Pattern.compile("(?i).*(embed|vae|mmproj|reranker).*");
    // Of a split GGUF, the first shard names the model (llama.cpp loads the rest beside it).
    private static final Pattern SHARD = Pattern.compile("(?i)(.*)-(\\d{5})-of-(\\d{5})\\.gguf");

    private final Path root;

    Catalog(Path root) { this.root = root; }

    Catalog() { this(HfCache.root()); }

    Path root() { return root; }

    List<Model> models() {
        List<Model> all = new ArrayList<>(chatModels());
        all.addAll(imagePresets());
        all.addAll(songPresets());
        return all;
    }

    Optional<Model> find(String id) { return models().stream().filter(m -> m.id().equals(id)).findFirst(); }

    // ---- chat: scanned ----

    List<Model> chatModels() {
        List<Model> out = new ArrayList<>();
        if (!Files.isDirectory(root)) return out;
        try (Stream<Path> repos = Files.list(root)) {
            for (Path repoDir : repos.filter(p -> p.getFileName().toString().startsWith("models--")).sorted().toList()) {
                String repo = repoDir.getFileName().toString().substring("models--".length()).replace("--", "/");
                if (NOT_CHAT_REPOS.contains(repo)) continue;
                Path snapshots = repoDir.resolve("snapshots");
                if (!Files.isDirectory(snapshots)) continue;
                // The same file in several snapshots is one model: the first snapshot to have it.
                Map<String, Path> files = new LinkedHashMap<>();
                try (Stream<Path> snaps = Files.list(snapshots)) {
                    for (Path snap : snaps.sorted().toList()) {
                        try (Stream<Path> walk = Files.walk(snap)) {
                            walk.filter(p -> p.getFileName().toString().toLowerCase().endsWith(".gguf"))
                                .filter(Files::isRegularFile)  // follows the cache's symlinks into blobs
                                .sorted(Comparator.comparing(Path::toString))
                                .forEach(p -> files.putIfAbsent(snap.relativize(p).toString(), p));
                        }
                    }
                }
                for (var e : files.entrySet()) {
                    String file = e.getKey();
                    String base = Path.of(file).getFileName().toString();
                    if (NOT_CHAT_FILE.matcher(base).matches()) continue;
                    var shard = SHARD.matcher(base);
                    List<Part> parts = new ArrayList<>();
                    if (shard.matches()) {
                        if (!shard.group(2).equals("00001")) continue;
                        int n = Integer.parseInt(shard.group(3));
                        String dir = file.contains("/") ? file.substring(0, file.lastIndexOf('/') + 1) : "";
                        for (int k = 1; k <= n; k++) {
                            String f = dir + String.format("%s-%05d-of-%05d.gguf", shard.group(1), k, n);
                            parts.add(new Part(k == 1 ? "gguf" : "shard", repo, f, files.get(f)));
                        }
                    } else {
                        parts.add(new Part("gguf", repo, file, e.getValue()));
                    }
                    String name = base.replaceAll("(?i)(-\\d{5}-of-\\d{5})?\\.gguf$", "");
                    out.add(new Model("chat:" + repo + "/" + file, Kind.chat, name, "llama.cpp", parts,
                            Json.obj("maxTokens", 1024L, "temperature", 0.7, "contextSize", 8192L)));
                }
            }
        } catch (IOException e) { throw new UncheckedIOException(e); }
        return out;
    }

    // ---- image and song: presets, as the Kiln examples have them ----

    List<Model> imagePresets() {
        String qwenVl = "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors";
        return List.of(
                preset("image:flux2-klein", Kind.image, "FLUX.2 [klein] base 4B", "stable-diffusion.cpp",
                        Json.obj("width", 768L, "height", 768L, "steps", 12L, "cfg", 4.0),
                        part("diffusion_model", "black-forest-labs/FLUX.2-klein-base-4b-fp8", "flux-2-klein-base-4b-fp8.safetensors"),
                        part("vae", "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
                        part("llm", "Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors")),
                preset("image:qwen-image-2512", Kind.image, "Qwen-Image 2512", "stable-diffusion.cpp",
                        Json.obj("width", 1024L, "height", 1024L, "steps", 20L, "cfg", 2.5),
                        part("diffusion_model", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/diffusion_models/qwen_image_2512_fp8_e4m3fn.safetensors"),
                        part("vae", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors"),
                        anyPart("llm", "Comfy-Org/Qwen-Image_ComfyUI", qwenVl, "Comfy-Org/HunyuanVideo_1.5_repackaged", qwenVl)));
    }

    List<Model> songPresets() {
        String ace = "Serveurperso/ACE-Step-1.5-GGUF";
        Map<String, Object> song = Json.obj("duration", 60L, "bpm", 96L, "keyscale", "G major", "steps", 8L, "cfg", 1.0, "shift", 3.0);
        return List.of(
                preset("song:acestep-turbo", Kind.song, "ACE-Step 1.5 turbo", "acestep.cpp", song,
                        part("text_encoder", ace, "Qwen3-Embedding-0.6B-Q8_0.gguf"),
                        part("dit", ace, "acestep-v15-turbo-Q8_0.gguf"),
                        part("vae", ace, "vae-BF16.gguf")),
                preset("song:acestep-turbo-lm4b", Kind.song, "ACE-Step 1.5 turbo + 4B LM", "acestep.cpp", song,
                        part("lm", ace, "acestep-5Hz-lm-4B-Q8_0.gguf"),
                        part("text_encoder", ace, "Qwen3-Embedding-0.6B-Q8_0.gguf"),
                        part("dit", ace, "acestep-v15-turbo-Q8_0.gguf"),
                        part("vae", ace, "vae-BF16.gguf")),
                preset("song:acestep-xl-turbo", Kind.song, "ACE-Step 1.5 XL turbo", "acestep.cpp", song,
                        part("text_encoder", ace, "Qwen3-Embedding-0.6B-Q8_0.gguf"),
                        part("dit", ace, "acestep-v15-xl-turbo-Q8_0.gguf"),
                        part("vae", ace, "vae-BF16.gguf")));
    }

    private static Model preset(String id, Kind kind, String name, String engine, Map<String, Object> defaults, Part... parts) {
        return new Model(id, kind, name, engine, List.of(parts), defaults);
    }

    private Part part(String role, String repo, String file) {
        return new Part(role, repo, file, find(repo, file));
    }

    /** The first of two places a file may be in, as HfCache.requireAny has it. */
    private Part anyPart(String role, String repo1, String file1, String repo2, String file2) {
        Path p = find(repo1, file1);
        if (p != null) return new Part(role, repo1, file1, p);
        Path q = find(repo2, file2);
        return q != null ? new Part(role, repo2, file2, q) : new Part(role, repo1, file1, null);
    }

    // HfCache.find, on this catalogue's root (HfCache reads its root from the environment, which a test can't set).
    private Path find(String repo, String file) {
        Path snapshots = root.resolve("models--" + repo.replace("/", "--")).resolve("snapshots");
        if (!Files.isDirectory(snapshots)) return null;
        try (Stream<Path> s = Files.list(snapshots)) {
            return s.sorted().map(snap -> snap.resolve(file)).filter(Files::exists).findFirst().orElse(null);
        } catch (IOException e) { throw new UncheckedIOException(e); }
    }
}
