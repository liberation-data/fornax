package org.liberationdata.fornax.hub;

import org.liberationdata.fornax.hub.ModelSet.Part;
import org.liberationdata.fornax.hub.ModelSet.Source;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Model folders outside the Hugging Face cache, searched after it: a machine that has run ComfyUI already has many of
 * Fornax's weights in {@code ComfyUI/models/}, and can use them where they are instead of downloading or copying tens of
 * gigabytes. Fornax only reads these folders; it never writes to them or deletes from them.
 *
 * <p>For a part whose file is {@code split_files/vae/flux2-vae.safetensors}, a folder {@code D} is searched for:
 * <ol>
 *   <li>{@code D/split_files/vae/flux2-vae.safetensors}, the file at its path in the repo;</li>
 *   <li>{@code D/vae/flux2-vae.safetensors}, the path without Comfy-Org's {@code split_files/} prefix;</li>
 *   <li>{@code D/<folder>/flux2-vae.safetensors} for the ComfyUI folders of the part's role ({@code vae/} for a VAE,
 *       {@code diffusion_models/}, {@code unet/} or {@code checkpoints/} for a diffusion model, ...), then the others.</li>
 * </ol>
 * A folder that has a {@code models/} directory (ComfyUI's own root) is searched there too.
 */
public final class ModelFolders {
    /** The system property, then environment variable, that list model folders, separated by {@link File#pathSeparator}. */
    public static final String PROPERTY = "fornax.modelDirs", ENV = "FORNAX_MODEL_DIRS";
    /** Fornax Studio's earlier names for the same setting, still read. */
    static final String STUDIO_PROPERTY = "fornax.studio.modelDirs", STUDIO_ENV = "FORNAX_STUDIO_MODEL_DIRS";

    /** ComfyUI's folder names under models/, by the role a part has in a {@link ModelSet}. */
    private static final Map<String, List<String>> BY_ROLE = Map.ofEntries(
            Map.entry("diffusion_model", List.of("diffusion_models", "unet", "checkpoints")),
            Map.entry("dit", List.of("diffusion_models", "unet", "checkpoints")),
            Map.entry("model", List.of("diffusion_models", "checkpoints", "unet")),
            Map.entry("gguf", List.of("text_encoders", "clip", "LLM")),
            Map.entry("llm", List.of("text_encoders", "clip", "LLM")),
            Map.entry("lm", List.of("text_encoders", "LLM")),
            Map.entry("text_encoder", List.of("text_encoders", "clip")),
            Map.entry("mmproj", List.of("text_encoders", "clip", "clip_vision", "LLM")),
            Map.entry("vae", List.of("vae")),
            Map.entry("audio_vae", List.of("vae")),
            Map.entry("upscaler", List.of("latent_upscale_models", "upscale_models")),
            Map.entry("lora", List.of("loras")));
    private static final List<String> ALL_FOLDERS = List.of("diffusion_models", "unet", "checkpoints", "text_encoders", "clip",
            "clip_vision", "vae", "loras", "latent_upscale_models", "upscale_models", "LLM");

    private final List<Path> dirs;

    public ModelFolders(List<Path> dirs) { this.dirs = List.copyOf(dirs); }

    /** No folders: only the Hugging Face cache is searched. */
    public static ModelFolders none() { return new ModelFolders(List.of()); }

    /**
     * The folders in {@value #PROPERTY} or {@value #ENV}, and in Fornax Studio's {@value #STUDIO_PROPERTY} or
     * {@value #STUDIO_ENV}, in that order, each list separated by {@link File#pathSeparator}.
     */
    public static ModelFolders configured() {
        Set<Path> out = new LinkedHashSet<>();
        for (String v : new String[] {System.getProperty(PROPERTY), System.getenv(ENV), System.getProperty(STUDIO_PROPERTY), System.getenv(STUDIO_ENV)}) {
            if (v == null || v.isBlank()) continue;
            Arrays.stream(v.split(File.pathSeparator)).map(String::trim).filter(d -> !d.isEmpty()).map(Path::of).forEach(out::add);
        }
        return new ModelFolders(new ArrayList<>(out));
    }

    public List<Path> dirs() { return dirs; }

    /** The first file in these folders that is one of the part's sources, by the search order above. */
    public Optional<Path> find(Part part) {
        for (Source s : part.sources()) {
            Optional<Path> p = find(part.role(), s.file());
            if (p.isPresent()) return p;
        }
        return Optional.empty();
    }

    /** The file {@code file} (a path in a repo) for a part in {@code role}, by the search order above. */
    public Optional<Path> find(String role, String file) {
        String name = file.substring(file.lastIndexOf('/') + 1);
        List<String> candidates = new ArrayList<>();
        candidates.add(file);
        if (file.startsWith("split_files/")) candidates.add(file.substring("split_files/".length()));
        LinkedHashSet<String> folders = new LinkedHashSet<>(BY_ROLE.getOrDefault(role, List.of()));
        folders.addAll(ALL_FOLDERS);
        for (String f : folders) candidates.add(f + "/" + name);
        for (Path d : dirs) {
            for (Path base : Files.isDirectory(d.resolve("models")) ? List.of(d, d.resolve("models")) : List.of(d)) {
                for (String c : candidates) {
                    Path p = base.resolve(c);
                    if (Files.isRegularFile(p)) return Optional.of(p);
                }
            }
        }
        return Optional.empty();
    }
}
