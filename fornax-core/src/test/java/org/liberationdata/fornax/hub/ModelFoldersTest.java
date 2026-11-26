package org.liberationdata.fornax.hub;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.liberationdata.fornax.hub.ModelSet.Part;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class ModelFoldersTest {
    @TempDir Path dir;

    private Path file(String rel) throws Exception {
        Path p = dir.resolve(rel);
        Files.createDirectories(p.getParent());
        Files.write(p, new byte[1]);
        return p;
    }

    @Test void findsAFileAtItsRepoPath() throws Exception {
        Path p = file("diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors");
        var folders = new ModelFolders(List.of(dir));
        assertEquals(Optional.of(p), folders.find(ModelCatalog.require("ltx-2.5-distilled").part("diffusion_model").orElseThrow()));
    }

    @Test void dropsComfyOrgsSplitFilesPrefix() throws Exception {
        Path p = file("vae/flux2-vae.safetensors");
        assertEquals(Optional.of(p), new ModelFolders(List.of(dir)).find(Part.of("vae", "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors")));
    }

    @Test void looksInTheRolesComfyFoldersByFileName() throws Exception {
        Path p = file("ComfyUI/models/unet/flux-2-klein-base-4b-fp8.safetensors");   // a ComfyUI root: searched in its models/
        var folders = new ModelFolders(List.of(dir.resolve("ComfyUI")));
        assertEquals(Optional.of(p), folders.find(ModelCatalog.require("flux2-klein").part("diffusion_model").orElseThrow()));
        assertEquals(Optional.empty(), folders.find(ModelCatalog.require("flux2-klein").part("vae").orElseThrow()));
    }

    @Test void theRolesFolderComesFirst() throws Exception {
        file("checkpoints/model.safetensors");
        Path vae = file("vae/model.safetensors");
        assertEquals(Optional.of(vae), new ModelFolders(List.of(dir)).find("vae", "model.safetensors"));
    }

    @Test void readsFornaxsSettingAndStudiosOld() {
        String a = dir.resolve("a").toString(), b = dir.resolve("b").toString(), c = dir.resolve("c").toString();
        try {
            System.setProperty(ModelFolders.PROPERTY, a + File.pathSeparator + b);
            System.setProperty(ModelFolders.STUDIO_PROPERTY, c + File.pathSeparator + a);
            List<Path> dirs = ModelFolders.configured().dirs();
            assertEquals(List.of(Path.of(a), Path.of(b)), dirs.subList(0, 2));
            assertTrue(dirs.contains(Path.of(c)));
            assertEquals(dirs.size(), dirs.stream().distinct().count());
        } finally {
            System.clearProperty(ModelFolders.PROPERTY);
            System.clearProperty(ModelFolders.STUDIO_PROPERTY);
        }
    }
}
