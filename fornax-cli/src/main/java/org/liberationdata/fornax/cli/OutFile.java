package org.liberationdata.fornax.cli;

import org.liberationdata.fornax.cli.Args.UsageException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Where image, song and speak write. Chosen, and its folder made, before the model loads, so that a bad {@code --out}
 * fails in a moment rather than after minutes of loading and generating.
 */
final class OutFile {
    private OutFile() {}

    /**
     * {@code --out} as given (a file to write over, or a folder to put {@code fallback} in), else {@code fallback} in
     * {@code dir}, numbered ({@code image-2.png}, ...) so as not to overwrite what an earlier run made.
     *
     * @param ext the one extension the command writes, such as {@code ".png"}
     */
    static Path choose(String given, Path dir, String fallback, String ext) throws IOException {
        Path file;
        if (given == null) file = unused(dir.resolve(fallback));
        else if (Files.isDirectory(Path.of(given))) file = unused(Path.of(given).resolve(fallback));
        else file = Path.of(given);
        if (!file.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(ext))
            throw new UsageException("--out " + given + ": this writes a " + ext + " file");
        Path parent = file.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
        return file;
    }

    /** {@code file}, or the first of {@code name-2.ext}, {@code name-3.ext}, ... that doesn't exist. */
    static Path unused(Path file) {
        if (!Files.exists(file)) return file;
        String name = file.getFileName().toString();
        int dot = name.lastIndexOf('.');
        String base = dot > 0 ? name.substring(0, dot) : name, ext = dot > 0 ? name.substring(dot) : "";
        for (int i = 2; ; i++) {
            Path next = file.resolveSibling(base + "-" + i + ext);
            if (!Files.exists(next)) return next;
        }
    }
}
