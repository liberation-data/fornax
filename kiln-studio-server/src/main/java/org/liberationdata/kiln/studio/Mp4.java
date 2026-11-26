package org.liberationdata.kiln.studio;

import org.liberationdata.kiln.sd.DiffusionModel;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

/**
 * A video's frames and soundtrack made into an .mp4 with ffmpeg, as VideoClip writes them, and a keyframe through one
 * H.264 frame, as VideoClip (and ComfyUI's LTXVPreprocess) prepares it. ffmpeg is {@code kiln.studio.ffmpeg} /
 * {@code KILN_STUDIO_FFMPEG}, else the first on the PATH or where Homebrew puts it (an app started from the Dock on a Mac
 * has no shell's PATH).
 */
final class Mp4 {
    private Mp4() {}

    /** The ffmpeg to run, or null when there is none. */
    static String find(String configured) {
        if (configured != null && !configured.isBlank()) return configured;
        List<Path> dirs = new ArrayList<>();
        for (String d : System.getenv().getOrDefault("PATH", "").split(java.io.File.pathSeparator)) if (!d.isBlank()) dirs.add(Path.of(d));
        dirs.addAll(List.of(Path.of("/opt/homebrew/bin"), Path.of("/usr/local/bin"), Path.of("/usr/bin")));
        return dirs.stream().map(d -> d.resolve("ffmpeg")).filter(Files::isExecutable).findFirst().map(Path::toString).orElse(null);
    }

    /**
     * Writes out.mp4 (H.264, with AAC audio when the video has a soundtrack). The frames are written to a folder beside it
     * first, removed once the .mp4 is made and kept when it can't be, which the failure says.
     */
    static void write(DiffusionModel.Video video, Path out, String ffmpeg) throws IOException {
        String base = out.toString().replaceAll("\\.mp4$", "");
        Path frames = Files.createDirectories(Path.of(base + "-frames"));
        for (int i = 0; i < video.frames().size(); i++) {
            ImageIO.write(video.frames().get(i), "png", frames.resolve(String.format("%05d.png", i)).toFile());
        }
        Path wav = video.audio() != null ? Path.of(base + ".wav") : null;
        if (wav != null) video.audio().writeWav(wav.toFile());
        if (ffmpeg == null) {
            throw new IllegalStateException("making the .mp4 needs ffmpeg (on the PATH, or KILN_STUDIO_FFMPEG); the frames are in "
                    + frames + (wav != null ? " and the audio in " + wav : ""));
        }
        List<String> cmd = new ArrayList<>(List.of(ffmpeg, "-y", "-loglevel", "error",
                "-framerate", String.valueOf(video.fps()), "-i", frames.resolve("%05d.png").toString()));
        if (wav != null) cmd.addAll(List.of("-i", wav.toString(), "-c:a", "aac", "-shortest"));
        // yuv420p wants even sizes; a browser plays nothing else.
        cmd.addAll(List.of("-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", out.toString()));
        String failed = run(cmd);
        if (failed != null) throw new IllegalStateException("ffmpeg failed (" + failed + "); the frames are in " + frames);
        delete(frames);
        if (wav != null) Files.deleteIfExists(wav);
    }

    /** The keyframe through one H.264 frame at the given CRF, or as it is without ffmpeg. */
    static BufferedImage preprocess(BufferedImage image, int crf, String ffmpeg) throws IOException {
        if (crf <= 0 || ffmpeg == null) return image;
        Path tmp = Files.createTempDirectory("kiln-ltx-preprocess");
        try {
            Path png = tmp.resolve("in.png"), mp4 = tmp.resolve("frame.mp4"), back = tmp.resolve("frame.png");
            ImageIO.write(image, "png", png.toFile());
            String failed = run(List.of(ffmpeg, "-y", "-loglevel", "error", "-i", png.toString(), "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", String.valueOf(crf), "-pix_fmt", "yuv420p", mp4.toString()));
            if (failed == null) failed = run(List.of(ffmpeg, "-y", "-loglevel", "error", "-i", mp4.toString(), "-frames:v", "1", back.toString()));
            return failed == null ? ImageIO.read(back.toFile()) : image;
        } finally {
            delete(tmp);
        }
    }

    /** null when the command succeeded; else what it said. */
    private static String run(List<String> cmd) throws IOException {
        var p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        String said = new String(p.getInputStream().readAllBytes()).trim();
        try {
            int rc = p.waitFor();
            return rc == 0 ? null : said.isEmpty() ? "exit " + rc : said;
        } catch (InterruptedException e) {
            p.destroy();
            Thread.currentThread().interrupt();
            return "interrupted";
        }
    }

    private static void delete(Path dir) {
        try (Stream<Path> s = Files.walk(dir)) {
            s.sorted(Comparator.reverseOrder()).forEach(p -> {
                try { Files.deleteIfExists(p); } catch (IOException e) { throw new UncheckedIOException(e); }
            });
        } catch (IOException | UncheckedIOException ignored) {
            // Left behind: the outputs folder is the user's to tidy.
        }
    }
}
