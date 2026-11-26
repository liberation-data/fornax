package org.liberationdata.fornax.hub;

import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A model on the Hugging Face Hub by name: {@code owner/repo[@revision][:tag]}, the form {@code fornax pull} takes and
 * {@link FornaxHub#fetch} downloads. The tag picks a file in the repo: a quant such as {@code Q4_K_M} (matched against
 * the repo's GGUF file names, ignoring case), or a file's name or path. With no tag, the repo must hold exactly one
 * GGUF model. A split GGUF ({@code -00001-of-00003.gguf}) brings all its parts.
 *
 * <pre>{@code
 * unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M
 * Qwen/Qwen3-0.6B-GGUF                          (its only GGUF model)
 * ggerganov/whisper.cpp:ggml-base.en.bin        (a file by name)
 * some/repo@v1.2:Q8_0                           (a branch, tag or commit; default main)
 * }</pre>
 *
 * A URL isn't a reference: {@link #parse} refuses one and says which reference to use instead.
 *
 * @param revision null for the default branch ({@code main})
 * @param tag      null to take the repo's only GGUF model
 */
public record ModelRef(String repo, String revision, String tag) {
    private static final Pattern REF = Pattern.compile("([\\w.-]+/[\\w.-]+)(?:@([^:@\\s]+))?(?::([^:\\s]+))?");
    private static final Pattern HF_URL = Pattern.compile(
            "https?://(?:www\\.)?(?:huggingface\\.co|hf\\.co)/(?:models/)?([\\w.-]+/[\\w.-]+)(?:/(?:blob|resolve|tree)/([^/?#]+)(?:/([^?#]+))?)?/?(?:[?#].*)?");

    public ModelRef {
        Objects.requireNonNull(repo, "repo");
        if (revision != null && revision.isBlank()) revision = null;
        if (tag != null && tag.isBlank()) tag = null;
    }

    /** {@code owner/repo[@revision][:tag]}, or an IllegalArgumentException saying what a reference looks like. */
    public static ModelRef parse(String s) {
        return tryParse(s).orElseThrow(() -> new IllegalArgumentException(urlHint(s).map(ref ->
                        "a URL isn't a model reference: use " + ref + " (owner/repo[@revision][:tag])")
                .orElse("not a model reference: " + s + " (owner/repo[@revision][:tag], e.g. "
                        + "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M)")));
    }

    /** The reference s spells, or empty when it isn't one (a URL, a path with more than one '/', ...). */
    public static Optional<ModelRef> tryParse(String s) {
        if (s == null) return Optional.empty();
        Matcher m = REF.matcher(s.strip());
        return m.matches() ? Optional.of(new ModelRef(m.group(1), m.group(2), m.group(3))) : Optional.empty();
    }

    /**
     * Whether s, which names no existing file, reads as a reference rather than a mistyped path: it matches the
     * grammar, and isn't {@code folder/model.gguf} (a relative path with a model file's extension and no tag).
     */
    static boolean looksLikeReference(String s) {
        Optional<ModelRef> r = tryParse(s);
        if (r.isEmpty()) return false;
        String lower = r.get().repo.toLowerCase(Locale.ROOT);
        boolean fileLike = lower.endsWith(".gguf") || lower.endsWith(".bin") || lower.endsWith(".safetensors");
        return !(fileLike && r.get().tag == null && r.get().revision == null);
    }

    /** For a Hugging Face URL, the reference that names the same file (or repo), as the error for a URL suggests. */
    public static Optional<String> urlHint(String s) {
        if (s == null) return Optional.empty();
        Matcher m = HF_URL.matcher(s.strip());
        if (!m.matches()) return Optional.empty();
        String rev = m.group(2), file = m.group(3);
        return Optional.of(new ModelRef(m.group(1), "main".equals(rev) ? null : rev, file).toString());
    }

    @Override public String toString() {
        return repo + (revision == null ? "" : "@" + revision) + (tag == null ? "" : ":" + tag);
    }
}
