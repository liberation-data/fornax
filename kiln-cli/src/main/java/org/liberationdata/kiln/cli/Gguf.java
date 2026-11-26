package org.liberationdata.kiln.cli;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** GGUF file names: split models ({@code name-00001-of-00003.gguf}) and quant tags ({@code repo:Q4_K_M}). */
final class Gguf {
    private Gguf() {}

    private static final Pattern SPLIT = Pattern.compile("(.*)-(\\d{5})-of-(\\d{5})\\.gguf", Pattern.CASE_INSENSITIVE);

    static boolean isGguf(String path) { return path.toLowerCase(Locale.ROOT).endsWith(".gguf"); }

    /** Vision projectors and the like: GGUFs that aren't models you can chat with on their own. */
    static boolean isAuxiliary(String path) { return name(path).toLowerCase(Locale.ROOT).contains("mmproj"); }

    /**
     * The chat models among {@code paths}: each key is the file to load (the first part of a split model),
     * each value every file that model needs, in order.
     */
    static Map<String, List<String>> models(Collection<String> paths) {
        Map<String, List<String>> out = new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
        Map<String, List<String>> splits = new TreeMap<>();
        for (String p : paths) {
            if (!isGguf(p) || isAuxiliary(p)) continue;
            Matcher m = SPLIT.matcher(p);
            if (m.matches()) splits.computeIfAbsent(m.group(1) + "/" + m.group(3), k -> new ArrayList<>()).add(p);
            else out.put(p, List.of(p));
        }
        for (List<String> parts : splits.values()) {
            parts.sort(null);
            out.put(parts.get(0), List.copyOf(parts));
        }
        return out;
    }

    /**
     * Every file a model needs, given the parts of it {@link #models} found: a split model's
     * {@code -00001-of-0000N} through {@code -0000N-of-0000N}, whichever of them are present.
     */
    static List<String> allParts(List<String> found) {
        String first = found.get(0);
        Matcher m = SPLIT.matcher(first);
        if (!m.matches()) return found;
        int n = Integer.parseInt(m.group(3));
        List<String> all = new ArrayList<>();   // only the part number changes, so "-OF-" or ".GGUF" stay as spelt
        for (int i = 1; i <= n; i++)
            all.add(first.substring(0, m.start(2)) + String.format(Locale.ROOT, "%05d", i) + first.substring(m.end(2)));
        return all;
    }

    /**
     * The models (keys of {@link #models}) that {@code tag} selects: an exact path or file name, else every model
     * whose path contains the tag, ignoring case ({@code Q4_K_M}, {@code q8_0}, ...).
     */
    static List<String> match(Map<String, List<String>> models, String tag) {
        List<String> exact = models.keySet().stream()
                .filter(k -> k.equals(tag) || name(k).equals(tag) || models.get(k).contains(tag)).toList();
        if (!exact.isEmpty()) return exact;
        String t = tag.toLowerCase(Locale.ROOT);
        return models.keySet().stream().filter(k -> k.toLowerCase(Locale.ROOT).contains(t)).toList();
    }

    static String name(String path) { return path.substring(path.lastIndexOf('/') + 1); }
}
