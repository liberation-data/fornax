package org.liberationdata.kiln.demucs;

import org.liberationdata.kiln.Audio;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The stems a separation produced, by name, in the model's order: {@code drums, bass, other, vocals} for HTDemucs
 * ({@code guitar, piano} too for htdemucs_6s). Each is stereo at the input's sample rate and length.
 */
public final class Stems {
    private final Map<String, Audio> stems;
    private final int sampleRate;

    private Stems(Map<String, Audio> stems, int sampleRate) {
        this.stems = stems;
        this.sampleRate = sampleRate;
    }

    /** Pairs names with audio of one sample rate and length, in order. */
    public static Stems of(List<String> names, List<Audio> audio) {
        if (names.isEmpty() || names.size() != audio.size())
            throw new IllegalArgumentException(names.size() + " names for " + audio.size() + " stems");
        Audio first = audio.getFirst();
        Map<String, Audio> m = new LinkedHashMap<>();
        for (int i = 0; i < names.size(); i++) {
            Audio a = audio.get(i);
            if (a.sampleRate() != first.sampleRate() || a.left().length != first.left().length || a.right().length != first.left().length)
                throw new IllegalArgumentException("stem \"" + names.get(i) + "\" has a different rate or length from \"" + names.getFirst() + "\"");
            m.put(names.get(i), a);
        }
        return new Stems(Collections.unmodifiableMap(m), first.sampleRate());
    }

    /** The stem names, in the model's order. */
    public List<String> names() { return List.copyOf(stems.keySet()); }

    public boolean has(String name) { return stems.containsKey(name); }

    /** One stem by name ("drums", "bass", "other", "vocals", ...). */
    public Audio get(String name) {
        Audio a = stems.get(name);
        if (a == null) throw new IllegalArgumentException("no stem \"" + name + "\"; this model makes " + String.join(", ", stems.keySet()));
        return a;
    }

    /** Every stem, by name, in the model's order. */
    public Map<String, Audio> asMap() { return stems; }

    public int sampleRate() { return sampleRate; }

    /** The vocals stem. */
    public Audio vocals() { return get("vocals"); }

    /**
     * Everything but the vocals: the sum of every other stem (demucs' {@code --two-stems vocals} "no_vocals"), or the
     * model's own "instrumental" stem when it has one.
     */
    public Audio backing() {
        vocals();
        if (has("instrumental")) return get("instrumental");
        List<Audio> rest = new ArrayList<>();
        stems.forEach((name, a) -> { if (!name.equals("vocals")) rest.add(a); });
        return sum(rest);
    }

    /** The sum of every stem, which for HTDemucs comes back to the input mix to within the model's error. */
    public Audio mix() { return sum(List.copyOf(stems.values())); }

    private Audio sum(List<Audio> parts) {
        int n = parts.getFirst().left().length;
        float[] l = new float[n], r = new float[n];
        for (Audio a : parts) {
            for (int i = 0; i < n; i++) { l[i] += a.left()[i]; r[i] += a.right()[i]; }
        }
        return new Audio(l, r, sampleRate);
    }

    @Override public String toString() { return "Stems" + stems.keySet(); }
}
