package org.liberationdata.fornax.cli;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Command-line arguments: positionals plus {@code --name value}, {@code --name=value} and boolean {@code --flag}s.
 * {@code --} ends the options. {@code -h}/{@code --help} is always accepted.
 */
record Args(List<String> positional, Map<String, String> options, Set<String> flags) {

    /** A mistake on the command line: printed with the command's usage, exit code 2. */
    static final class UsageException extends RuntimeException {
        private static final long serialVersionUID = 1L;
        UsageException(String message) { super(message); }
    }

    static Args parse(List<String> argv, Set<String> valueOptions, Set<String> flagOptions) {
        List<String> pos = new ArrayList<>();
        Map<String, String> opts = new HashMap<>();
        Set<String> flags = new HashSet<>();
        for (int i = 0; i < argv.size(); i++) {
            String a = argv.get(i);
            if (a.equals("--")) { pos.addAll(argv.subList(i + 1, argv.size())); break; }
            if (a.equals("-h") || a.equals("--help")) { flags.add("help"); continue; }
            if (!a.startsWith("--") || a.length() == 2) { pos.add(a); continue; }
            String name = a.substring(2), value = null;
            int eq = name.indexOf('=');
            if (eq >= 0) { value = name.substring(eq + 1); name = name.substring(0, eq); }
            if (valueOptions.contains(name)) {
                if (value == null) {
                    if (i + 1 >= argv.size()) throw new UsageException("--" + name + " needs a value");
                    value = argv.get(++i);
                }
                opts.put(name, value);
            } else if (flagOptions.contains(name)) {
                if (value != null) throw new UsageException("--" + name + " doesn't take a value");
                flags.add(name);
            } else {
                throw new UsageException("unknown option --" + name);
            }
        }
        return new Args(List.copyOf(pos), Map.copyOf(opts), Set.copyOf(flags));
    }

    boolean has(String flag) { return flags.contains(flag); }

    String option(String name, String fallback) { return options.getOrDefault(name, fallback); }

    int intOption(String name, int fallback) {
        String v = options.get(name);
        if (v == null) return fallback;
        try { return Integer.parseInt(v); }
        catch (NumberFormatException e) { throw new UsageException("--" + name + " must be a whole number, not " + v); }
    }

    float floatOption(String name, float fallback) {
        String v = options.get(name);
        if (v == null) return fallback;
        try { return Float.parseFloat(v); }
        catch (NumberFormatException e) { throw new UsageException("--" + name + " must be a number, not " + v); }
    }

    String positional(int index) { return index < positional.size() ? positional.get(index) : null; }

    List<String> rest(int from) { return from < positional.size() ? positional.subList(from, positional.size()) : List.of(); }
}
