import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Fills the docs' Java blocks from compiled snippet sources, so an API change that breaks an example breaks the build.
 *
 * <p>A Java block in README.md or a page under website/src/content/docs is marked by the line before it:
 * <pre>
 * &lt;!-- snippet: Images#flux --&gt;        (in .mdx: {/* snippet: Images#flux *&#47;})
 * ```java
 * ...
 * ```
 * </pre>
 * and its body is the region {@code flux} of {@code fornax-examples/src/snippets/java/org/liberationdata/fornax/snippets/Images.java}:
 * every stretch between {@code // --8<-- [start:flux]} and {@code // --8<-- [end:flux]}, each dedented, joined by a
 * blank line (so a block can show its imports, then code from inside a method). Lines with other regions' markers are
 * dropped. A line ending in {@code // --8<-- [skip]} is left out of every region, for code that must compile but
 * shouldn't show.
 *
 * <pre>
 * java scripts/Snippets.java            rewrite the blocks that differ from their regions
 * java scripts/Snippets.java --check    fail, listing them, instead (the build runs this)
 * java scripts/Snippets.java FILE...    only these docs
 * </pre>
 * Either way it fails for a marker whose region isn't there, and for a Java block with no marker.
 */
public class Snippets {
    static final Path ROOT = Path.of("").toAbsolutePath();
    static final Path SOURCES = Path.of("fornax-examples/src/snippets/java/org/liberationdata/fornax/snippets");
    static final Pattern MARKER = Pattern.compile("^\\s*(?:<!--|\\{/\\*)\\s*snippet:\\s*([\\w-]+)#([\\w-]+)\\s*(?:-->|\\*/})\\s*$");
    static final Pattern FENCE = Pattern.compile("^(\\s*)```(\\S*).*$");
    static final Pattern REGION = Pattern.compile("//\\s*--8<--\\s*\\[(start|end):([\\w-]+)]");
    static final String SKIP = "// --8<-- [skip]";

    public static void main(String[] args) throws IOException {
        boolean check = List.of(args).contains("--check");
        List<Path> docs = new ArrayList<>();
        for (String a : args) if (!a.startsWith("--")) docs.add(Path.of(a));
        if (docs.isEmpty()) docs = allDocs();
        List<String> problems = new ArrayList<>(), stale = new ArrayList<>();
        Map<String, Map<String, String>> sources = new LinkedHashMap<>();
        for (Path doc : docs) {
            String text = Files.readString(doc);
            String filled = fill(doc, text, sources, problems);
            if (filled.equals(text)) continue;
            if (check) stale.add(doc.toString());
            else { Files.writeString(doc, filled); System.out.println("updated " + doc); }
        }
        for (String p : problems) System.err.println(p);
        if (!stale.isEmpty())
            System.err.println("Docs out of date with their snippets (run: java scripts/Snippets.java):\n  " + String.join("\n  ", stale));
        if (!problems.isEmpty() || !stale.isEmpty()) System.exit(1);
    }

    static List<Path> allDocs() throws IOException {
        List<Path> out = new ArrayList<>(List.of(Path.of("README.md")));
        try (Stream<Path> s = Files.walk(Path.of("website/src/content/docs"))) {
            s.filter(p -> p.toString().endsWith(".md") || p.toString().endsWith(".mdx")).sorted().forEach(out::add);
        }
        return out;
    }

    /** The doc with every marked Java block's body replaced by its region. */
    static String fill(Path doc, String text, Map<String, Map<String, String>> sources, List<String> problems) throws IOException {
        List<String> lines = new ArrayList<>(List.of(text.split("\n", -1)));
        List<String> out = new ArrayList<>();
        String[] pending = null;   // {file, tag, line} from a marker, for the next fence
        boolean inOther = false;   // inside a non-Java fence, where markers and fences are text
        String otherIndent = "";
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i);
            Matcher f = FENCE.matcher(line);
            if (inOther) {
                out.add(line);
                if (f.matches() && f.group(2).isEmpty() && f.group(1).equals(otherIndent)) inOther = false;
                continue;
            }
            Matcher m = MARKER.matcher(line);
            if (m.matches()) {
                if (pending != null) problems.add(doc + ":" + pending[2] + ": snippet marker with no ```java block after it");
                pending = new String[]{m.group(1), m.group(2), String.valueOf(i + 1)};
                out.add(line);
                continue;
            }
            if (!f.matches()) {
                if (pending != null && !line.isBlank()) {
                    problems.add(doc + ":" + pending[2] + ": snippet marker with no ```java block after it");
                    pending = null;
                }
                out.add(line);
                continue;
            }
            String indent = f.group(1);
            if (!f.group(2).equals("java")) { inOther = true; otherIndent = indent; out.add(line); continue; }
            int end = i + 1;
            while (end < lines.size() && !lines.get(end).equals(indent + "```")) end++;
            if (end == lines.size()) { problems.add(doc + ":" + (i + 1) + ": unclosed ```java block"); out.addAll(lines.subList(i, lines.size())); break; }
            out.add(line);
            if (pending == null) {
                problems.add(doc + ":" + (i + 1) + ": ```java block with no snippet marker before it");
                out.addAll(lines.subList(i + 1, end));
            } else {
                String body = region(pending[0], pending[1], sources);
                if (body == null) {
                    problems.add(doc + ":" + pending[2] + ": no region " + pending[1] + " in " + SOURCES.resolve(pending[0] + ".java"));
                    out.addAll(lines.subList(i + 1, end));
                } else {
                    for (String b : body.split("\n", -1)) out.add(b.isEmpty() ? "" : indent + b);
                }
                pending = null;
            }
            out.add(lines.get(end));
            i = end;
        }
        if (pending != null) problems.add(doc + ":" + pending[2] + ": snippet marker with no ```java block after it");
        return String.join("\n", out);
    }

    /** The region's text, or null when the file or region isn't there. */
    static String region(String file, String tag, Map<String, Map<String, String>> sources) throws IOException {
        Map<String, String> regions = sources.get(file);
        if (regions == null) {
            Path p = SOURCES.resolve(file + ".java");
            regions = Files.isRegularFile(p) ? regions(Files.readAllLines(p)) : Map.of();
            sources.put(file, regions);
        }
        return regions.get(tag);
    }

    /** Every region in a source, by tag: its stretches dedented and joined by a blank line. */
    static Map<String, String> regions(List<String> lines) {
        Map<String, List<List<String>>> parts = new LinkedHashMap<>();
        Map<String, List<String>> open = new LinkedHashMap<>();
        for (String line : lines) {
            Matcher m = REGION.matcher(line);
            if (m.find()) {
                if (m.group(1).equals("start")) {
                    List<String> part = new ArrayList<>();
                    open.put(m.group(2), part);
                    parts.computeIfAbsent(m.group(2), k -> new ArrayList<>()).add(part);
                } else open.remove(m.group(2));
                continue;
            }
            if (line.stripTrailing().endsWith(SKIP)) continue;
            for (List<String> part : open.values()) part.add(line);
        }
        Map<String, String> out = new LinkedHashMap<>();
        parts.forEach((tag, ps) -> {
            List<String> joined = new ArrayList<>();
            for (List<String> p : ps) {
                if (!joined.isEmpty()) joined.add("");
                joined.addAll(dedent(p));
            }
            out.put(tag, String.join("\n", joined));
        });
        return out;
    }

    static List<String> dedent(List<String> lines) {
        int min = lines.stream().filter(l -> !l.isBlank()).mapToInt(l -> l.length() - l.stripLeading().length()).min().orElse(0);
        List<String> out = new ArrayList<>();
        for (String l : lines) out.add(l.isBlank() ? "" : l.substring(min).stripTrailing());
        while (!out.isEmpty() && out.getFirst().isEmpty()) out.removeFirst();
        while (!out.isEmpty() && out.getLast().isEmpty()) out.removeLast();
        return out;
    }
}
