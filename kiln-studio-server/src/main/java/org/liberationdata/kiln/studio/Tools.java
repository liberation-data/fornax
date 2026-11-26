package org.liberationdata.kiln.studio;

import java.time.ZoneId;
import java.time.ZonedDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Map;

/**
 * The tools the Chat window's tool-calling demo offers a model, in OpenAI's format for {@code LlamaModel.chatJson}, and
 * what calling each does. Both are safe to run on anything a model asks: a clock and arithmetic.
 */
final class Tools {
    private Tools() {}

    static final List<Object> DEFINITIONS = List.of(
            function("get_current_time", "The current date and time in a time zone.",
                    Json.obj("timezone", Json.obj("type", "string",
                            "description", "An IANA time zone such as Australia/Sydney or Europe/London. Defaults to the system's.")),
                    List.of()),
            function("calculate", "Evaluates an arithmetic expression with + - * / ^ and parentheses, exactly.",
                    Json.obj("expression", Json.obj("type", "string", "description", "For example (17 + 4) * 3 / 2")),
                    List.of("expression")));

    private static Map<String, Object> function(String name, String description, Map<String, Object> properties, List<String> required) {
        return Json.obj("type", "function", "function", Json.obj("name", name, "description", description,
                "parameters", Json.obj("type", "object", "properties", properties, "required", required)));
    }

    /** What the tool answers, as the text the model is given back; a failure is answered as text too. */
    static String call(String name, Map<String, Object> args) {
        try {
            return switch (name) {
                case "get_current_time" -> {
                    String tz = Json.str(args, "timezone", "");
                    ZoneId zone = tz.isBlank() ? ZoneId.systemDefault() : ZoneId.of(tz);
                    yield ZonedDateTime.now(zone).format(DateTimeFormatter.ofPattern("EEEE d MMMM yyyy, HH:mm z"));
                }
                case "calculate" -> format(new Arithmetic(Json.str(args, "expression", "")).evaluate());
                default -> "error: no tool named " + name;
            };
        } catch (RuntimeException e) {
            return "error: " + e.getMessage();
        }
    }

    private static String format(double v) {
        if (v == Math.rint(v) && Math.abs(v) < 1e15) return Long.toString((long) v);
        return Double.toString(v);
    }

    /** + - * / ^ and parentheses over decimals, by recursive descent: no scripting engine to evaluate what a model wrote. */
    static final class Arithmetic {
        private final String s;
        private int i;

        Arithmetic(String s) { this.s = s.replace(" ", ""); }

        double evaluate() {
            if (s.isEmpty()) throw new IllegalArgumentException("empty expression");
            double v = sum();
            if (i != s.length()) throw new IllegalArgumentException("unexpected '" + s.charAt(i) + "' at " + i);
            return v;
        }

        private double sum() {
            double v = product();
            while (i < s.length() && (s.charAt(i) == '+' || s.charAt(i) == '-')) v = s.charAt(i++) == '+' ? v + product() : v - product();
            return v;
        }

        private double product() {
            double v = power();
            while (i < s.length() && (s.charAt(i) == '*' || s.charAt(i) == '/' || s.charAt(i) == '×' || s.charAt(i) == '÷')) {
                char op = s.charAt(i++);
                double r = power();
                if ((op == '/' || op == '÷') && r == 0) throw new ArithmeticException("division by zero");
                v = op == '*' || op == '×' ? v * r : v / r;
            }
            return v;
        }

        private double power() {
            double base = unary();
            if (i < s.length() && s.charAt(i) == '^') {
                i++;
                return Math.pow(base, power());  // right-associative
            }
            return base;
        }

        private double unary() {
            if (i < s.length() && s.charAt(i) == '-') { i++; return -unary(); }
            if (i < s.length() && s.charAt(i) == '+') { i++; return unary(); }
            if (i < s.length() && s.charAt(i) == '(') {
                i++;
                double v = sum();
                if (i >= s.length() || s.charAt(i) != ')') throw new IllegalArgumentException("missing ')'");
                i++;
                return v;
            }
            int start = i;
            while (i < s.length() && (Character.isDigit(s.charAt(i)) || s.charAt(i) == '.')) i++;
            if (start == i) throw new IllegalArgumentException(i < s.length() ? "unexpected '" + s.charAt(i) + "'" : "unexpected end");
            return Double.parseDouble(s.substring(start, i));
        }
    }
}
