/**
 * Captions as they are read: at most two lines of at most 42 characters each, the usual limit for subtitles. A longer
 * sentence is split into several captions, at its commas, colons and dashes where it can be and between words where
 * it can't, and each caption shows for its share of the sentence's time. A caption's two lines are broken where they
 * come out most even, so no player wraps it to a third line, and none runs from edge to edge of the picture.
 *
 * The video build (build.mjs) times its captions with these, and `node videos/build/captions.mjs <in.vtt> <out.vtt>`
 * redoes an existing cut's captions without rendering it again.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export const LINE = 42;

/* Words that read badly at the end of a line, away from what they lead into. */
const SMALL = new Set(['a', 'an', 'the', 'no', 'to', 'of', 'in', 'on', 'at', 'for', 'and', 'or', 'but', 'your', 'its', 'it', 'is', 'as', 'with', 'from', 'what', 'where', 'which', 'who', 'one']);

/** The words of `text` in the fewest captions of at most two lines, each caption's lines broken most evenly. */
export function split(text) {
    const words = text.split(/\s+/).filter(Boolean);
    /* Where a caption may end, best first: after a clause's punctuation, else between any two words. */
    const clause = (word) => /[,;:]["')]*$|^[—–-]$/.test(word);
    const pages = [];
    let from = 0;
    while (from < words.length) {
        /* Every end that fits in two lines; the last is the fullest caption. */
        const fits = [];
        for (let to = from + 1; to <= words.length; to++) {
            const lines = twoLines(words.slice(from, to));
            if (!lines) break;
            fits.push({ to, lines });
        }
        if (!fits.length) fits.push({ to: from + 1, lines: [words[from]] });
        const fullest = fits.at(-1);
        const length = (fit) => fit.lines.join(' ').length;
        const rest = (fit) => words.slice(fit.to).join(' ').length;
        /* The rest of the sentence if it fits. Else the last clause that fills more than a line, or any last clause if
           the fullest caption would leave a scrap of a few words for the next. Else the fullest. */
        const scrap = rest(fullest) < LINE / 2;
        const atClause = fits.findLast((fit) => clause(words[fit.to - 1]) && (length(fit) > LINE || scrap));
        const chosen = fullest.to === words.length ? fullest : (atClause ?? fullest);
        pages.push(chosen.lines.join('\n'));
        from = chosen.to;
    }
    return pages;
}

/** The words as one line, or two broken most evenly, each at most LINE long; null if they don't fit in two. */
function twoLines(words) {
    const whole = words.join(' ');
    if (whole.length <= LINE) return [whole];
    let best = null;
    for (let k = 1; k < words.length; k++) {
        const a = words.slice(0, k).join(' ');
        const b = words.slice(k).join(' ');
        if (a.length > LINE || b.length > LINE) continue;
        /* As even as can be, but a comma is a better place to break, and a little word ends no line. */
        const spread = Math.abs(a.length - b.length) + (/,["')]*$/.test(a) ? -12 : 0) + (SMALL.has(words[k - 1].toLowerCase()) ? 12 : 0);
        if (!best || spread < best.spread) best = { lines: [a, b], spread };
    }
    return best?.lines ?? null;
}

/** `[from, to, text]` cues, each split by `split` and its time shared out by the characters of each piece. */
export function paced(cues) {
    return cues.flatMap(([from, to, text]) => {
        const pages = split(text);
        const total = pages.reduce((n, page) => n + page.length, 0);
        let t = from;
        return pages.map((page) => {
            const d = ((to - from) * page.length) / total;
            const cue = [t, t + d, page];
            t += d;
            return cue;
        });
    });
}

const stamp = (s) => new Date(Math.max(0, s) * 1000).toISOString().slice(11, 23);
const seconds = (stamp) => stamp.split(':').reduce((n, part) => n * 60 + Number(part), 0);

/** `[from, to, text]` cues as WebVTT. */
export function vtt(cues) {
    return `WEBVTT\n\n${cues.map(([from, to, text]) => `${stamp(from)} --> ${stamp(to)}\n${text}`).join('\n\n')}\n`;
}

/** `[from, to, text]` cues as SubRip, for YouTube: numbered, with a comma before the milliseconds. */
export function srt(cues) {
    return `${cues.map(([from, to, text], k) => `${k + 1}\n${stamp(from).replace('.', ',')} --> ${stamp(to).replace('.', ',')}\n${text}`).join('\n\n')}\n`;
}

/** A WebVTT file's cues as `[from, to, text]`, a cue's lines joined by spaces, or kept on their lines with `{ lines: true }`. */
export function read(file, { lines: keep = false } = {}) {
    /* Only the blocks that are cues: none in a file without speech, and no NOTE or STYLE. */
    return file.replace(/^WEBVTT[^\n]*\n/, '').trim().split(/\r?\n\r?\n/).filter((block) => block.includes('-->')).map((block) => {
        const lines = block.split(/\r?\n/);
        const at = lines.findIndex((line) => line.includes('-->'));
        const [from, to] = lines[at].split('-->').map((it) => seconds(it.trim().split(/\s/)[0]));
        return [from, to, lines.slice(at + 1).join(keep ? '\n' : ' ')];
    });
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
    const [input, output] = process.argv.slice(2);
    if (!input || !output) throw new Error('usage: node videos/build/captions.mjs <in.vtt> <out.vtt>');
    writeFileSync(output, vtt(paced(read(readFileSync(input, 'utf8')))));
}
