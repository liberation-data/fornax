/**
 * Builds a scripted video from videos/<name>/script.md: `npm run video -- <name>`, run in videos/. Paths below are in
 * the video's folder, videos/<name>/, unless they say otherwise. Ported from the Embabel Worlds landing page's video
 * build (worlds-landing/videos/build/build.mjs), without its publishing, review, hero cards and sting.
 *
 *   1. Voices each scene with Deepgram (DEEPGRAM_API_KEY) or Fornax (`voice: fornax:...`), into narration/. A scene whose words have not changed is
 *      not sent again: narration/<n>.txt holds what its mp3 says. Committed, so a rebuild needs no key.
 *   2. Renders each card in cards/ (HTML, 1920 × 1080, sharing shared/cards/card.css) to a PNG with Chrome or
 *      Chromium, and gives it a slow push-in.
 *   3. Cuts each scene: its cards crossfaded, timed to its narration. A scene's `Clip:` line plays clips/<name>.mp4
 *      after its narration, with its own sound; an `outro:` plays last. Then the scenes, joined. When the music has
 *      a sections.txt with a duck, the cut is fitted to the music, not the other way round (see timingOf).
 *   4. Mixes in the music: the longest audio file in music/out/ (rendered elsewhere), under the voice, ducked under
 *      each word (`duck: on`) and further under a clip, up at the end; then the whole levelled to -16 LUFS.
 *   5. Writes out/<name>.mp4, its captions (out/<name>.vtt), out/timing.json (when each scene and clip starts, for
 *      whoever fits the music) and one frame per scene in out/frames/.
 *
 * Flags: `--cards` renders only the cards (out/work/cards/); `--no-voice` cuts with the narration already on disk and
 * never calls Deepgram; `--music` writes only the music bed as it will sit under the cut (out/work/music.wav);
 * `--final` renders at 60fps with push-ins drawn from 8K (drafts are 30fps from 4K); `--pace=1.1` overrides the
 * script's pace. Needs ffmpeg, and Chrome or Chromium (CHROME_PATH, else /usr/bin/chromium-browser, else Chrome).
 *
 * A script's header may set `voice:` (a Deepgram voice, or `fornax:<model>`, `fornax:<model>:<voice>` or
 * `fornax:<model>:@<clip>` for one voiced through Fornax's SpeakBatch, with `voice-prompt:` and `voice-seed:`), `pace:`, `tail:` (silence after each scene's last word, 0.7 s), `opening:`
 * (silence before the first word, 0.5 s), `duck: on` and `outro: a.mp4 | b.mp4` (the first that exists).
 */
import { createHash } from 'node:crypto';
import { execFileSync, spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright-core';
import { paced, vtt as webVtt } from './captions.mjs';

const VIDEOS = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const SHARED = join(VIDEOS, 'shared');
const name = process.argv.slice(2).find((arg) => !arg.startsWith('--'));
if (!/^[a-z0-9][a-z0-9-]*$/.test(name ?? '')) throw new Error('name a video to build: npm run video -- <name>');
const VIDEO = join(VIDEOS, name);
if (!existsSync(join(VIDEO, 'script.md'))) throw new Error(`videos/${name}/script.md is missing`);
const OUT_ROOT = join(VIDEO, 'out');
const OUT = join(OUT_ROOT, 'work');
const args = new Set(process.argv.slice(2));
const FINAL = args.has('--final');
const FPS = FINAL ? 60 : 30;
const SCRIPT = readFileSync(join(VIDEO, 'script.md'), 'utf8');
/* Silence before and after each scene's narration, and the crossfade between cards. */
const LEAD = 0.5, FADE = 0.5;
const OPENING = Number(/^opening:\s*([\d.]+)/m.exec(SCRIPT)?.[1] ?? LEAD);
const leadOf = (index) => (index === 0 ? OPENING : LEAD);
const TAIL = Number(/^tail:\s*([\d.]+)/m.exec(SCRIPT)?.[1] ?? 0.7);
/* The last scene holds longer, for the music to finish on. */
const END = 3;
/* DUCKING, when script.md says `duck: on`: the music dips under each word, keyed by the sound itself (a sidechain). */
const DUCK = /^duck:\s*on\b/m.test(SCRIPT);
/* Under a clip the music sits this far below where it sits under the voice: the clip's own sound carries it. */
const CLIP_DUCK = -14;
/* Clips are levelled to this, a little under the narration's -16, as footage cut into an explainer is. */
const CLIP_LUFS = -18;
/* The outro's own sound (the stinger's fire), over the music's last bars. */
const OUTRO_LUFS = -19;
/* A beat on the last card after the last word, before the outro starts. */
const OUTRO_BREATH = 1.6;

const run = (cmd, argv) => execFileSync(cmd, argv, { stdio: ['ignore', 'pipe', 'inherit'], maxBuffer: 1 << 28 }).toString();
const ffmpeg = (...argv) => run('ffmpeg', ['-v', 'error', '-y', ...argv]);
const duration = (file) => Number(run('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', file]));
const hasAudio = (file) => !!run('ffprobe', ['-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', file]).trim();

/** A pause in the narration: `[pause]`, half a second and a bit, or `[pause 1.2s]`. */
const PAUSE = /\s*\[pause(?:\s+([\d.]+)s?)?\]\s*/g;

/** The script: its voice, and each scene's cards, narration and clip. */
function readScript() {
    const voice = /^voice:\s*(\S+)/m.exec(SCRIPT)?.[1] ?? 'aura-2-hyperion-en';
    // A Fornax voice (`voice: fornax:<model>`) may be designed by a description, `voice-prompt: (A warm narrator)`, at a seed.
    const voicePrompt = /^voice-prompt:\s*(.+)$/m.exec(SCRIPT)?.[1]?.trim() ?? '';
    const voiceSeed = Number(/^voice-seed:\s*(\d+)/m.exec(SCRIPT)?.[1] ?? 1);
    const paceArg = process.argv.find((a) => a.startsWith('--pace='))?.slice(7);
    const pace = Number(paceArg ?? /^pace:\s*(\S+)/m.exec(SCRIPT)?.[1] ?? 1);
    if (!(pace >= 0.8 && pace <= 1.25)) throw new Error(`script.md: pace ${pace} is outside 0.8 to 1.25, where speech still sounds natural`);
    const scenes = SCRIPT.split(/^## /m).slice(1).map((block) => {
        const [heading, ...lines] = block.split('\n');
        const n = /^(\d+)\./.exec(heading)?.[1];
        if (!n) throw new Error(`script.md: a scene heading needs its number, "## 1. Name": ${heading}`);
        const cardsLine = lines.find((l) => /^Cards:/i.test(l));
        if (!cardsLine) throw new Error(`script.md: scene ${n} has no Cards: line`);
        /* Split at commas outside quotes, so a cue's phrase may hold one. */
        const cards = cardsLine.replace(/^Cards:/i, '').split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/).map((c) => c.trim()).filter(Boolean).map((c) => {
            const cue = /@"([^"]+)"/.exec(c)?.[1] ?? null;
            const [card, secs, ...extra] = c.replace(/@"[^"]*"/, '').trim().split(/\s+/);
            if (extra.length) throw new Error(`script.md: scene ${n}: "${c}" is a card, its seconds and a cue; move notes to a line of their own`);
            /* A card is an HTML page in cards/, or silent footage in footage/<name>.mp4 (a screen recording, say). */
            const footage = join(VIDEO, 'footage', `${card}.mp4`);
            if (!existsSync(join(VIDEO, 'cards', `${card}.html`)) && !existsSync(footage)) throw new Error(`script.md: scene ${n}: no card cards/${card}.html or footage/${card}.mp4`);
            const seconds = secs ? Number(secs.replace(/s$/, '')) : null;
            if (seconds !== null && !(seconds > 0)) throw new Error(`script.md: scene ${n}: "${c}" needs its time as seconds, like "${card} 3s"`);
            return { name: card, seconds, cue, footage: existsSync(join(VIDEO, 'cards', `${card}.html`)) ? null : footage };
        });
        /* `Clip: <name> (notes)` plays clips/<name>.mp4 after the scene's narration. */
        const clipName = /^Clip:\s*([\w.-]+)/im.exec(lines.join('\n'))?.[1] ?? null;
        const clip = clipName ? join(VIDEO, 'clips', clipName.endsWith('.mp4') ? clipName : `${clipName}.mp4`) : null;
        if (clip && !existsSync(clip)) throw new Error(`script.md: scene ${n}: its clip ${clip} is missing (clips/ is not committed: copy it in)`);
        const words = lines.filter((l) => l.startsWith('>')).map((l) => l.replace(/^>\s?/, '')).join(' ').replace(/\s+/g, ' ').trim();
        if (!words) throw new Error(`script.md: scene ${n} has no narration (lines starting with >)`);
        if (cards[0].cue) throw new Error(`script.md: scene ${n}: its first card starts with the scene, so it can't have a cue`);
        const text = words.replace(PAUSE, ' ').replace(/\s+/g, ' ').trim();
        for (const card of cards) {
            if (card.cue && !text.toLowerCase().includes(card.cue.toLowerCase())) throw new Error(`script.md: scene ${n}: "${card.cue}" is not in its narration`);
            if (card.cue && card.seconds !== null) throw new Error(`script.md: scene ${n}: ${card.name} has both a time and a cue; give it one`);
        }
        return { n, title: heading.replace(/^\d+\.\s*/, '').trim(), cards, words, text, clip };
    });
    const twice = scenes.find((s, k) => scenes.findIndex((t) => t.n === s.n) !== k);
    if (twice) throw new Error(`script.md: two scenes are numbered ${twice.n}`);
    return { voice, voicePrompt, voiceSeed, pace, scenes };
}

/* How names are spoken. Only what is sent to the voice changes; the captions keep the script's spelling. */
const spellings = JSON.parse(readFileSync(join(SHARED, 'spellings.json'), 'utf8'));
const spoken = (words) => Object.entries(spellings).reduce((t, [from, to]) => t.replaceAll(from, to), words);

async function voiceScenes({ voice, voicePrompt, voiceSeed, scenes }) {
    const fornax = voice.startsWith('fornax:');
    mkdirSync(join(VIDEO, 'narration'), { recursive: true });
    for (const scene of scenes) {
        const mp3 = join(VIDEO, 'narration', `${scene.n}.mp3`);
        const said = join(VIDEO, 'narration', `${scene.n}.txt`);
        /* A reference clip (`@file`) and its transcript (`file.txt`) are keyed by content, so changing either re-voices. */
        const clip = fornax ? voice.match(/:@(.+)$/)?.[1] : undefined;
        const ref = clip ? resolve(VIDEO, clip) : undefined;
        /* voices/ is not committed: without the clip, --no-voice cuts with the narration on disk. */
        if (ref && !existsSync(ref)) {
            if (args.has('--no-voice') && existsSync(mp3)) continue;
            throw new Error(`scene ${scene.n}: the reference clip ${ref} is missing`);
        }
        const refHash = ref ? ` ref ${createHash('sha256').update(readFileSync(ref)).update(existsSync(`${ref}.txt`) ? readFileSync(`${ref}.txt`) : '').digest('hex').slice(0, 16)}` : '';
        const want = `${fornax ? `${voice}${refHash} seed ${voiceSeed} ${voicePrompt}`.trim() : voice}\n${spoken(scene.words)}`;
        if (existsSync(mp3) && existsSync(said) && readFileSync(said, 'utf8') === want) continue;
        if (args.has('--no-voice')) throw new Error(`scene ${scene.n} changed and --no-voice was given`);
        const key = process.env.DEEPGRAM_API_KEY;
        if (!fornax && !key) throw new Error(`scene ${scene.n} needs voicing: set DEEPGRAM_API_KEY`);
        /* Each stretch between pauses is voiced on its own, then joined with the silence the pause asks for. */
        const pieces = spoken(scene.words).split(new RegExp(PAUSE.source));
        // `fornax:<model>` or `fornax:<model>:<voice>` (a named voice the model ships, e.g. fornax:zonos3:Rupert).
        // A reference clip's path is the video folder's; SpeakBatch runs in the repository's root, so it gets it absolute.
        const [fornaxModel, named = ''] = voice.slice(5).split(':');
        const fornaxVoice = ref ? `@${ref}` : named;
        const parts = fornax ? voiceInFornax(scene, pieces, fornaxModel, voicePrompt, voiceSeed, fornaxVoice) : [];
        for (let k = 0; !fornax && k < pieces.length; k += 2) {
            if (k > 0) parts.push(Number(pieces[k - 1] ?? 0.65) || 0.65);
            if (!pieces[k].trim()) continue;
            // Deepgram clips the last sound it is given (the "U" of "GPU"), so each stretch is voiced with a throwaway
            // word after it and cut at the silence before that word: the real last word keeps its release.
            const response = await fetch(`https://api.deepgram.com/v1/speak?model=${voice}&encoding=linear16&container=wav&sample_rate=24000`, {
                method: 'POST',
                headers: { Authorization: `Token ${key}`, 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: `${pieces[k].trim()} ${SENTINEL}` }),
            });
            if (!response.ok) throw new Error(`Deepgram refused scene ${scene.n}: ${response.status} ${await response.text()}`);
            const raw = join(OUT, `voice-${scene.n}-${k}-raw.wav`);
            writeFileSync(raw, Buffer.from(await response.arrayBuffer()));
            const part = join(OUT, `voice-${scene.n}-${k}.wav`);
            withoutSentinel(raw, part, scene.n);
            parts.push(part);
        }
        if (parts.length === 1) ffmpeg('-i', parts[0], '-c:a', 'libmp3lame', '-q:a', '2', mp3);
        else {
            const inputs = [];
            const labels = [];
            parts.forEach((p, i) => {
                if (typeof p === 'number') inputs.push('-f', 'lavfi', '-t', String(p), '-i', 'anullsrc=r=24000:cl=mono');
                else inputs.push('-i', p);
                labels.push(`[${i}:a]aresample=24000,aformat=channel_layouts=mono[p${i}]`);
            });
            ffmpeg(...inputs, '-filter_complex', `${labels.join(';')};${parts.map((_, i) => `[p${i}]`).join('')}concat=n=${parts.length}:v=0:a=1[a]`, '-map', '[a]', '-c:a', 'libmp3lame', '-q:a', '2', mp3);
        }
        writeFileSync(said, want);
        console.log(`  voiced scene ${scene.n}`);
    }
    for (const scene of scenes) if (scene.cards.some((c) => c.cue)) await timeWords(scene);
}

/*
 * WORD TIMES, for the cues: a scene with cues has its narration transcribed once by Deepgram (narration/<n>.words.json,
 * committed beside the mp3, kept while the mp3 is the same), so a cued card starts on the word, not on an estimate.
 */
const wordsFile = (scene) => join(VIDEO, 'narration', `${scene.n}.words.json`);
async function timeWords(scene) {
    const mp3 = join(VIDEO, 'narration', `${scene.n}.mp3`);
    const said = readFileSync(join(VIDEO, 'narration', `${scene.n}.txt`), 'utf8');
    if (existsSync(wordsFile(scene)) && JSON.parse(readFileSync(wordsFile(scene), 'utf8')).said === said) return;
    const key = process.env.DEEPGRAM_API_KEY;
    if (args.has('--no-voice') || !key) return console.warn(`  note: scene ${scene.n}'s cues are estimated (no word times yet: build once with DEEPGRAM_API_KEY)`);
    const response = await fetch('https://api.deepgram.com/v1/listen?model=nova-3&language=en', {
        method: 'POST',
        headers: { Authorization: `Token ${key}`, 'Content-Type': 'audio/mpeg' },
        body: readFileSync(mp3),
    });
    if (!response.ok) throw new Error(`Deepgram couldn't time scene ${scene.n}'s words: ${response.status} ${await response.text()}`);
    const words = ((await response.json()).results?.channels?.[0]?.alternatives?.[0]?.words ?? []).map((w) => ({ word: w.word, start: w.start, end: w.end }));
    writeFileSync(wordsFile(scene), `${JSON.stringify({ said, words })}\n`);
    console.log(`  timed scene ${scene.n}'s words`);
}

function launch() {
    const path = process.env.CHROME_PATH ?? ['/usr/bin/chromium-browser', '/usr/bin/chromium'].find(existsSync);
    return chromium.launch(path ? { executablePath: path } : { channel: 'chrome' });
}

async function renderCards(browser, names) {
    const dir = join(OUT, 'cards');
    mkdirSync(dir, { recursive: true });
    const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
    for (const card of names) {
        await page.goto(pathToFileURL(join(VIDEO, 'cards', `${card}.html`)).href, { waitUntil: 'load' });
        await page.evaluate(() => document.fonts.ready);
        await page.screenshot({ path: join(dir, `${card}.png`) });
    }
    await page.close();
    return dir;
}

/** A card held for `seconds`, pushed in by 4%: drawn from a 4K (8K for --final) copy, so the push is smooth. */
function cardClip(png, seconds, out) {
    const frames = Math.round(seconds * FPS);
    ffmpeg('-loop', '1', '-i', png, '-vf',
        `scale=${FINAL ? '7680:4320' : '3840:2160'}:flags=lanczos,zoompan=z='1+0.04*on/${frames}':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=${frames}:s=1920x1080:fps=${FPS},format=yuv420p`,
        '-frames:v', String(frames), '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', out);
}

/**
 * Footage as a card: silent, scaled to 1920 × 1080, its first FOOTAGE_SKIP seconds dropped, and sped up (to at most
 * FOOTAGE_MAX_SPEED×) so it plays through in the card's time; what still doesn't fit is cut from the start, so the card
 * ends on the footage's last frame (a result, for a screen recording). A short take holds its last frame.
 */
const FOOTAGE_SKIP = 0.6, FOOTAGE_MAX_SPEED = 3;
function footageCard(mp4, seconds, out) {
    const total = duration(mp4);
    // A take shorter than the skip keeps its last quarter second, rather than seeking past its end to no frames at all.
    const skip = Math.min(FOOTAGE_SKIP, Math.max(0, total - 0.25));
    const length = total - skip;
    const speed = Math.min(FOOTAGE_MAX_SPEED, Math.max(1, length / seconds));
    const start = skip + Math.max(0, length - seconds * speed);
    ffmpeg('-ss', start.toFixed(3), '-i', mp4, '-an', '-vf',
        `setpts=(PTS-STARTPTS)/${speed.toFixed(4)},scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,fps=${FPS},tpad=stop_mode=clone:stop_duration=${seconds},format=yuv420p`,
        '-frames:v', String(Math.round(seconds * FPS)), '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', out);
}

/** Clips joined with a crossfade; every clip but the last is held for the overlap, so each card starts on time. */
function crossfade(files, out) {
    if (files.length === 1) return copyFileSync(files[0], out);
    const inputs = files.flatMap((f) => ['-i', f]);
    let graph = files.map((_, k) => `[${k}:v]fps=${FPS},format=yuv420p,setsar=1${k < files.length - 1 ? `,tpad=stop_mode=clone:stop_duration=${FADE}` : ''}[p${k}];`).join('');
    let offset = 0, prev = 'p0';
    for (let k = 1; k < files.length; k++) {
        offset += duration(files[k - 1]);
        graph += `[${prev}][p${k}]xfade=transition=fade:duration=${FADE}:offset=${offset}[x${k}];`;
        prev = `x${k}`;
    }
    ffmpeg(...inputs, '-filter_complex', graph.slice(0, -1), '-map', `[${prev}]`, '-an', '-c:v', 'libx264', '-crf', '18', out);
}

/*
 * HOW LONG EACH CARD OF A SCENE SHOWS. A cued card starts as its phrase is said: where the phrase falls in the
 * narration's text, as a share of the narration's length, which is right to about half a second. Between two fixed
 * starts (the scene's, a cue's, the scene's end), a card with seconds holds that long and the rest share what is left.
 */
function cardLengths(scene, length, talk, lead = LEAD, squeezed = (t) => t) {
    const n = scene.cards.length;
    const starts = Array(n + 1).fill(null);
    starts[0] = 0;
    starts[n] = length;
    scene.cards.forEach((c, k) => { if (k > 0 && c.cue) starts[k] = lead + cueAt(scene, c.cue, talk, squeezed); });
    const lengths = [];
    for (let from = 0, to; from < n; from = to) {
        for (to = from + 1; starts[to] === null; to++);
        const group = scene.cards.slice(from, to);
        const span = starts[to] - starts[from];
        const fixed = group.reduce((s, c) => s + (c.seconds ?? 0), 0);
        const open = group.filter((c) => c.seconds === null).length;
        if (!open && Math.abs(span - fixed) > 0.05) throw new Error(`scene ${scene.n}: ${group.map((c) => c.name).join(', ')} must fill ${span.toFixed(2)}s up to ${to === n ? 'the end of the scene' : 'the next cue'}, not ${fixed}s; drop the seconds`);
        for (const c of group) lengths.push(c.seconds ?? (span - fixed) / open);
    }
    lengths.forEach((d, k) => {
        if (!(d >= 0.8)) throw new Error(`scene ${scene.n}: ${scene.cards[k].name} would show for ${d.toFixed(2)}s; move its cue or give it room`);
    });
    return lengths;
}

/**
 * When a phrase is said, in seconds from the scene's first word: its share of the spoken characters, as a share of the
 * time spent speaking, plus the pauses before it. Right to about half a second.
 */
function cueAt(scene, cue, talk, squeezed, pace = script.pace) {
    const plain = (w) => w.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (existsSync(wordsFile(scene))) {
        const { said, words } = JSON.parse(readFileSync(wordsFile(scene), 'utf8'));
        /* As written, as sent to the voice ("Embaybel"), or by its last word alone if the first was heard otherwise. */
        const tries = [cue, spoken(cue)].map((c) => c.split(/\s+/).map(plain).filter(Boolean));
        const find = (want) => words.findIndex((_, k) => want.every((w, j) => plain(words[k + j]?.word ?? '') === w));
        let at = tries.map(find).find((k) => k >= 0) ?? -1;
        if (at < 0 && tries[0].length > 1) {
            const rest = find(tries[0].slice(1));
            if (rest > 0) at = rest - 1;
        }
        if (said === readFileSync(join(VIDEO, 'narration', `${scene.n}.txt`), 'utf8') && at >= 0) return squeezed(words[at].start) / pace;
        console.warn(`  note: scene ${scene.n}: "${cue}" not found in its word times; estimated`);
    }
    const pauses = [...scene.words.matchAll(PAUSE)].map((m) => ({ at: m.index, secs: Number(m[1] ?? 0.65) || 0.65 }));
    const silence = pauses.reduce((s, p) => s + p.secs, 0) / pace;
    const index = scene.words.toLowerCase().replace(PAUSE, (m) => ' '.repeat(m.length)).indexOf(cue.toLowerCase());
    const before = pauses.filter((p) => p.at < index).reduce((s, p) => s + p.secs, 0) / pace;
    const chars = (upTo) => scene.words.slice(0, upTo).replace(PAUSE, ' ').length;
    return (talk - silence) * (chars(index) / chars(scene.words.length)) + before;
}

/** The music's landmarks, from the sections.txt beside it: `<seconds> <bar> <label>` a line. */
function sectionsOf(music) {
    const file = music && join(dirname(music), 'sections.txt');
    if (!file || !existsSync(file)) return [];
    return readFileSync(file, 'utf8').split('\n').map((l) => /^\s*([\d.]+)\s+\S+\s+(.*)$/.exec(l)).filter(Boolean).map(([, at, label]) => ({ at: Number(at), label }));
}

/** The video's outro, from script.md's `outro:` line: the first of its paths (in the video's folder) that exists. */
function outroFile() {
    const line = /^outro:\s*(.+)$/m.exec(SCRIPT)?.[1];
    if (!line) return null;
    const found = line.split('|').map((p) => join(VIDEO, p.trim())).find(existsSync);
    if (!found) console.warn(`  note: none of the outro's files is here yet (${line.trim()}): ending on the last card`);
    return found ?? null;
}

/** The silences inside a narration (not before its first word or after its last), at least 0.3 s, as [from, to]. */
function silencesOf(mp3) {
    const length = duration(mp3);
    const report = spawnSync('ffmpeg', ['-nostdin', '-nostats', '-i', mp3, '-af', 'silencedetect=n=-40dB:d=0.3', '-f', 'null', '-'], { encoding: 'utf8', maxBuffer: 1 << 28 }).stderr;
    const starts = [...report.matchAll(/silence_start: ([\d.]+)/g)].map((m) => Number(m[1]));
    const ends = [...report.matchAll(/silence_end: ([\d.]+)/g)].map((m) => Number(m[1]));
    return starts.map((from, k) => [from, ends[k] ?? length]).filter(([from, to]) => from > 0.05 && to < length - 0.05);
}

/*
 * A VOICE IN FORNAX: every stretch of a scene is spoken by fornax-examples' SpeakBatch, with one loaded model at one seed,
 * each stretch led by the voice's description (VoxCPM2's voice design), then trimmed of the silence around it.
 */
function voiceInFornax(scene, pieces, model, prompt, seed, named = '') {
    const fornaxRoot = resolve(VIDEOS, '..');
    const parts = [];
    const lines = [];
    for (let k = 0; k < pieces.length; k += 2) {
        if (k > 0) parts.push(Number(pieces[k - 1] ?? 0.65) || 0.65);
        if (!pieces[k].trim()) continue;
        const raw = join(OUT, `voice-${scene.n}-${k}-raw.wav`);
        lines.push(`${raw}\t${prompt}${pieces[k].trim().replace(/\s+/g, ' ')}`);
        parts.push(raw);
    }
    const tsv = join(OUT, `voice-${scene.n}.tsv`);
    writeFileSync(tsv, `${lines.join('\n')}\n`);
    run(join(fornaxRoot, 'gradlew'), ['-q', '-p', fornaxRoot, 'speakBatch', `--args=${tsv} ${named}`.trim(), `-Pmodel=${model}`, `-Pseed=${seed}`]);
    return parts.map((p) => {
        if (typeof p === 'number') return p;
        const out = p.replace('-raw.wav', '.wav');
        // Trim the silence before and after the words, as Deepgram's are.
        ffmpeg('-i', p, '-af', 'silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.05,areverse,silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.12,areverse', out);
        return out;
    });
}

/* The throwaway word voiced after each stretch (see voicing), and the cut that removes it. */
const SENTINEL = 'Okay.';
function withoutSentinel(raw, out, n) {
    const report = spawnSync('ffmpeg', ['-nostdin', '-nostats', '-i', raw, '-af', 'silencedetect=n=-45dB:d=0.18', '-f', 'null', '-'], { encoding: 'utf8', maxBuffer: 1 << 28 }).stderr;
    /* Only gaps after some speech: a short line ("Introducing Fornax.") can open with a silence that would otherwise be
       taken for the gap before the throwaway word, leaving nothing. */
    const gaps = [...report.matchAll(/silence_start: ([\d.]+)/g)].map((m) => Number(m[1])).filter((t) => t > 0.35);
    const length = duration(raw);
    const cut = gaps.filter((t) => t < length - 0.2).pop();
    if (cut === undefined || length - cut > 1.6) throw new Error(`scene ${n}: no gap before the throwaway word in ${raw}; voice it again`);
    ffmpeg('-i', raw, '-af', `atrim=end=${(cut + 0.12).toFixed(3)},afade=t=out:st=${(cut + 0.07).toFixed(3)}:d=0.05`, out);
}

/*
 * SQUEEZING A NARRATION: each silence inside it keeps KEEP seconds and `k` of the rest (k = 1 leaves it as voiced, 0
 * keeps only KEEP), cut from its middle. The words are untouched.
 */
const KEEP = 0.3;
function squeeze(silences, k) {
    return silences.map(([from, to]) => {
        const keep = KEEP + (to - from - KEEP) * k;
        return [from + keep / 2, to - keep / 2];
    }).filter(([from, to]) => to - from > 0.01);
}
/** A time in the narration as voiced, in the squeezed narration. */
const squeezer = (cuts) => (t) => t - cuts.reduce((s, [from, to]) => s + Math.max(0, Math.min(t, to) - from), 0);

const frame = (t) => Math.round(t * FPS) / FPS;

/*
 * THE CUT'S TIMING: when each scene starts and speaks, where its clip plays, and where the outro is.
 *
 * MUSIC SYNC. When the music's sections.txt has a "duck" section (rests written for the clip) and a scene has a clip,
 * the music is not moved or stretched; the cut is fitted to it. The clip starts as the duck does, the next scene's
 * first word comes with the section after it (the close), and the outro starts at the music's "final cadence" (or
 * once the last word has had a moment). The scenes before the clip are fitted into the time before the duck by their
 * gaps: the silences between their sentences (squeeze) and the gaps between scenes, both shortened by one factor
 * found to fit; if they're too short, the gaps grow.
 */
function timingOf({ scenes, pace }) {
    const music = musicFile();
    const marks = sectionsOf(music);
    const duckAt = marks.findIndex((m) => /^duck\b/i.test(m.label));
    const duck = marks[duckAt], close = marks[duckAt + 1];
    const cadence = marks.find((m) => /final cadence/i.test(m.label));
    const clipScene = scenes.findIndex((s) => s.clip);
    const sync = !!(duck && close && clipScene >= 0);
    const outro = outroFile();
    const voiced = scenes.map((scene) => {
        const mp3 = join(VIDEO, 'narration', `${scene.n}.mp3`);
        return { scene, mp3, raw: duration(mp3), silences: silencesOf(mp3) };
    });
    const fitted = (i) => sync && i <= clipScene;
    const leadAt = (i, k) => (i === 0 ? OPENING : fitted(i) ? 0.2 + (LEAD - 0.2) * k : LEAD);
    const tailAt = (i, k) => (fitted(i) ? 0.3 + (TAIL - 0.3) * k : TAIL);
    const talkAt = (v, k) => (v.raw - squeeze(v.silences, k).reduce((s, [a, b]) => s + b - a, 0)) / pace;
    const before = (k) => voiced.slice(0, clipScene + 1).reduce((s, v, i) => s + leadAt(i, k) + talkAt(v, k) + tailAt(i, k), 0);
    let k = 1, extra = 0;
    if (sync) {
        if (before(1) <= duck.at) extra = (duck.at - before(1)) / (clipScene + 1);
        else if (before(0) > duck.at) {
            k = 0;
            console.warn(`  warning: the scenes before the clip need ${before(0).toFixed(1)}s even squeezed, and the music's duck is at ${duck.at}s: the clip comes late`);
        } else {
            let lo = 0, hi = 1;
            for (let n = 0; n < 40; n++) { const mid = (lo + hi) / 2; if (before(mid) > duck.at) hi = mid; else lo = mid; }
            k = lo;
        }
        console.log(`  synced to the music: the clip at its duck (${duck.at}s), scene ${scenes[clipScene + 1]?.n ?? '-'} at its close (${close.at}s); gaps before the clip at ${(k * 100).toFixed(0)}%${extra ? ` plus ${extra.toFixed(2)}s each` : ''}`);
    }
    const timeline = [];
    const clips = [];
    let at = 0;
    for (const [i, v] of voiced.entries()) {
        const kk = fitted(i) ? k : 1;
        const cuts = squeeze(v.silences, kk);
        const talk = talkAt(v, kk);
        let lead = leadAt(i, kk);
        if (sync && i === clipScene + 1) lead = Math.max(0.1, close.at - at);
        const last = i === voiced.length - 1;
        let length = lead + talk + (fitted(i) ? tailAt(i, kk) + extra : last ? (outro ? OUTRO_BREATH : END) : TAIL);
        if (sync && i === clipScene) length = duck.at - at;
        if (last && outro && cadence && at + length < cadence.at) length = cadence.at - at;
        length = frame(at + length) - at;
        timeline.push({ scene: v.scene, at, start: at + lead, talk, lead, length, mp3: v.mp3, raw: v.raw, cuts, squeezed: squeezer(cuts) });
        at += length;
        if (v.scene.clip) {
            const d = frame(duration(v.scene.clip));
            clips.push({ start: at, end: at + d, file: v.scene.clip });
            at += d;
        }
    }
    const tail = outro ? { start: at, end: at + frame(duration(outro)), file: outro } : null;
    return { timeline, clips, outro: tail, total: tail ? tail.end : at, sync };
}

/**
 * Footage, as a part of the cut, its sound levelled to `lufs`, faded in and out: a clip pillarboxed into 1920 × 1080,
 * or with `fill` scaled to cover it (an outro a little off 16:9).
 */
function footagePart(file, out, { lufs = CLIP_LUFS, fill = false } = {}) {
    const d = frame(duration(file));
    const sound = hasAudio(file)
        ? `[0:a]aresample=48000,aformat=channel_layouts=stereo,volume=${(lufs - loudnessOf(file)).toFixed(2)}dB,afade=t=in:d=0.08,afade=t=out:st=${d - 0.25}:d=0.25,apad=whole_dur=${d}[a]`
        : `anullsrc=r=48000:cl=stereo:d=${d}[a]`;
    const fit = fill
        ? 'scale=1920:1080:force_original_aspect_ratio=increase:flags=lanczos,crop=1920:1080'
        : 'scale=1920:1080:force_original_aspect_ratio=decrease:flags=lanczos,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=0x170d08';
    ffmpeg('-i', file, '-filter_complex',
        `[0:v]${fit},fps=${FPS},format=yuv420p,setsar=1,tpad=stop_mode=clone:stop_duration=1,trim=duration=${d},fade=t=in:st=0:d=0.3,fade=t=out:st=${d - 0.3}:d=0.3[v];${sound}`,
        '-map', '[v]', '-map', '[a]', '-t', String(d),
        '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-r', String(FPS),
        '-c:a', 'aac', '-ar', '48000', '-ac', '2', '-b:a', '192k', out);
}

async function cutScenes(script, cut, cards) {
    const { pace } = script;
    const parts = [];
    for (const [index, t] of cut.timeline.entries()) {
        const { scene, lead, length, talk, mp3, raw, cuts } = t;
        const lengths = cardLengths(scene, length, talk, lead, t.squeezed);
        let from = 0;
        console.log(`  ${scene.n}: ${scene.cards.map((c, k) => { const at = from; from += lengths[k]; return `${c.name} ${at.toFixed(1)}s`; }).join(', ')}`);
        const clips = scene.cards.map((c, k) => {
            const file = join(OUT, `s${scene.n}-${k}.mp4`);
            if (c.footage) footageCard(c.footage, lengths[k], file);
            else cardClip(join(cards, `${c.name}.png`), lengths[k], file);
            return file;
        });
        const pictures = join(OUT, `s${scene.n}-v.mp4`);
        crossfade(clips, pictures);
        /* The narration as kept: everything between the squeezed silences' cuts. */
        const keep = [];
        let kept = 0;
        for (const [a, b] of cuts) { keep.push([kept, a]); kept = b; }
        keep.push([kept, raw]);
        const voice = `${keep.map(([a, b], k) => `[1:a]atrim=${a.toFixed(4)}:${b.toFixed(4)},asetpts=PTS-STARTPTS[k${k}];`).join('')}${keep.map((_, k) => `[k${k}]`).join('')}concat=n=${keep.length}:v=0:a=1,`;
        const lastScene = index === cut.timeline.length - 1;
        const fadeOut = lastScene && !cut.outro ? 1.2 : 0.3;
        const part = join(OUT, `s${scene.n}.mp4`);
        ffmpeg('-i', pictures, '-i', mp3, '-filter_complex',
            `[0:v]tpad=stop_mode=clone:stop_duration=${length},trim=duration=${length},setpts=PTS-STARTPTS,fade=t=in:st=0:d=0.3,fade=t=out:st=${length - fadeOut}:d=${fadeOut}[v];` +
            /* Levelled by a fixed gain from its measured loudness: loudnorm's dynamic levelling pumped the voice. */
            `${voice}aresample=48000,${pace === 1 ? '' : `atempo=${pace},`}volume=${(-16 - loudnessOf(mp3)).toFixed(2)}dB,aformat=channel_layouts=mono[n];` +
            `anullsrc=r=48000:cl=mono:d=${lead}[l];[l][n]concat=n=2:v=0:a=1,apad=whole_dur=${length},atrim=0:${length}[a]`,
            '-map', '[v]', '-map', '[a]', '-t', String(length),
            '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-r', String(FPS),
            '-c:a', 'aac', '-ar', '48000', '-ac', '2', '-b:a', '192k', part);
        parts.push(part);
        if (scene.clip) {
            const clipped = join(OUT, `s${scene.n}-clip.mp4`);
            footagePart(scene.clip, clipped);
            parts.push(clipped);
        }
    }
    if (cut.outro) {
        const outro = join(OUT, 'outro.mp4');
        footagePart(cut.outro.file, outro, { lufs: OUTRO_LUFS, fill: true });
        parts.push(outro);
    }
    const joined = join(OUT, 'joined.mp4');
    ffmpeg(...parts.flatMap((p) => ['-i', p]), '-filter_complex', `${parts.map((_, k) => `[${k}:v][${k}:a]`).join('')}concat=n=${parts.length}:v=1:a=1[v][a]`,
        '-map', '[v]', '-map', '[a]', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', joined);
    const drift = duration(joined) - cut.total;
    if (Math.abs(drift) > 0.1) console.warn(`  warning: the cut is ${drift.toFixed(2)}s off its timing; the music may not land`);
    return joined;
}

/** A file's highest sample, in dBFS, as decoded. */
function peakOf(file) {
    const report = spawnSync('ffmpeg', ['-nostats', '-i', file, '-af', 'volumedetect', '-f', 'null', '-'], { encoding: 'utf8', maxBuffer: 1 << 28 }).stderr;
    return Number(/max_volume: (-?[\d.]+) dB/.exec(report)?.[1] ?? 0);
}

/** A file's integrated loudness, in LUFS. */
function loudnessOf(file) {
    const measured = spawnSync('ffmpeg', ['-nostats', '-i', file, '-af', 'ebur128', '-f', 'null', '-'], { encoding: 'utf8', maxBuffer: 1 << 28 }).stderr;
    const loudness = Number([...measured.matchAll(/\bI:\s*(-?[\d.]+) LUFS/g)].at(-1)?.[1]);
    if (!Number.isFinite(loudness)) throw new Error(`could not measure the loudness of ${file}`);
    return loudness;
}

/*
 * THE MUSIC is rendered elsewhere (for the explainer, with Muse Sounds) into music/out/: the longest audio file there
 * is the bed. It starts with the cut. If it runs long it fades out at the cut's end; if it runs short the build says so.
 */
function musicFile() {
    const dir = join(VIDEO, 'music', 'out');
    if (!existsSync(dir)) return null;
    /* The longest is the bed (beside it may be shorter cues and stems); of two as long, the lossless one. */
    const audio = readdirSync(dir).filter((f) => /\.(wav|flac|mp3|ogg|m4a)$/i.test(f)).map((f) => ({ file: join(dir, f), secs: duration(join(dir, f)), lossless: /\.(wav|flac)$/i.test(f) }));
    return audio.sort((a, b) => b.secs - a.secs > 0.05 || b.secs - a.secs < -0.05 ? b.secs - a.secs : b.lossless - a.lossless)[0]?.file ?? null;
}

/** A stretch of a file's integrated loudness, in LUFS; -70 for silence. */
function loudnessBetween(file, from, to) {
    const measured = spawnSync('ffmpeg', ['-nostdin', '-nostats', '-ss', String(from), '-to', String(to), '-i', file, '-af', 'ebur128', '-f', 'null', '-'], { encoding: 'utf8', maxBuffer: 1 << 28 }).stderr;
    const loudness = Number([...measured.matchAll(/\bI:\s*(-?[\d.]+) LUFS/g)].at(-1)?.[1]);
    return Number.isFinite(loudness) ? loudness : -70;
}

/* Where the music sits, in LUFS: under the voice (-16 LUFS), and on its own after the last word. */
const UNDER_LUFS = -29, END_LUFS = -20;

/**
 * The music's gain over the cut, in dB, linear between points. Each scene's stretch of the music is measured and set
 * to sit at UNDER_LUFS under the voice, so a loud passage (the opening flourish) and a quiet one (a piano bar) sit
 * alike; further under a clip (whose own sound carries it); at END_LUFS after the last word; then out.
 */
function musicCurve(music, { timeline, clips, total }) {
    const last = timeline.at(-1);
    const spoken = last.start + last.talk;
    const gain = (from, to, target) => {
        const level = loudnessBetween(music, from, to);
        return level <= -60 ? 0 : Math.max(-24, Math.min(14, target - level));
    };
    const points = [];
    timeline.forEach((t, k) => {
        const clip = clips.find((c) => c.start >= t.at && c.start < (timeline[k + 1]?.at ?? total));
        const to = k === timeline.length - 1 ? spoken : (clip?.start ?? timeline[k + 1].at);
        const g = gain(t.start, to, UNDER_LUFS);
        points.push([t.at + (k ? 0.6 : 0), g], [to - 0.4, g]);
        if (clip) points.push([clip.start + 0.2, g + CLIP_DUCK], [clip.end - 0.4, g + CLIP_DUCK]);
    });
    const g = gain(spoken + 0.2, Math.max(spoken + 1, total - 1), END_LUFS);
    points.push([spoken + 1.4, g], [total - 2, g], [total, g - 40]);
    points.sort((a, b) => a[0] - b[0]);
    let expr = `${points.at(-1)[1].toFixed(2)}`;
    for (let k = points.length - 2; k >= 0; k--) {
        const [t0, d0] = points[k];
        const [t1, d1] = points[k + 1];
        const seg = t1 > t0 ? `${d0.toFixed(2)}+(t-${t0.toFixed(3)})*${((d1 - d0) / (t1 - t0)).toFixed(4)}` : `${d1.toFixed(2)}`;
        expr = `if(lt(t,${t1.toFixed(3)}),${seg},${expr})`;
    }
    return expr;
}

/** The bed as it sits under the cut: fitted, levelled by the curve, as long as the cut. */
function bed(music, cut, out) {
    const length = duration(music);
    if (length < cut.total - 0.5) console.warn(`  warning: the music is ${length.toFixed(1)}s, the cut ${cut.total.toFixed(1)}s: the end is without music`);
    ffmpeg('-i', music, '-af',
        `aresample=48000,aformat=channel_layouts=stereo,volume=eval=frame:volume='pow(10,(${musicCurve(music, cut)})/20)',apad=whole_dur=${cut.total},atrim=0:${cut.total}`,
        '-c:a', 'pcm_s16le', out);
}

function mix(joined, cut) {
    const music = musicFile();
    const master = join(OUT_ROOT, `${name}.mp4`);
    let input = joined;
    if (music) {
        const under = join(OUT, 'music.wav');
        bed(music, cut, under);
        const mixed = join(OUT, 'mixed.mov');
        ffmpeg('-i', joined, '-i', under, '-filter_complex',
            /* Ducked: a few dB more off the music while a word (or the clip's sound) plays, back within half a second. */
            (DUCK
                ? '[0:a]aresample=48000,aformat=channel_layouts=stereo,asplit[v][key];[1:a]aformat=channel_layouts=stereo[mf];[mf][key]sidechaincompress=threshold=0.03:ratio=4:attack=40:release=500[m];'
                : '[0:a]aresample=48000[v];[1:a]anull[m];') +
            '[v][m]amix=inputs=2:normalize=0:duration=first[a]',
            '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'pcm_s16le', mixed);
        input = mixed;
    }
    /*
     * -16 LUFS, under a -3 dB ceiling: AAC overshoots the limiter by a dB or two. The encoded peak is measured, and if
     * it is still above -1 dBFS the whole is turned down by the excess and encoded again.
     */
    let trim = 0;
    const loudness = loudnessOf(input);
    for (let attempt = 0; ; attempt++) {
        ffmpeg('-i', input, '-af', `volume=${(-16 - loudness - trim).toFixed(2)}dB,alimiter=limit=0.708:attack=5:release=60:level=false`,
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-movflags', '+faststart', master);
        const peak = peakOf(master);
        if (peak <= -1) break;
        if (attempt === 2) { console.warn(`  warning: the encoded peak is still ${peak} dBFS after turning it down ${trim.toFixed(1)} dB`); break; }
        trim += peak + 1.2;
    }
    return { master, music };
}

/** WebVTT: each sentence timed across its scene's narration by its share of the characters. */
function captions(timeline) {
    const sentences = [];
    for (const { scene, start, talk } of timeline) {
        const said = scene.text.match(/[^.!?]+[.!?]+["')]*|\S[^.!?]*$/g).map((s) => s.trim());
        const total = said.reduce((n, s) => n + s.length, 0);
        let t = start;
        for (const sentence of said) {
            const d = (talk * sentence.length) / total;
            sentences.push([t, t + d, sentence]);
            t += d;
        }
    }
    return webVtt(paced(sentences));
}

/** What the music is fitted to, for whoever composes it: out/timing.json. */
function writeTiming(cut) {
    const last = cut.timeline.at(-1);
    const timing = {
        total: +cut.total.toFixed(3),
        spokenEnd: +(last.start + last.talk).toFixed(3),
        scenes: cut.timeline.map(({ scene, at, start, talk }) => ({ n: Number(scene.n), title: scene.title, at: +at.toFixed(3), speechStart: +start.toFixed(3), speechEnd: +(start + talk).toFixed(3) })),
        clips: cut.clips.map((c) => ({ start: +c.start.toFixed(3), end: +c.end.toFixed(3) })),
        outro: cut.outro && { start: +cut.outro.start.toFixed(3), end: +cut.outro.end.toFixed(3) },
    };
    writeFileSync(join(OUT_ROOT, 'timing.json'), `${JSON.stringify(timing, null, 2)}\n`);
}

const script = readScript();
const names = [...new Set(script.scenes.flatMap((s) => s.cards.filter((c) => !c.footage).map((c) => c.name)))];

if (args.has('--music')) {
    mkdirSync(OUT, { recursive: true });
    await voiceScenes(script);
    const cut = timingOf(script);
    writeTiming(cut);
    const music = musicFile();
    if (!music) throw new Error(`videos/${name}/music/out/ has no audio yet`);
    bed(music, cut, join(OUT, 'music.wav'));
    console.log(`  music: ${join(OUT, 'music.wav')} (from ${music}, fitted to the cut's ${cut.total.toFixed(1)}s)`);
    process.exit(0);
}

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });
const browser = await launch();
let cards;
try {
    cards = await renderCards(browser, names);
} finally {
    await browser.close();
}
if (args.has('--cards')) {
    console.log(`cards rendered to ${cards}`);
    process.exit(0);
}
await voiceScenes(script);
const cut = timingOf(script);
writeTiming(cut);
const joined = await cutScenes(script, cut, cards);
const { master, music } = mix(joined, cut);
writeFileSync(join(OUT_ROOT, `${name}.vtt`), captions(cut.timeline));
/* One frame from the middle of each scene's speech, and one from each clip, to look at. */
const frames = join(OUT_ROOT, 'frames');
rmSync(frames, { recursive: true, force: true });
mkdirSync(frames, { recursive: true });
for (const { scene, start, talk } of cut.timeline) ffmpeg('-ss', (start + talk / 2).toFixed(2), '-i', master, '-frames:v', '1', '-q:v', '3', join(frames, `scene-${scene.n}.jpg`));
cut.clips.forEach((c, k) => ffmpeg('-ss', ((c.start + c.end) / 2).toFixed(2), '-i', master, '-frames:v', '1', '-q:v', '3', join(frames, `clip-${k + 1}.jpg`)));
if (cut.outro) ffmpeg('-ss', (cut.outro.start + (cut.outro.end - cut.outro.start) * 0.7).toFixed(2), '-i', master, '-frames:v', '1', '-q:v', '3', join(frames, 'outro.jpg'));

const seconds = duration(master);
if (script.pace !== 1) console.log(`  narration at pace ${script.pace}`);
for (const { scene, start, talk } of cut.timeline) console.log(`  ${scene.n}. ${scene.title.padEnd(14)} ${start.toFixed(2)}s, ${talk.toFixed(1)}s of voice`);
for (const c of cut.clips) console.log(`  clip           ${c.start.toFixed(2)}s to ${c.end.toFixed(2)}s`);
if (cut.outro) console.log(`  outro          ${cut.outro.start.toFixed(2)}s to ${cut.outro.end.toFixed(2)}s (${cut.outro.file})`);
console.log(`  ${music ? `with ${music}` : 'no music yet (music/out/ has no audio): rebuild when it lands'}`);
console.log(`  ${master}: ${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')} (${seconds.toFixed(1)}s), ${FINAL ? 'final, 60fps' : 'draft, 30fps'}`);
