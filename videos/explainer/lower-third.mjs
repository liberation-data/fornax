// Renders the clip's lower-third ("endlesscountry.com · made with Fornax") as a transparent 1920×1080 PNG, in the cards'
// fonts, and burns it into the clip: clips/endless-country-ep2.mp4 → clips/endless-country-ep2-titled.mp4.
//   node explainer/lower-third.mjs        (from videos/)
import { execFileSync } from 'node:child_process';
import { existsSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright-core';

const here = dirname(fileURLToPath(import.meta.url));
const css = pathToFileURL(join(here, '../shared/cards/card.css')).href;
const html = `<!doctype html><html><head><link rel="stylesheet" href="${css}"><style>
html, body { margin: 0; width: 1920px; height: 1080px; background: transparent !important; }
.lt { position: absolute; left: 236px; bottom: 84px; padding: 18px 30px; border-radius: 14px;
      background: rgba(18, 8, 3, 0.62); border: 1px solid rgba(255, 255, 255, 0.16); color: #fff;
      font: 700 40px Ub, sans-serif; letter-spacing: 0.01em; }
.lt em { font: italic 400 42px Serif, serif; color: #f39a3d; }
</style></head><body><div class="lt">endlesscountry.com · <em>made with Fornax</em></div></body></html>`;

const path = process.env.CHROME_PATH ?? ['/usr/bin/chromium-browser', '/usr/bin/chromium'].find(existsSync);
const browser = await chromium.launch(path ? { executablePath: path } : { channel: 'chrome' });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
// A file page, so the stylesheet's relative font URLs load (setContent's about:blank can't read file: URLs).
const htmlFile = join(here, 'clips', 'lower-third.html');
writeFileSync(htmlFile, html);
await page.goto(pathToFileURL(htmlFile).href, { waitUntil: 'networkidle' });
await page.evaluate(async () => { await document.fonts.load('700 40px Ub'); await document.fonts.ready; });
const png = join(here, 'clips', 'lower-third.png');
await page.screenshot({ path: png, omitBackground: true });
await browser.close();

// In after 0.6 s, out 0.8 s before the clip ends; the clip is pillarboxed to 1920×1080 first.
const src = join(here, 'clips', 'endless-country-ep2.mp4'), out = join(here, 'clips', 'endless-country-ep2-titled.mp4');
const len = Number(execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', src]).toString());
execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', src, '-loop', '1', '-i', png, '-filter_complex',
    `[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1[v];` +
    `[1:v]format=rgba,fade=t=in:st=0.6:d=0.4:alpha=1,fade=t=out:st=${(len - 1.2).toFixed(2)}:d=0.4:alpha=1[t];` +
    `[v][t]overlay=shortest=1[o]`, '-map', '[o]', '-map', '0:a', '-c:v', 'libx264', '-crf', '17', '-c:a', 'copy', out]);
console.log(`wrote ${out}`);
