# Kiln website

The landing page for Kiln, built with [Astro](https://astro.build).

```sh
npm install
npm run dev       # http://localhost:4321
npm run build     # static site in dist/
npm run preview
```

- `src/pages/index.astro` is the whole page: content, styles and the little script for tabs and copy buttons.
- `src/assets/splash.jpg` and `src/assets/logo.png` were generated with OpenAI's image API (gpt-image-2.6-sunburst).
  The favicons in `public/` are the logo, resized. Alternates live in `art-candidates/` (not committed).
- `src/assets/cover.png` and `src/lyrics.txt` are the real output of `./gradlew single`, unretouched.
- Every number on the page comes from the repository's README or from `out/bench/results.md`. Keep it that way.
