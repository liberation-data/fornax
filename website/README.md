# Fornax website

The landing page and the docs for Fornax, built with [Astro](https://astro.build) and, for the docs,
[Starlight](https://starlight.astro.build).

```sh
npm install
npm run dev       # http://localhost:4321
npm run build     # static site in dist/
npm run preview
```

- `src/pages/index.astro` is the whole landing page: content, styles and the little script for tabs and copy buttons.
- The docs are Markdown in `src/content/docs/docs/`, served at `/docs/`; the sidebar is in `astro.config.mjs`.
  `src/styles/docs.css` gives Starlight the landing page's palette, and `src/components/docs/` holds its overrides:
  the fornax4j wordmark in the header, Share beside each page's title, and the share card as each page's `og:image`.
  Search (Pagefind) is built with the site, so it works in `npm run build` and `npm run preview`, not `npm run dev`.
- Every Java sample in the docs compiles against the real API; keep it that way when the API changes. Pages for
  work not yet on `main` start with a `:::note` saying so.
- `src/assets/splash.jpg` and `src/assets/logo.png` were generated with OpenAI's image API (gpt-image-2.6-sunburst).
  The favicons in `public/` are the logo, resized. Alternates live in `art-candidates/` (not committed).
- `src/assets/cover.png` and `src/lyrics.txt` are the real output of `./gradlew single`, unretouched.
- Every number on the page comes from the repository's README or from `out/bench/results.md`. Keep it that way.
