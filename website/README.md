# Fornax website

The landing page and the docs for Fornax, built with [Astro](https://astro.build) and, for the docs,
[Starlight](https://starlight.astro.build).

```sh
npm install
npm run dev       # http://localhost:4321
npm run build     # static site in dist/
npm run preview
npm test          # src/lib's tests (node:test)
```

- `src/pages/index.astro` is the whole landing page: content, styles and the little script for tabs and copy buttons.
- The docs are Markdown in `src/content/docs/docs/`, served at `/docs/`; the sidebar is in `astro.config.mjs`.
  `src/styles/docs.css` gives Starlight the landing page's palette, and `src/components/docs/` holds its overrides:
  the fornax4j wordmark in the header, Share beside each page's title, and the share card as each page's `og:image`.
  Search (Pagefind) is built with the site, so it works in `npm run build` and `npm run preview`, not `npm run dev`.
- Every Java sample in the docs is copied from compiled code in `fornax-examples/src/snippets/`: edit it there and run
  `./gradlew syncSnippets` (see [CONTRIBUTING.md](../CONTRIBUTING.md)). Pages for
  work not yet on `main` start with a `:::note` saying so.
- `src/assets/splash.jpg` and `src/assets/logo.png` were generated with OpenAI's image API (gpt-image-2.6-sunburst).
  The favicons in `public/` are the logo, resized. Alternates live in `art-candidates/` (not committed).
- `src/lib/github.ts` reads Fornax Studio's latest release and the repos' stars from GitHub's API while the site builds
  (with `GITHUB_TOKEN` if it's set), so a visitor's browser never calls GitHub. Without a release, or without GitHub,
  the page says the downloads are on their way. `.github/workflows/website.yml` rebuilds the site daily so a new
  release shows up; `npm test` runs `src/lib/github.test.ts` (node:test) against each answer GitHub can give.
- `scripts/video.sh` makes the explainer's web copy, HLS segments, poster, end frame and captions in `public/video/`
  from the video build's mp4; the page reads its length from the playlist, and serves the poster as AVIF, WebP and
  JPEG at the player's widths (`getImage`, as for the splash), lazily.
- Every number on the page comes from the repository's README or from `out/bench/results.md`. Keep it that way.
