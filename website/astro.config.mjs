// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// The landing page is src/pages/index.astro; the docs are Starlight's, under /docs/ (src/content/docs/docs/), as
// worlds.embabel.com's guide is. `site` makes canonical and og:image URLs absolute.
export default defineConfig({
    site: 'https://fornax4j.org',
    integrations: [
        starlight({
            title: 'fornax4j',
            description: 'How to call open models from Java with Fornax: language models, vision, speech, images, video and songs, in-process on your GPU.',
            logo: { src: './src/assets/logo.png', alt: 'Fornax' },
            favicon: '/favicon-64.png',
            social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/liberation-data/fornax' }],
            editLink: { baseUrl: 'https://github.com/liberation-data/fornax/edit/main/website/' },
            head: [
                { tag: 'link', attrs: { rel: 'icon', href: '/favicon.ico', sizes: '32x32' } },
                { tag: 'link', attrs: { rel: 'apple-touch-icon', href: '/apple-touch-icon.png' } },
                { tag: 'meta', attrs: { name: 'twitter:card', content: 'summary_large_image' } }
            ],
            components: {
                // The fornax4j wordmark, as on the landing page, beside the logo.
                SiteTitle: './src/components/docs/SiteTitle.astro',
                // Each page's title, with Share beside it.
                PageTitle: './src/components/docs/PageTitle.astro',
                // Starlight's head, plus the site's share card as each page's og:image.
                Head: './src/components/docs/Head.astro'
            },
            expressiveCode: {
                themes: ['vitesse-dark', 'vitesse-light'],
                // The page's warm browns behind the code, rather than the themes' own grey and white.
                styleOverrides: {
                    borderRadius: '8px',
                    borderColor: ({ theme }) => (theme.type === 'dark' ? '#45321f' : '#ecdfd3'),
                    codeBackground: ({ theme }) => (theme.type === 'dark' ? '#1d110b' : '#fffdfa'),
                    codeFontFamily: "'JetBrains Mono', ui-monospace, monospace"
                }
            },
            customCss: [
                '@fontsource/ubuntu/latin-400.css',
                '@fontsource/ubuntu/latin-500.css',
                '@fontsource/ubuntu/latin-700.css',
                '@fontsource/noto-serif/latin-400-italic.css',
                '@fontsource/jetbrains-mono/latin-400.css',
                '@fontsource/jetbrains-mono/latin-500.css',
                './src/styles/docs.css'
            ],
            sidebar: [
                {
                    label: 'Start here',
                    items: [
                        { label: 'Introduction', link: '/docs/' },
                        { slug: 'docs/getting-started' },
                        'docs/platforms',
                        'docs/gpu-drivers',
                        'docs/maven-and-gradle'
                    ]
                },
                {
                    label: 'Guides',
                    items: [
                        { slug: 'docs/guides/language-models' },
                        'docs/guides/vision',
                        'docs/guides/embeddings',
                        { slug: 'docs/guides/speech' },
                        { slug: 'docs/guides/speech-to-text' },
                        { slug: 'docs/guides/stem-separation' },
                        { slug: 'docs/guides/images' },
                        { slug: 'docs/guides/video' },
                        { slug: 'docs/guides/songs' },
                        { slug: 'docs/guides/models' },
                        { slug: 'docs/guides/cli', badge: { text: 'Next', variant: 'default' } },
                        { slug: 'docs/guides/openai-server', badge: { text: 'Next', variant: 'default' } },
                        { slug: 'docs/guides/model-pool' },
                        { slug: 'docs/guides/threading', badge: { text: 'Next', variant: 'default' } },
                        { slug: 'docs/guides/isolation', badge: { text: 'Next', variant: 'default' } },
                        { slug: 'docs/guides/dispatch', badge: { text: 'Next', variant: 'default' } },
                        { slug: 'docs/guides/gpu-backends' }
                    ]
                },
                { label: 'Benchmarks', items: [{ slug: 'docs/benchmarks' }] },
                { label: 'Troubleshooting', items: [{ slug: 'docs/troubleshooting' }] },
                { label: 'Reference', items: [{ slug: 'docs/reference' }] }
            ]
        })
    ]
});
