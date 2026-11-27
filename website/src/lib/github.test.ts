/**
 * What the page is given for each answer GitHub can give at build time: a public repository with installers, a private
 * one, no release, a release without installers, and no GitHub at all. Run with `npm test` (node:test; Node strips the
 * types). Each test imports github.ts afresh, since it asks GitHub once per build and keeps the answer.
 */
import { afterEach, test } from 'node:test';
import assert from 'node:assert/strict';

const realFetch = globalThis.fetch;
afterEach(() => {
    globalThis.fetch = realFetch;
    delete process.env.GITHUB_TOKEN;
});

let fresh = 0;
const load = (): Promise<typeof import('./github.ts')> => import(`./github.ts?fresh=${fresh++}`);

/** GitHub's API as `routes` (path to status and JSON body) answers it, recording each path asked for and its headers. */
const github = (routes: Record<string, [number, unknown]>) => {
    const asked: { path: string; headers: Record<string, string> }[] = [];
    globalThis.fetch = (async (input: string | URL, init?: RequestInit) => {
        const path = String(input).replace('https://api.github.com/', '');
        asked.push({ path, headers: (init?.headers ?? {}) as Record<string, string> });
        const [status, body] = routes[path] ?? [404, { message: 'Not Found' }];
        return new Response(JSON.stringify(body), { status });
    }) as typeof fetch;
    return asked;
};

const asset = (name: string, size = 123_400_000) => ({ name, size, browser_download_url: `https://example.test/${name}` });
const PUBLIC = [200, { private: false, stargazers_count: 42 }] as [number, unknown];

test('a public repository with a release offers each platform its installer', async () => {
    github({
        'repos/liberation-data/fornax-studio': PUBLIC,
        'repos/liberation-data/fornax-studio/releases/latest': [200, {
            tag_name: 'v0.3.1',
            html_url: 'https://github.com/liberation-data/fornax-studio/releases/tag/v0.3.1',
            assets: [asset('fornax-studio-mac-arm64.dmg'), asset('fornax-studio-mac-x64.dmg'), asset('fornax-studio.deb'), asset('fornax-studio.rpm'), asset('latest.yml')],
        }],
    });
    const { studioRelease } = await load();
    const studio = await studioRelease();
    assert.equal(studio?.version, '0.3.1');
    assert.equal(studio?.mac.appleSilicon?.url, 'https://example.test/fornax-studio-mac-arm64.dmg');
    assert.equal('intel' in (studio?.mac ?? {}), false, 'an x64 dmg is not offered: no Intel Macs (#123)');
    assert.equal(studio?.mac.appleSilicon?.size, '123 MB');
    assert.deepEqual(studio?.linux.map((d) => d.label), ['Debian / Ubuntu', 'Fedora / RHEL']);
});

test('a private repository offers nothing, and its releases are never asked for', async () => {
    const asked = github({ 'repos/liberation-data/fornax-studio': [200, { private: true }] });
    const { studioRelease, studioRepoPublic } = await load();
    assert.equal(await studioRepoPublic(), false);
    assert.equal(await studioRelease(), undefined);
    assert.ok(!asked.some((it) => it.path.includes('releases')));
});

test('a private repository this build cannot see (GitHub answers 404) offers nothing', async () => {
    const asked = github({});
    const { studioRelease, studioRepoPublic } = await load();
    assert.equal(await studioRepoPublic(), false);
    assert.equal(await studioRelease(), undefined);
    assert.ok(!asked.some((it) => it.path.includes('releases')));
});

test('a public repository with no release yet offers nothing', async () => {
    github({ 'repos/liberation-data/fornax-studio': PUBLIC });
    const { studioRelease, studioRepoPublic } = await load();
    assert.equal(await studioRepoPublic(), true);
    assert.equal(await studioRelease(), undefined);
});

test('a release without an installer for any platform offers nothing', async () => {
    github({
        'repos/liberation-data/fornax-studio': PUBLIC,
        'repos/liberation-data/fornax-studio/releases/latest': [200, { tag_name: 'v0.3.1', html_url: 'https://example.test/', assets: [asset('source.zip')] }],
    });
    const { studioRelease } = await load();
    assert.equal(await studioRelease(), undefined);
});

test('a release GitHub answers without a tag or a page offers nothing', async () => {
    for (const release of [{ html_url: 'https://example.test/' }, { tag_name: 'v0.3.1' }]) {
        github({
            'repos/liberation-data/fornax-studio': PUBLIC,
            'repos/liberation-data/fornax-studio/releases/latest': [200, { ...release, assets: [asset('fornax-studio.rpm')] }],
        });
        const { studioRelease } = await load();
        assert.equal(await studioRelease(), undefined);
    }
});

test('a release with only a Linux installer offers no Mac download', async () => {
    github({
        'repos/liberation-data/fornax-studio': PUBLIC,
        'repos/liberation-data/fornax-studio/releases/latest': [200, { tag_name: '0.3.1', html_url: 'https://example.test/', assets: [asset('fornax-studio.rpm')] }],
    });
    const { studioRelease } = await load();
    const studio = await studioRelease();
    assert.equal(studio?.version, '0.3.1');
    assert.deepEqual(studio?.mac, { appleSilicon: undefined });
    assert.deepEqual(studio?.linux.map((d) => d.label), ['Fedora / RHEL']);
});

test('with GitHub out of reach the page gets no release, no Studio link and no star counts', async () => {
    globalThis.fetch = (async () => {
        throw new TypeError('fetch failed');
    }) as typeof fetch;
    const { repoStars, studioRelease, studioRepoPublic } = await load();
    assert.equal(await studioRepoPublic(), false);
    assert.equal(await studioRelease(), undefined);
    assert.equal(await repoStars('liberation-data/fornax'), undefined);
});

test('star counts are what GitHub says, and none when it answers with an error', async () => {
    github({ 'repos/liberation-data/fornax': PUBLIC });
    const { repoStars } = await load();
    assert.equal(await repoStars('liberation-data/fornax'), 42);
    assert.equal(await repoStars('liberation-data/missing'), undefined);
});

test('GITHUB_TOKEN, where it is set, goes with every request', async () => {
    process.env.GITHUB_TOKEN = 'secret';
    const asked = github({ 'repos/liberation-data/fornax-studio': [200, { private: true }] });
    const { studioRepoPublic } = await load();
    await studioRepoPublic();
    assert.equal(asked[0].headers.Authorization, 'Bearer secret');
});
