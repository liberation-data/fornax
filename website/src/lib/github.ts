/**
 * Fornax Studio's latest release on GitHub, fetched when the site is built, as drivine.org does for Drivine Studio: a
 * visitor's browser never calls GitHub, and the scheduled build keeps it fresh. What can't be fetched is left out, and
 * the page says the downloads are on their way instead.
 */
export const STUDIO = { name: 'Fornax Studio', repo: 'liberation-data/fornax-studio' } as const;
export const studioUrl = `https://github.com/${STUDIO.repo}`;

/** A GET of GitHub's API, with GITHUB_TOKEN if it's set, given up after 10 s. */
const github = (path: string): Promise<Response> => {
    const headers: Record<string, string> = { Accept: 'application/vnd.github+json' };
    if (process.env.GITHUB_TOKEN) {
        headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
    }
    return fetch(`https://api.github.com/${path}`, { headers, signal: AbortSignal.timeout(10_000) });
};

const githubJson = async (path: string): Promise<any> => {
    const response = await github(path);
    if (!response.ok) {
        throw new Error(`GitHub answered ${response.status} for ${path}`);
    }
    return response.json();
};

/** One downloadable build of Fornax Studio: a short label for its button, where GitHub serves the file, and its size. */
export interface Download {
    label: string;
    url: string;
    size: string;
}

/** Fornax Studio's latest release: its version, its release page, and the per-platform installers attached to it. */
export interface StudioRelease {
    version: string;
    url: string;
    mac: { appleSilicon?: Download; intel?: Download };
    linux: Download[];
}

interface Asset {
    name: string;
    size: number;
    browser_download_url: string;
}

/** A GitHub asset's size the way a download link shows it: whole megabytes. */
const megabytes = (bytes: number): string => `${Math.round(bytes / 1_000_000)} MB`;

async function fetchStudioRelease(): Promise<StudioRelease | undefined> {
    // A private repository's assets are no download for a visitor, even if this build's token can see them.
    if (!(await studioRepoPublic())) {
        return undefined;
    }
    // `releases/latest` is GitHub's newest published, non-prerelease, non-draft release, so a draft or a pre-release
    // build is never offered here.
    const release = await githubJson(`repos/${STUDIO.repo}/releases/latest`);
    const version = typeof release?.tag_name === 'string' ? release.tag_name.replace(/^v/, '') : undefined;
    if (!version || typeof release.html_url !== 'string') {
        return undefined;
    }
    const assets: Asset[] = Array.isArray(release.assets) ? release.assets : [];
    const download = (match: (name: string) => boolean, label: string): Download | undefined => {
        const asset = assets.find((a) => match(a.name));
        return asset ? { label, url: asset.browser_download_url, size: megabytes(asset.size) } : undefined;
    };
    // fornax-studio-mac-arm64.dmg, fornax-studio-mac-x64.dmg: package.json's mac.artifactName.
    const mac = {
        appleSilicon: download((name) => name.endsWith('.dmg') && name.includes('arm64'), 'Apple Silicon'),
        intel: download((name) => name.endsWith('.dmg') && name.includes('x64'), 'Intel'),
    };
    // The two native installers, labelled by the distros that use them. A software-centre can't open a local package on
    // current Fedora/GNOME, so the page's install hint gives the terminal command for each.
    const linux = [
        download((name) => name.endsWith('.deb'), 'Debian / Ubuntu'),
        download((name) => name.endsWith('.rpm'), 'Fedora / RHEL'),
    ].filter((d): d is Download => d !== undefined);
    // A release with no installer for any platform is no download.
    if (!mac.appleSilicon && !mac.intel && linux.length === 0) {
        return undefined;
    }
    return { version, url: release.html_url, mac, linux };
}

let studio: Promise<StudioRelease | undefined> | undefined;

/**
 * Fornax Studio's latest release and the installers it attached, fetched once per build. If the fetch fails (the
 * repository is still private, or GitHub can't be reached), or the release has no installers, there is none, and the
 * page says the downloads are coming.
 */
export function studioRelease(): Promise<StudioRelease | undefined> {
    studio ??= fetchStudioRelease().catch(() => undefined);
    return studio;
}

let open: Promise<boolean> | undefined;

/** Whether anyone can see Fornax Studio's repository, so the page links to it only once it is public. Asked once per build. */
export function studioRepoPublic(): Promise<boolean> {
    open ??= github(`repos/${STUDIO.repo}`)
        .then(async (response) => response.ok && (await response.json()).private === false)
        .catch(() => false);
    return open;
}

/** A public repo's star count for the nav's GitHub menu, fetched at build time; undefined if GitHub can't say. */
export async function repoStars(repo: string): Promise<number | undefined> {
    try {
        const it = await githubJson(`repos/${repo}`);
        return typeof it?.stargazers_count === 'number' ? it.stargazers_count : undefined;
    } catch {
        return undefined;
    }
}
