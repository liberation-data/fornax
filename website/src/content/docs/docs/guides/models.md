---
title: Downloading models
description: Pull a model's files into the Hugging Face cache, find weights you already have in a ComfyUI folder, see what is on disk, and evict what you no longer need.
---

A Kiln model is often several files. FLUX.2 [klein] is a diffusion model, a VAE and a text encoder, from three
Hugging Face repos; LTX-2.5 is five files. `KilnHub`, in `kiln-core`, handles a model's files as one **model set**:
it downloads what is missing, finds what is already on the machine, reports what is there, and removes it again.

Files go into the Hugging Face cache, in the layout `huggingface_hub` uses. `hf`, transformers and diffusers see what
Kiln downloads, and Kiln sees what they download. Nothing is stored twice.

## From the command line

The quickest way to fetch a model is the [`kiln` command](/docs/guides/cli/), which pulls through `KilnHub`:

```sh
kiln pull flux2-klein     # an alias from the catalogue: all three files
kiln list                 # what of each model is on disk
kiln rm flux2-klein       # evict it again
```

To get `kiln`, choose **Install kiln command** from Kiln Studio's menu (coming in the next Kiln Studio release), or
build it from source with `./gradlew :kiln-cli:installDist`. The Hugging Face CLI is the alternative:
`hf download black-forest-labs/FLUX.2-klein-base-4b-fp8 flux-2-klein-base-4b-fp8.safetensors` and so on, one command
per file, into the same cache. The rest of this page is the Java API.

## Pulling a model

```java
import org.liberationdata.kiln.hub.KilnHub;
import org.liberationdata.kiln.hub.ModelCatalog;
import org.liberationdata.kiln.hub.ModelSet;

import java.nio.file.Path;
import java.util.Map;

KilnHub hub = KilnHub.standard();
ModelSet flux = ModelCatalog.require("flux2-klein");

Map<String, Path> files = hub.pull(flux, p ->
        System.out.printf("%s %s %.0f%%%n", p.part().role(), p.phase(), p.fraction() * 100));

Path diffusionModel = files.get("diffusion_model");
Path vae = files.get("vae");
Path textEncoder = files.get("llm");
```

`pull` returns each part's file, by role. It downloads only the parts that are not already on disk; if every part is
there, it makes no network requests at all.

A pull:

- **resumes.** A download in progress is kept as `blobs/<hash>.incomplete`. If the connection drops, the pull
  retries with an HTTP `Range` request from the last byte received, for as long as each try adds bytes; it gives up
  after five tries in a row that add none. If the pull is cancelled or the process exits, the
  next pull continues from the same place. `hf download` uses the same file name, so either tool can finish a download
  the other started.
- **verifies.** Each file is checked against the hash the Hub gives for it: the SHA-256 of an LFS file (the
  `X-Linked-Etag` header), or the git blob hash of a small file. If the hash does not match, the file is deleted and the
  pull fails with `HubException` (reason `CHECKSUM_MISMATCH`).
- **checks free space first.** Before it downloads anything, the pull adds up what it still needs to fetch. If that
  would leave less than 1 GB free on the cache's disk, it fails with reason `DISK_SPACE`. Change the margin with
  `KilnHub.builder().reserveBytes(...)`.
- **waits for another pull of the same file.** Each file is downloaded under a lock file,
  `.locks/models--org--name/<hash>.lock` in the cache. A second pull of the same file, from another thread or another
  process, waits until the first finishes and then uses its file. It does not download the file a second time.
  `hf download` uses the same lock file but a different kind of lock, so the lock does not hold between Kiln and `hf`:
  don't download the same file with both at once.
- **sends a token for gated repos.** See [Gated repos](#gated-repos).

## Progress and cancelling

The listener receives a `PullProgress` when a part changes phase, and about ten times a second while bytes arrive.
It has what a progress bar needs:

| Field | |
|---|---|
| `phase()` | `CHECKING` (asking the Hub for the file's size and hash), `PRESENT` (already on disk), `WAITING` (another pull is downloading this file), `DOWNLOADING`, `DONE` |
| `part()`, `index()`, `count()` | which part, and how many there are |
| `bytesDone()`, `bytesTotal()` | this part's bytes on disk, and its size |
| `setBytesDone()`, `setBytesTotal()` | the same for all the parts this pull downloads. Parts that were already on disk are not counted |
| `fraction()` | `setBytesDone / setBytesTotal`, from 0 to 1; `-1` while the sizes are not known yet (the `CHECKING` events) |
| `bytesPerSecond()` | the download rate so far. Bytes resumed from an earlier pull are not counted |
| `eta()` | the time left at that rate |

The listener runs on the pulling thread, so keep it short. To run a pull in the background and cancel it, use
`pullAsync`:

```java
PullTask task = hub.pullAsync(flux, progress -> ui.update(progress));
// later, if the user clicks Cancel:
task.cancel();
// or wait for it:
Map<String, Path> files = task.result().join();
```

Cancelling closes the connection immediately. The partial file stays on disk, so pulling again resumes it.
`task.result()` then completes with a `CancellationException`. A blocking `pull` is cancelled by interrupting its
thread.

To show the download size before the user starts a pull, call `plan`. It makes one HEAD request for each missing part:

```java
PullPlan plan = hub.plan(flux);
long bytes = plan.bytesToDownload();   // what is left to fetch, minus what an earlier pull left
```

## Status

`status` reads only the disk, so it is cheap enough to call for every model each time a list is drawn:

```java
ModelStatus s = hub.status(flux);
s.state();         // PRESENT, PARTIAL or MISSING
for (var part : s.parts())
    System.out.println(part.part().role() + " " + part.state() + " " + part.bytesOnDisk() + " / " + part.size());
```

A part is `PRESENT` when its file is in the Hugging Face cache or a model folder. It is `PARTIAL` when a pull started
it and did not finish; its `size()` is then the full size, recorded when the pull started. A `MISSING` part's size is
`-1`, because only the Hub knows it; use `plan` to find out. The set's own state is `PRESENT` when every part is,
`MISSING` when no part has any bytes on disk, and `PARTIAL` otherwise.

## Finding files

`find(part)` looks for a part's file:

1. In the Hugging Face cache: in the snapshot that the source's branch (usually `main`) points to, then in any other
   snapshot. An older snapshot's file still loads, and when you are offline it is the only one there is.
2. In the configured model folders, in order.

`require(set)` returns every part's file, by role, or throws an exception that names the missing parts.

A part can list more than one source. Qwen-Image's text encoder is published in both the Qwen-Image and the
HunyuanVideo repos, and a machine that has either copy does not need to download it again. `pull` downloads the first
source when none of them is on disk.

### Model folders

A machine that has run ComfyUI already has many of these weights in `ComfyUI/models/`. List such folders in the system
property `kiln.modelDirs` or the environment variable `KILN_MODEL_DIRS`, separated by `:` (`;` on Windows). Kiln
Studio's earlier setting, `KILN_STUDIO_MODEL_DIRS`, is still read.

```sh
export KILN_MODEL_DIRS=~/ComfyUI/models:/mnt/models
```

For a part whose file is `split_files/vae/flux2-vae.safetensors`, Kiln looks in each folder for:

1. `split_files/vae/flux2-vae.safetensors`, the file's path in the repo;
2. `vae/flux2-vae.safetensors`, the same path without the `split_files/` prefix that Comfy-Org's repos use;
3. `flux2-vae.safetensors` in the ComfyUI folders for the part's role, then in the others: `diffusion_models/`, `unet/`
   and `checkpoints/` for a diffusion model; `text_encoders/` and `clip/` for a text encoder; `vae/` for a VAE;
   `latent_upscale_models/` for an upscaler; `loras/` for a LoRA.

A folder that contains a `models/` directory (the ComfyUI folder itself) is searched there too. Kiln only reads model
folders. It never writes to them or deletes anything from them.

## Gated repos

Some repos, such as `Lightricks/LTX-2.5`, ask you to accept their terms before you can download from them. Accept the
terms on the repo's page while signed in to Hugging Face, then create a token (Settings, Access Tokens; read access is
enough). Give the token to Kiln in one of these ways:

- set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`, its older name, which Kiln reads after it, as `hf` does);
- run `hf auth login`, which saves it where Kiln looks for it (`HF_TOKEN_PATH`, else `$HF_HOME/token`, else
  `$XDG_CACHE_HOME/huggingface/token`, else `~/.cache/huggingface/token`);
- pass it to the builder: `KilnHub.builder().token("hf_...").build()`.

Without a token, or with a token that has not accepted the terms, the pull fails with `HubException` (reason
`ACCESS_DENIED`). The message names the repo and the page where you accept its terms. `ModelSet.gated()` tells an app
in advance which models need a token.

The token is sent to the Hub's own host only. It is not sent to the CDN that the Hub redirects downloads to.

## Evicting a model

```java
Eviction e = hub.evict(flux);
e.bytesFreed();
e.kept();   // parts left in place because another model on disk uses the same file
```

`evict` removes the set's files from the Hugging Face cache: the file's entry in every snapshot of its repo, then the
blob that entry pointed to, and any partial download of it. A part that more than one repo publishes is removed only
from the repo it was found in (the one `find` and `status` report): evicting `qwen-image` removes its text encoder
from the Qwen-Image repo, and leaves the copy in the HunyuanVideo repo, which other tools may use. It keeps:

- a blob that another snapshot entry still points to;
- a file that another model set on disk also uses. Evicting ACE-Step turbo keeps the VAE and text encoder that ACE-Step
  XL turbo also needs, if XL is downloaded. "Other model sets" means the catalogue's, or the ones you pass to
  `evict(set, others)`;
- a file that another pull is downloading right now.

When a repo has no snapshots or blobs left, its folder is removed. Files in model folders are never deleted.

## The catalogue

`ModelCatalog` lists the models Kiln's examples, the CLI and Kiln Studio use, so that all of them offer the same
models with the same files:

| Kind | Sets |
|---|---|
| `CHAT` | `qwen3-0.6b`, `qwen3-4b` |
| `VISION` | `qwen3-vl-4b`, `paddleocr-vl`, `medgemma-4b` |
| `SPEECH` | `kokoro` |
| `TRANSCRIPTION` | `whisper-large-v3-turbo`, `whisper-base.en` |
| `IMAGE` | `flux2-klein`, `qwen-image`, `qwen-image-edit` |
| `SONG` | `ace-step-turbo`, `ace-step` (turbo with the 4B lyric LM), `ace-step-xl-turbo`, `ace-step-xl`, `yue2` |
| `VIDEO` | `ltx-2.5-distilled`, `ltx-2.5-dev` (gated) |

`ModelCatalog.byName` also accepts Kiln Studio's ids, such as `image:flux2-klein`. A set's `license()` is filled in
where the weights' license limits what you may do with them or with what they make: YuE2 is CC BY-NC 4.0, and MedGemma
comes with Google's Health AI Developer Foundations terms.

Your own models are model sets too:

```java
ModelSet mine = ModelSet.of("my-model", ModelSet.Kind.IMAGE, "My model", "stable-diffusion.cpp",
        ModelSet.Part.of("diffusion_model", "me/my-model", "model.safetensors"),
        ModelSet.Part.of("vae", "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"));
hub.pull(mine, PullListener.NONE);
```

A `Source` can pin a revision: `new ModelSet.Source(repo, file, "<40-character commit>")`. Without one, it uses `main`.

## Settings

| Setting | |
|---|---|
| `HF_HUB_CACHE`, `HF_HOME` | where the cache is: `HF_HUB_CACHE`, else `$HF_HOME/hub`, else `$XDG_CACHE_HOME/huggingface/hub`, else `~/.cache/huggingface/hub`, as `hf` has it. `KilnHub.builder().cacheRoot(path)` overrides them |
| `-Dkiln.modelDirs`, `KILN_MODEL_DIRS` | model folders searched after the cache |
| `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN_PATH` | the access token for gated repos |
| `HF_ENDPOINT` | the Hub to download from (default `https://huggingface.co`), for a mirror |
| `HF_HUB_OFFLINE` | when set to `1`, a pull that needs the network fails at once with reason `OFFLINE` |
