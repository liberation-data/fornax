---
title: The fornax command
description: Fetch models with fornax pull, see what is on disk, chat with a language model and try an image or song model from a terminal, and how fornax is installed and launched.
---

`fornax` is Fornax's command-line tool. It fetches models into the Hugging Face cache and tries them without any code:

```sh
fornax pull qwen3-4b                                 # an alias from Fornax's catalogue: every file the model needs
fornax chat qwen3-4b "Write a haiku about a fornax"
```

## Installing it

- **With Fornax Studio** (coming in the next Fornax Studio release): choose **Install fornax command** from Studio's menu.
  Studio writes a `fornax` launcher into a folder on your `PATH` (`~/.local/bin` on Linux; `/usr/local/bin`, or
  `~/.local/bin`, on macOS). That `fornax` runs on Studio's own Java and engines, so it needs nothing else installed.
- **From source**, for a machine without a desktop, such as a server. It needs JDK 25 to build and Java 22 or newer to
  run (`JAVA_HOME`, or `java` on `PATH`):

  ```sh
  git clone https://github.com/liberation-data/fornax.git && cd fornax
  native/build.sh vulkan               # the engines, or skip this and add -Pfornax.cli.nativesVersion=0.2.0 below
  ./gradlew :fornax-cli:installDist      # -> fornax-cli/build/install/fornax-cli/bin/fornax and lib/
  export PATH="$PWD/fornax-cli/build/install/fornax-cli/bin:$PATH"
  ```

  The distribution's `lib/` holds the Fornax jars and the natives jar for each platform built in `build/native/`.
  `-Pfornax.cli.nativesVersion=0.2.0` takes this platform's natives jar from Maven Central instead. Copy the
  `fornax-cli` folder anywhere; `bin/fornax` can also be reached through a symlink.

`fornax info` shows which Java, engines and cache it uses.

## Fetching models

```sh
fornax pull flux2-klein                                   # three files, from three repos
fornax pull unsloth/Qwen3-4B-Instruct-2507-GGUF           # lists the repo's GGUF models and asks which
fornax pull unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M    # a quant by its tag
fornax pull Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf README.md
fornax pull some/repo '*.json' --revision v1.0            # globs match paths in the repo
fornax pull some/repo@v1.0:Q8_0                           # a revision in the reference itself
```

`org/name[@REV][:TAG]` is the same reference `new LlamaModel("org/name:TAG")` and `FornaxHub.fetch` take, so a model
your code names can be fetched ahead of time with `fornax pull` and the same string. One difference: with no `:TAG`,
`fornax pull` lists the repo's GGUF models to choose from, where the code takes the repo's only one.

An alias is a model set from [the catalogue](/docs/guides/models/#the-catalogue); `fornax` with no arguments lists
them. `fornax pull` downloads with [`FornaxHub`](/docs/guides/models/): into the Hugging Face cache, in the layout
`hf download` uses, so `fornax`, `hf`, Fornax Studio and your own app share one copy of each file. Files already in the
cache, or in a [model folder](/docs/guides/models/#model-folders), are used as they are. An interrupted download
resumes; each file is checked against the Hub's hash. The progress line shows each file's bytes, the download rate
and the time left.

A gated repo (`ltx-2.5-distilled`, for one) needs its terms accepted on its huggingface.co page and a token in
`HF_TOKEN` (or from `hf auth login`). Without them, the pull stops with a message that names the repo and what to do.
With `HF_HUB_OFFLINE=1`, `fornax pull` touches no network: an alias stops at the first file not already on disk, and
a repo (`org/name`) stops at once, as listing its files needs the Hub.

```sh
fornax list            # each alias: downloaded, partly downloaded or not; sizes; other GGUF models in the cache
fornax rm flux2-klein  # delete a model's files from the cache, keeping any another downloaded model uses
```

`hf download` works as well: `fornax` sees what it downloads, and the other way round.

## Trying models

```sh
fornax chat qwen3-0.6b "What is 2+2?"     # one reply, streamed
fornax chat qwen3-0.6b                    # interactive: /reset clears the history, /exit quits
echo "Summarise this" | fornax chat qwen3-4b
fornax test qwen3-4b                      # load time, time to first token, tokens a second
fornax test flux2-klein                   # one 256x256 image in 4 steps
fornax test ace-step --seconds 5          # a short song
```

`<model>` is an alias, `org/name:TAG`, `org/name:file.gguf`, `org/name` (when the cache has one GGUF model from it)
or a path to a `.gguf` file.

A reasoning model, such as Qwen3, thinks aloud in `<think>…</think>` before it answers. `fornax chat` hides that, and
shows only the answer; `--think` shows the reasoning too. Some models (Qwen3 Thinking, DeepSeek-R1 distills) are
trained with the `<think>` in the prompt and only close the block. `fornax chat` holds the first couple of hundred
characters of a reply until it can tell, or longer while the reply opens the way reasoning does ("Okay, the user…").
If the reasoning runs past that, you see it, but the `</think>` is still hidden. Once a reply opens inside, an
interactive chat takes the model's later replies as starting inside too, until `/reset`. Other options: `--system`, `--max-tokens`,
`--temperature`, `--seed`, `--ctx` and `--gpu-layers` (`fornax chat --help`).

Set `FORNAX_VERBOSE=1` to see the engines' own logs.

## The launcher

`bin/fornax` (and `bin/fornax.cmd` on Windows) runs `org.liberationdata.fornax.cli.Main` from three inputs. Each comes from
an environment variable, else from a value written into the launcher, else from a default:

| Input | Environment variable | Placeholder | Default |
|---|---|---|---|
| the `java` to run, Java 22 or newer | `FORNAX_JAVA` | `@FORNAX_JAVA@` | `$JAVA_HOME/bin/java`, else `java` on `PATH` |
| the folder of jars: `fornax-cli`, the `fornax-*` modules it uses and, for the engines, a `fornax-natives-<os>-<arch>-<backend>` jar | `FORNAX_CLI_LIB` | `@FORNAX_CLI_LIB@` | `../lib` next to the launcher, after following symlinks |
| a folder of native engine libraries, loaded before any natives jar | `FORNAX_LIBRARY_PATH` | `@FORNAX_LIBRARY_PATH@` | none |

`FORNAX_OPTS` and `JAVA_OPTS` add JVM options; the launcher always passes `--enable-native-access=ALL-UNNAMED`.

An installer such as Fornax Studio renders the launcher by replacing each placeholder with an absolute path (or
leaving it, for the default) and writes the result, executable, into a folder on `PATH`. Studio's `fornax` points at
Studio's bundled Java, its jars and its engines, and uses the same Hugging Face cache as everything else.
