---
title: The kiln command
description: Fetch models with kiln pull, see what is on disk, chat with a language model and try an image or song model from a terminal, and how kiln is installed and launched.
---

`kiln` is Kiln's command-line tool. It fetches models into the Hugging Face cache and tries them without any code:

```sh
kiln pull qwen3-4b                                 # an alias from Kiln's catalogue: every file the model needs
kiln chat qwen3-4b "Write a haiku about a kiln"
```

## Installing it

- **With Kiln Studio** (coming in the next Kiln Studio release): choose **Install kiln command** from Studio's menu.
  Studio writes a `kiln` launcher into a folder on your `PATH` (`~/.local/bin` on Linux; `/usr/local/bin`, or
  `~/.local/bin`, on macOS). That `kiln` runs on Studio's own Java and engines, so it needs nothing else installed.
- **From source**, for a machine without a desktop, such as a server. It needs JDK 25 to build and Java 22 or newer to
  run (`JAVA_HOME`, or `java` on `PATH`):

  ```sh
  git clone https://github.com/liberation-data/kiln.git && cd kiln
  native/build.sh vulkan               # the engines, or skip this and add -Pkiln.cli.nativesVersion=0.1.0 below
  ./gradlew :kiln-cli:installDist      # -> kiln-cli/build/install/kiln-cli/bin/kiln and lib/
  export PATH="$PWD/kiln-cli/build/install/kiln-cli/bin:$PATH"
  ```

  The distribution's `lib/` holds the Kiln jars and the natives jar for each platform built in `build/native/`.
  `-Pkiln.cli.nativesVersion=0.1.0` takes this platform's natives jar from Maven Central instead. Copy the
  `kiln-cli` folder anywhere; `bin/kiln` can also be reached through a symlink.

`kiln info` shows which Java, engines and cache it uses.

## Fetching models

```sh
kiln pull flux2-klein                                   # three files, from three repos
kiln pull unsloth/Qwen3-4B-Instruct-2507-GGUF           # lists the repo's GGUF models and asks which
kiln pull unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M    # a quant by its tag
kiln pull Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf README.md
kiln pull some/repo '*.json' --revision v1.0            # globs match paths in the repo
```

An alias is a model set from [the catalogue](/docs/guides/models/#the-catalogue); `kiln` with no arguments lists
them. `kiln pull` downloads with [`KilnHub`](/docs/guides/models/): into the Hugging Face cache, in the layout
`hf download` uses, so `kiln`, `hf`, Kiln Studio and your own app share one copy of each file. Files already in the
cache, or in a [model folder](/docs/guides/models/#model-folders), are used as they are. An interrupted download
resumes; each file is checked against the Hub's hash. The progress line shows each file's bytes, the download rate
and the time left.

A gated repo (`ltx-2.5-distilled`, for one) needs its terms accepted on its huggingface.co page and a token in
`HF_TOKEN` (or from `hf auth login`). Without them, the pull stops with a message that names the repo and what to do.

```sh
kiln list            # each alias: downloaded, partly downloaded or not; sizes; other GGUF models in the cache
kiln rm flux2-klein  # delete a model's files from the cache, keeping any another downloaded model uses
```

`hf download` works as well: `kiln` sees what it downloads, and the other way round.

## Trying models

```sh
kiln chat qwen3-0.6b "What is 2+2?"     # one reply, streamed
kiln chat qwen3-0.6b                    # interactive: /reset clears the history, /exit quits
echo "Summarise this" | kiln chat qwen3-4b
kiln test qwen3-4b                      # load time, time to first token, tokens a second
kiln test flux2-klein                   # one 256x256 image in 4 steps
kiln test ace-step --seconds 5          # a short song
```

`<model>` is an alias, `org/name:TAG`, `org/name:file.gguf`, `org/name` (when the cache has one GGUF model from it)
or a path to a `.gguf` file.

A reasoning model, such as Qwen3, thinks aloud in `<think>…</think>` before it answers. `kiln chat` hides that, and
shows only the answer; `--think` shows the reasoning too. Other options: `--system`, `--max-tokens`,
`--temperature`, `--seed`, `--ctx` and `--gpu-layers` (`kiln chat --help`).

Set `KILN_VERBOSE=1` to see the engines' own logs.

## The launcher

`bin/kiln` (and `bin/kiln.cmd` on Windows) runs `org.liberationdata.kiln.cli.Main` from three inputs. Each comes from
an environment variable, else from a value written into the launcher, else from a default:

| Input | Environment variable | Placeholder | Default |
|---|---|---|---|
| the `java` to run, Java 22 or newer | `KILN_JAVA` | `@KILN_JAVA@` | `$JAVA_HOME/bin/java`, else `java` on `PATH` |
| the folder of jars: `kiln-cli`, the `kiln-*` modules it uses and, for the engines, a `kiln-natives-<os>-<arch>-<backend>` jar | `KILN_CLI_LIB` | `@KILN_CLI_LIB@` | `../lib` next to the launcher, after following symlinks |
| a folder of native engine libraries, loaded before any natives jar | `KILN_LIBRARY_PATH` | `@KILN_LIBRARY_PATH@` | none |

`KILN_OPTS` and `JAVA_OPTS` add JVM options; the launcher always passes `--enable-native-access=ALL-UNNAMED`.

An installer such as Kiln Studio renders the launcher by replacing each placeholder with an absolute path (or
leaving it, for the default) and writes the result, executable, into a folder on `PATH`. Studio's `kiln` points at
Studio's bundled Java, its jars and its engines, and uses the same Hugging Face cache as everything else.
