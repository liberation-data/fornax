# Contributing to Fornax

Thanks for helping. These are the rules a change is held to, written down before the first outside contribution
arrives so that nobody finds out about them in review.

## Before you start

- **Open an issue first for anything bigger than a bug fix.** A new engine, model family, public API or option is a
  design conversation before it is code. A bug fix with a test can go straight to a pull request.
- Look at the [open issues](https://github.com/liberation-data/fornax/issues) and the [roadmap](README.md#roadmap)
  to see whether someone is already on it.

## Building and testing

See [Building](README.md#building) for the toolchain and `native/build.sh`. Then:

```sh
./gradlew build
```

The integration tests (`*IT`) skip themselves when their model isn't in the Hugging Face cache, so on a machine with
no models downloaded `./gradlew build` needs only the JDK. Those model-free tests must pass on every change. If you
do have models downloaded, build the natives first (`native/build.sh`): a test whose model is there loads it. When you touch an engine, also run its integration tests (`*IT`) with the model downloaded, and say in
the pull request which ones you ran and on what hardware.

## What every change needs

- **Code that matches the code around it.** There's no formatter to run; follow the file's style (4-space indent,
  lines up to about 120 characters, Javadoc that says why rather than what). The build compiles with `-Xlint:all`,
  so keep it free of new warnings.
- **Tests.** A new branch or failure path needs a test that drives it. A bug fix needs a test that failed before it.
- **Docs in the same pull request.** If behaviour a user can see changes, update the Javadoc, the README and the
  guide under `website/src/content/docs/` that describes it.
- **Numbers for performance claims.** Say what got faster, by how much, on what hardware, and give the command that
  produced the numbers (for example `./gradlew bench --args='out/bench acestep'` from `fornax-bench`, or the
  integration test's own timing output), so a reviewer can rerun it.
- **Proof for new engines and model families.** Integration tests that load the real model and check its output
  against a reference: the engine's own CLI on the same input, or a known-good fixture.

## What we decline, however good the code

- **Runtime dependencies in the library modules.** `fornax-core` and the engine modules (`fornax-llama`, `fornax-sd`,
  and the rest) depend only on the JDK and the native engines. Tooling that isn't published, like `fornax-bench`, can
  use libraries.
- **Options nobody asked for.** Every option is a promise to keep it working. Add one when a user needs it, with that
  need in the issue.

## AI-assisted contributions

Welcome. Say in the pull request which tool you used. You must have read every line you submit and be able to
explain and defend it in review; "the tool wrote it" isn't an answer to a review question.

## Commits and pull requests

- **One change per pull request.** A refactor that a feature needs can come with it; unrelated cleanups go in their
  own.
- **Commit subjects** are a short sentence saying what the change does, without a type prefix
  (`Embeddings and reranking in an isolated worker`, not `feat: add isolated embed`). The body says why, and anything
  a reviewer would otherwise have to work out.
- **Pull request descriptions** say what changed, why, how you tested it, and what's left. Link the issue.
- Rebase or merge `main` as you like; don't force-push `main`.
- By contributing you agree your work is licensed under the project's [Apache License 2.0](LICENSE).

## Security

Don't report vulnerabilities in public issues. See [SECURITY.md](SECURITY.md).
