# Security policy

## Reporting a vulnerability

Report it privately through GitHub:
[**Report a vulnerability**](https://github.com/liberation-data/fornax/security/advisories/new) (the Security tab, then
"Report a vulnerability"). Please don't open a public issue, discussion or pull request for it.

Include what you found, the version or commit, and how to reproduce it: a crafted file, a command, or a test. We'll
acknowledge the report within a week, keep you updated while we fix it, and credit you in the advisory unless you'd
rather we didn't.

## Supported versions

Fixes go into the latest release and `main`. Older releases don't get patches.

## Scope

Fornax loads model files and runs native code in your JVM's process, so a bug that would be a crash elsewhere can be
memory corruption here. In scope:

- **Parsing model files.** GGUF and safetensors parsing in the native engines, as Fornax
  builds and calls them (including the patches in `native/patches/`). A model file from the Hub is untrusted input: a
  crafted one that crashes, corrupts memory or runs code is a vulnerability.
- **Downloads and the model cache.** The hub client (`org.liberationdata.fornax.hub`) and `fornax pull`: path
  traversal out of the cache, a token sent to a host other than the Hub, a checksum that isn't checked, or a cache
  entry another model can overwrite.
- **The isolated worker.** The child process, its socket and its shared memory (`fornax.isolation=process`).
- **Fornax Studio's local server** (`fornax-studio-server`) and any future Fornax server: anything reachable from
  outside the machine, or from a web page in the user's browser.

Out of scope: bugs in an upstream engine that Fornax's use doesn't reach (report those upstream, though we'd like to
hear about them too), and what a model generates.
