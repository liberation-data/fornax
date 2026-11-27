# hello-fornax

The smallest Fornax app: it streams a chat reply from Qwen3 4B Instruct, running on your GPU inside the JVM. It
gets Fornax from Maven Central (`org.fornax4j:fornax-llama` plus the natives jar for your OS and arch), so you can copy
this folder out of the repository and build it anywhere.

## Prerequisites

- Java 22 or newer.
- A GPU driver: Vulkan on Linux x86-64 (glibc 2.39 or newer), or macOS 13.3 or newer on Apple silicon (Metal).

## Run

```sh
./gradlew run                                  # or: ./gradlew run --args='"Name a country song about a robot"'
mvn -q compile exec:java                       # or: mvn -q compile exec:java -Dexec.args='"Name a country song about a robot"'
```

The first run downloads the model (Qwen3 4B, 2.5 GB) into the Hugging Face cache, shared with `hf`, Fornax Studio and
the `fornax` command, and the natives jar (about 130 MB on Linux), whose engine Fornax extracts to its cache. To
fetch the model ahead of time: `fornax pull qwen3-4b`, or `hf download unsloth/Qwen3-4B-Instruct-2507-GGUF
Qwen3-4B-Instruct-2507-Q4_K_M.gguf`.

## Building against another Fornax

The version is `fornaxVersion` in `gradle.properties` and `fornax.version` in `pom.xml`. To try a Fornax you built
yourself, run `./gradlew publishToMavenLocal` in the Fornax repository, then:

```sh
./gradlew run -PfornaxLocal=true -PfornaxVersion=0.3.0-SNAPSHOT  # -PfornaxLocal adds ~/.m2 to the repositories
mvn -q compile exec:java -Dfornax.version=0.3.0-SNAPSHOT       # Maven always looks in ~/.m2 first
```

## Did it work?

If hello-fornax ran for you, a ⭐ on [Fornax on GitHub](https://github.com/liberation-data/fornax) helps other Java
developers find it. If it didn't, [open an issue](https://github.com/liberation-data/fornax/issues/new) and tell us what broke.
