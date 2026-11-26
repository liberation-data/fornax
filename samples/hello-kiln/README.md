# hello-kiln

The smallest Kiln app: it streams a chat reply from Qwen3 4B Instruct, running on your GPU inside the JVM. It
gets Kiln from Maven Central (`org.kiln4j:kiln-llama` plus the natives jar for your OS and arch), so you can copy
this folder out of the repository and build it anywhere.

## Prerequisites

- JDK 25.
- A GPU driver: Vulkan on Linux x86-64 (glibc 2.39 or newer), or macOS 13.3 or newer (Metal).
- The model, fetched with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

  ```sh
  hf download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf
  ```

## Run

```sh
./gradlew run                                  # or: ./gradlew run --args='"Name a country song about a robot"'
mvn -q compile exec:java                       # or: mvn -q compile exec:java -Dexec.args='"Name a country song about a robot"'
```

The first run downloads the natives jar (about 100 MB on Linux) and extracts the engine to Kiln's cache.

## Building against another Kiln

The version is `kilnVersion` in `gradle.properties` and `kiln.version` in `pom.xml`. To try a Kiln you built
yourself, run `./gradlew publishToMavenLocal` in the Kiln repository, then:

```sh
./gradlew run -PkilnLocal=true -PkilnVersion=0.1.0-SNAPSHOT  # -PkilnLocal adds ~/.m2 to the repositories
mvn -q compile exec:java -Dkiln.version=0.1.0-SNAPSHOT       # Maven always looks in ~/.m2 first
```
