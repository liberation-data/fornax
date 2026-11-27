plugins { application }

repositories {
    // -PfornaxLocal=true: try ~/.m2 first, for a Fornax from ./gradlew publishToMavenLocal
    if (providers.gradleProperty("fornaxLocal").getOrElse("false").toBoolean()) mavenLocal()
    mavenCentral()
}

java { toolchain { languageVersion = JavaLanguageVersion.of(25) } }

// gradle.properties sets fornaxVersion; override it with -PfornaxVersion=...
val fornaxVersion: String by project
val fornaxNatives =
    if (System.getProperty("os.name").startsWith("Mac")) "macos-aarch64-metal" else "linux-x86_64-vulkan"

dependencies {
    // Every Fornax module at one version: the lines below take theirs from here
    implementation(platform("org.fornax4j:fornax-bom:$fornaxVersion"))
    implementation("org.fornax4j:fornax-llama")
    runtimeOnly("org.fornax4j:fornax-natives-$fornaxNatives")
}

application {
    mainClass = "hello.HelloFornax"
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}
