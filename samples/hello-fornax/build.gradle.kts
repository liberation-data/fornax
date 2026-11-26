plugins { application }

repositories {
    // -PfornaxLocal=true: try ~/.m2 first, for a Fornax from ./gradlew publishToMavenLocal
    if (providers.gradleProperty("fornaxLocal").getOrElse("false").toBoolean()) mavenLocal()
    mavenCentral()
}

java { toolchain { languageVersion = JavaLanguageVersion.of(25) } }

// gradle.properties sets fornaxVersion; override it with -PfornaxVersion=...
val fornaxVersion: String by project
val fornaxNatives = when {
    !System.getProperty("os.name").startsWith("Mac") -> "linux-x86_64-vulkan"
    System.getProperty("os.arch") == "aarch64" -> "macos-aarch64-metal"
    else -> "macos-x86_64-metal"
}

dependencies {
    implementation("org.fornax4j:fornax-llama:$fornaxVersion")
    runtimeOnly("org.fornax4j:fornax-natives-$fornaxNatives:$fornaxVersion")
}

application {
    mainClass = "hello.HelloFornax"
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}
