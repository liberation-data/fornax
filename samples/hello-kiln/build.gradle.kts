plugins { application }

repositories {
    // -PkilnLocal=true: try ~/.m2 first, for a Kiln from ./gradlew publishToMavenLocal
    if (providers.gradleProperty("kilnLocal").getOrElse("false").toBoolean()) mavenLocal()
    mavenCentral()
}

java { toolchain { languageVersion = JavaLanguageVersion.of(25) } }

// gradle.properties sets kilnVersion; override it with -PkilnVersion=...
val kilnVersion: String by project
val kilnNatives = when {
    !System.getProperty("os.name").startsWith("Mac") -> "linux-x86_64-vulkan"
    System.getProperty("os.arch") == "aarch64" -> "macos-aarch64-metal"
    else -> "macos-x86_64-metal"
}

dependencies {
    implementation("org.kiln4j:kiln-llama:$kilnVersion")
    runtimeOnly("org.kiln4j:kiln-natives-$kilnNatives:$kilnVersion")
}

application {
    mainClass = "hello.HelloKiln"
    applicationDefaultJvmArgs = listOf("--enable-native-access=ALL-UNNAMED")
}
