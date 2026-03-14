import org.gradle.api.artifacts.VersionCatalogsExtension

plugins {
    id("com.vanniktech.maven.publish")
}

val libs = the<VersionCatalogsExtension>().named("libs")

group = "org.oremif"
version = libs.findVersion("kstats").get().toString()

mavenPublishing {
    publishToMavenCentral(automaticRelease = true)

    if (project.providers.gradleProperty("signing.keyId").isPresent ||
        project.providers.environmentVariable("ORG_GRADLE_PROJECT_signingInMemoryKey").isPresent
    ) {
        signAllPublications()
    }

    coordinates(group.toString(), project.name, version.toString())

    pom {
        name = project.name
        description = "Kotlin Multiplatform statistics library"
        inceptionYear = "2025"
        url = "https://github.com/oremif/kstats"
        licenses {
            license {
                name = "The Apache License, Version 2.0"
                url = "https://www.apache.org/licenses/LICENSE-2.0.txt"
                distribution = "repo"
            }
        }
        developers {
            developer {
                id = "devcrocod"
                name = "Pavel Gorgulov"
                url = "https://github.com/devcrocod"
            }
        }
        scm {
            url = "https://github.com/oremif/kstats"
            connection = "scm:git:git://github.com/oremif/kstats.git"
            developerConnection = "scm:git:ssh://github.com/oremif/kstats.git"
        }
    }
}
