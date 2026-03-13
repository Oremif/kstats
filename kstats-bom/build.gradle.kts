plugins {
    `java-platform`
    alias(libs.plugins.vanniktech.mavenPublish)
}

group = "org.oremif"
version = libs.versions.kstats.get()

dependencies {
    constraints {
        api(project(":kstats-core"))
        api(project(":kstats-distributions"))
        api(project(":kstats-hypothesis"))
        api(project(":kstats-correlation"))
        api(project(":kstats-sampling"))
    }
}

mavenPublishing {
    publishToMavenCentral(automaticRelease = true)

    signAllPublications()

    coordinates(group.toString(), "kstats-bom", version.toString())

    pom {
        name = "kstats-bom"
        description = "Kotlin Multiplatform statistics library (BOM)"
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
