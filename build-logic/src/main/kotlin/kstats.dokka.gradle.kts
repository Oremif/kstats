import org.jetbrains.dokka.gradle.engine.parameters.VisibilityModifier

plugins {
    id("org.jetbrains.dokka")
}

dokka {
    dokkaSourceSets.configureEach {
        sourceLink {
            localDirectory.set(projectDir.resolve("src"))
            remoteUrl("https://github.com/oremif/kstats/tree/master/${project.name}/src")
            remoteLineSuffix.set("#L")
        }

        documentedVisibilities(VisibilityModifier.Public)
    }

    dokkaPublications.configureEach {
        suppressInheritedMembers.set(true)
    }

    pluginsConfiguration {
        html {
            footerMessage.set("kstats — Kotlin Multiplatform Statistics Library")
        }
    }
}
