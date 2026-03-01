import org.jetbrains.dokka.gradle.engine.parameters.VisibilityModifier

plugins {
    id("org.jetbrains.dokka")
}

dokka {
    dokkaSourceSets.configureEach {
        includes.from("Module.md")

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
            footerMessage = "Copyright © 2025-2026 Oremif"
        }
    }
}
