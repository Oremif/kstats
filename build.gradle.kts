
plugins {
    id("kstats.dokka")
    alias(libs.plugins.korro)
    alias(libs.plugins.binary.compatibility.validator)
}

dependencies {
    dokka(project(":kstats-core"))
    dokka(project(":kstats-distributions"))
    dokka(project(":kstats-hypothesis"))
    dokka(project(":kstats-correlation"))
    dokka(project(":kstats-sampling"))
}

val kstatsVersion: String = libs.versions.kstats.get()

allprojects {
    group = "org.oremif"
    version = kstatsVersion
}

apiValidation {
    ignoredProjects.addAll(listOf("benchmark"))

    @OptIn(kotlinx.validation.ExperimentalBCVApi::class)
    klib {
        enabled = true
    }
}

dokka {
    moduleName = "Kstats"

    dokkaPublications.html {
        includes.from("dokka/modules.md")
    }
}

korro {
    docs {
        from(fileTree(rootProject.rootDir) {
            include("README.md")
            include("kstats-*/Module.md")
            include("docs/**/*.mdx")
            include("dokka/modules.md")
        })
        baseDir.set(rootProject.layout.projectDirectory)
    }

    samples {
        from(fileTree(project.projectDir) {
            include("kstats-*/src/commonTest/kotlin/org/oremif/kstats/**/samples/*.kt")
        })
    }
}
