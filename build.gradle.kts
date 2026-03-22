plugins {
    id("kstats.dokka")
}

dependencies {
    dokka(project(":kstats-core"))
    dokka(project(":kstats-distributions"))
    dokka(project(":kstats-hypothesis"))
    dokka(project(":kstats-correlation"))
    dokka(project(":kstats-sampling"))
}

dokka {
    moduleName = "Kstats"

    dokkaPublications.html {
        includes.from("docs/moduledoc.md")
    }
}
