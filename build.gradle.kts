plugins {
    id("kstats.dokka")
}

dokka {
    moduleName = "kstats"
}

dependencies {
    dokka(project(":kstats-core"))
    dokka(project(":kstats-distributions"))
    dokka(project(":kstats-hypothesis"))
    dokka(project(":kstats-correlation"))
    dokka(project(":kstats-sampling"))
}
