plugins {
    `java-platform`
    id("kstats.maven-publish")
    id("kstats.ktfmt")
}

dependencies {
    constraints {
        api(project(":kstats-core"))
        api(project(":kstats-distributions"))
        api(project(":kstats-hypothesis"))
        api(project(":kstats-correlation"))
        api(project(":kstats-sampling"))
    }
}
