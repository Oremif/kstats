plugins {
    `java-platform`
    id("kstats.maven-publish")
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
