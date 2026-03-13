pluginManagement {
    includeBuild("build-logic")
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "kstats"
include(":kstats-bom")
include(":kstats-core")
include(":kstats-distributions")
include(":kstats-hypothesis")
include(":kstats-correlation")
include(":kstats-sampling")
include(":benchmark")
