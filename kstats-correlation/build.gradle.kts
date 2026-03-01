plugins {
    id("kstats.kmp-library")
}

kotlin {
    sourceSets {
        commonMain.dependencies {
            api(project(":kstats-distributions"))
            implementation(project(":kstats-sampling"))
        }
        commonTest.dependencies {
            implementation(project(":kstats-hypothesis"))
        }
    }
}
