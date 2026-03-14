plugins {
    alias(libs.plugins.android.kotlin.multiplatform.library) apply false
    alias(libs.plugins.kotlinMultiplatform) apply  false
    alias(libs.plugins.vanniktech.mavenPublish) apply false
    alias(libs.plugins.kotlinx.benchmark) apply false
    alias(libs.plugins.kotlin.allopen) apply false
    alias(libs.plugins.dokka)
}

dependencies {
    dokka(project(":kstats-core"))
    dokka(project(":kstats-distributions"))
    dokka(project(":kstats-hypothesis"))
    dokka(project(":kstats-correlation"))
    dokka(project(":kstats-sampling"))
}
