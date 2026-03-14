@file:OptIn(ExperimentalWasmDsl::class)

import org.gradle.api.artifacts.VersionCatalogsExtension
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("org.jetbrains.kotlin.multiplatform")
    id("com.android.kotlin.multiplatform.library")
    id("kstats.maven-publish")
    id("kstats.dokka")
}

val libs = the<VersionCatalogsExtension>().named("libs")

group = "org.oremif"
version = libs.findVersion("kstats").get().toString()

kotlin {
    explicitApi()

    jvm()

    android {
        namespace = "org.oremif.${project.name.replace("-", ".")}"
        compileSdk = libs.findVersion("android-compileSdk").get().toString().toInt()
        minSdk = libs.findVersion("android-minSdk").get().toString().toInt()

        withJava()
        withHostTestBuilder {}.configure {}
        withDeviceTestBuilder {
            sourceSetTreeName = "test"
        }

        compilations.configureEach {
            compileTaskProvider.configure {
                compilerOptions.jvmTarget.set(JvmTarget.JVM_11)
            }
        }
    }
    androidNativeArm32()
    androidNativeArm64()
    androidNativeX86()
    androidNativeX64()


    // iOS
    iosX64()
    iosArm64()
    iosSimulatorArm64()

    // Desktop
    macosArm64()
    macosX64()
    linuxArm64()
    linuxX64()
    mingwX64()

    // WatchOS
    watchosArm32()
    watchosArm64()
    watchosX64()
    watchosDeviceArm64()
    watchosSimulatorArm64()

    // tvOS
    tvosArm64()
    tvosSimulatorArm64()


    // JS
    js {
        browser()
        nodejs()
        binaries.library()
    }

    // Wasm
    wasmJs {
        browser()
        nodejs()
        binaries.library()
    }

    wasmWasi {
        binaries.library()
    }

    sourceSets {
        commonTest.dependencies {
            implementation(kotlin("test"))
        }
    }
}
