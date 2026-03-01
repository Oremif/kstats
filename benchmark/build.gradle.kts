plugins {
    kotlin("jvm")
    alias(libs.plugins.kotlin.allopen)
    alias(libs.plugins.kotlinx.benchmark)
}

allOpen {
    annotation("org.openjdk.jmh.annotations.State")
}

dependencies {
    implementation(project(":library"))
    implementation(libs.kotlinx.benchmark.runtime)
    implementation(libs.commons.math3)
}

benchmark {
    targets { register("main") }
    configurations {
        named("main") {
            warmups = 5
            iterations = 5
            iterationTime = 3
            iterationTimeUnit = "s"
            mode = "avgt"
            outputTimeUnit = "us"
        }
        register("smoke") {
            warmups = 2
            iterations = 3
            iterationTime = 1
            iterationTimeUnit = "s"
            mode = "avgt"
            outputTimeUnit = "us"
        }
    }
}
