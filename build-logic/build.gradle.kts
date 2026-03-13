plugins {
    `kotlin-dsl`
}

dependencies {
    implementation("org.jetbrains.kotlin:kotlin-gradle-plugin:${libs.versions.kotlin.get()}")
    implementation("com.android.tools.build:gradle:${libs.versions.agp.get()}")
    implementation("com.vanniktech:gradle-maven-publish-plugin:${libs.versions.vanniktechMavenPublish.get()}")
}
