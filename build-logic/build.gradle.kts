plugins {
    `kotlin-dsl`
}

dependencies {
    implementation(libs.kotlin.gradle)
    implementation(libs.android.kotlin)
    implementation(libs.vanniktech.maven.publish)
    implementation(libs.dokka)
    implementation(libs.binary.compatibility.validator)
}
