[![Maven Central](https://img.shields.io/maven-central/v/org.oremif/kstats-core)](https://central.sonatype.com/artifact/org.oremif/kstats-core)
[![Build](https://github.com/Oremif/kstats/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/Oremif/kstats/actions/workflows/build.yml)
[![API Reference](https://img.shields.io/badge/API-Dokka-0f766e)](https://oremif.github.io/kstats)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/logo-light.svg">
  <img alt="kstats logo" src="docs/images/logo-light.svg" width="88">
</picture>

<!---IMPORT org.oremif.kstats.samples.ReadmeSamples-->
<!---IMPORT org.oremif.kstats.distributions.samples.ReadmeSamples-->
<!---IMPORT org.oremif.kstats.hypothesis.samples.ReadmeSamples-->
<!---IMPORT org.oremif.kstats.correlation.samples.ReadmeSamples-->
<!---IMPORT org.oremif.kstats.sampling.samples.ReadmeSamples-->

# kstats

A Kotlin Multiplatform statistics toolkit covering descriptive analysis, probability distributions, hypothesis testing,
correlation, regression, and sampling. Pure Kotlin, published to Maven Central as focused modules.

## Quickstart

<!---FUN quickstart-->

```kotlin
val data = doubleArrayOf(2.0, 4.0, 4.0, 5.0, 7.0, 9.0)
val summary = data.describe()
// => DescriptiveStatistics(count=6, mean=5.17, median=4.5, standardDeviation=2.48, ...)

data.mean()               // => 5.1667
data.standardDeviation()  // => 2.4833
data.skewness()           // => 0.3942
```

<!---END-->

## Installation

### Gradle (BOM)

```kotlin
dependencies {
    implementation(platform("org.oremif:kstats-bom:0.5.0"))
    implementation("org.oremif:kstats-core")
    // add other modules as needed
}
```

### Kotlin Multiplatform

```kotlin
kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform("org.oremif:kstats-bom:0.5.0"))
            implementation("org.oremif:kstats-core")
        }
    }
}
```

If you only need one module: `implementation("org.oremif:kstats-core:0.5.0")`.

## Modules

| Module                                                                     | Description                                                             |
|----------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`kstats-core`](https://kstats.oremif.org/core/overview)                   | Descriptive statistics, moments, quantiles, streaming stats             |
| [`kstats-distributions`](https://kstats.oremif.org/distributions/overview) | 28 probability distributions (18 continuous + 10 discrete)              |
| [`kstats-hypothesis`](https://kstats.oremif.org/hypothesis/overview)       | Parametric, non-parametric, normality, and categorical tests            |
| [`kstats-correlation`](https://kstats.oremif.org/correlation/overview)     | Correlation coefficients, covariance matrices, simple linear regression |
| [`kstats-sampling`](https://kstats.oremif.org/sampling/overview)           | Ranking, normalization, binning, bootstrap, weighted sampling           |

### kstats-core

<!---FUN coreDescriptiveStats-->

```kotlin
val data = doubleArrayOf(2.0, 4.0, 4.0, 5.0, 7.0, 9.0)
val summary = data.describe()
summary.mean              // => 5.1667
summary.median            // => 4.5
summary.standardDeviation // => 2.4833

val stats = OnlineStatistics()
stats.addAll(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0))
stats.mean                // => 3.0
stats.standardDeviation() // => 1.5811
```

<!---END-->

### kstats-distributions

<!---FUN distributionsNormal-->

```kotlin
val normal = NormalDistribution(mu = 0.0, sigma = 1.0)
normal.pdf(0.0)                  // => 0.3989
normal.cdf(1.96)                 // => 0.9750
normal.quantile(0.975)           // => 1.9600
normal.sample(5, Random(42))     // => [0.11, -0.87, ...]
```

<!---END-->

### kstats-hypothesis

<!---FUN hypothesisTTest-->

```kotlin
val sample = doubleArrayOf(2.0, 4.0, 4.0, 5.0, 7.0, 9.0)
val result = tTest(sample, mu = 5.0)
result.statistic                 // => 0.1644
result.pValue                    // => 0.8759
result.isSignificant(alpha = 0.05) // => false
```

<!---END-->

### kstats-correlation

<!---FUN correlationPearsonRegression-->

```kotlin
val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1)

val r = pearsonCorrelation(x, y)
r.coefficient                    // => 0.9987
r.pValue                         // => 0.0001

val reg = simpleLinearRegression(x, y)
reg.slope                        // => 1.99
reg.rSquared                     // => 0.9973
reg.predict(6.0)                 // => 11.99
```

<!---END-->

### kstats-sampling

<!---FUN samplingRankNormalize-->

```kotlin
val data = doubleArrayOf(3.0, 1.0, 4.0, 1.0, 5.0)
data.rank()                      // => [3.0, 1.5, 4.0, 1.5, 5.0]
data.zScore()                    // => [-0.16, -1.47, 0.49, -1.47, 1.14]

listOf(1, 2, 3, 4, 5).bootstrapSample(10, Random(42))

val dice = WeightedDice(mapOf("A" to 3.0, "B" to 1.0))
dice.roll()                      // => "A" (75% probability)
```

<!---END-->

## Platform Support

kstats targets every platform Kotlin supports.

| Platform | Targets                                                                                       |
|----------|-----------------------------------------------------------------------------------------------|
| JVM      | `jvm`                                                                                         |
| Android  | `android`, `androidNativeArm32`, `androidNativeArm64`, `androidNativeX86`, `androidNativeX64` |
| iOS      | `iosArm64`, `iosX64`, `iosSimulatorArm64`                                                     |
| macOS    | `macosArm64`                                                                                  |
| watchOS  | `watchosArm32`, `watchosArm64`, `watchosDeviceArm64`, `watchosSimulatorArm64`                 |
| tvOS     | `tvosArm64`, `tvosSimulatorArm64`                                                             |
| Linux    | `linuxArm64`, `linuxX64`                                                                      |
| Windows  | `mingwX64`                                                                                    |
| JS       | `js`                                                                                          |
| Wasm     | `wasmJs`, `wasmWasi`                                                                          |

## Documentation

- **Guides and tutorials**: [kstats.oremif.org](https://kstats.oremif.org/)
- **API reference**: [Dokka site](https://oremif.github.io/kstats)
- **Interactive notebooks**: [samples/](samples/) — Kotlin Notebooks with Kandy visualizations for each guide
- **Benchmarks**: [benchmark/README.md](benchmark/README.md) — JMH results comparing kstats vs Apache Commons Math

## Development

```bash
./gradlew jvmTest        # run JVM tests
./gradlew allTests       # run all platform tests
./gradlew build          # full build
```

```bash
./gradlew :benchmark:benchmark      # JMH benchmarks (kstats vs Apache Commons Math)
./gradlew :benchmark:smokeBenchmark # quick smoke run
```

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution workflow, issue guidelines, and
project conventions. Please also read [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) before participating in issues, pull
requests, or discussions.

## License

Licensed under the [Apache License 2.0](LICENSE).
