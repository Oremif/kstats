[![Maven Central](https://img.shields.io/maven-central/v/org.oremif/kstats-core)](https://central.sonatype.com/artifact/org.oremif/kstats-core)
[![Build](https://github.com/Oremif/kstats/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/Oremif/kstats/actions/workflows/build.yml)
[![API Reference](https://img.shields.io/badge/API-Dokka-0f766e)](https://oremif.github.io/kstats)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/images/logo-light.svg">
  <img alt="kstats logo" src="docs/images/logo-light.svg" width="88">
</picture>

# kstats

A Kotlin Multiplatform statistics toolkit for descriptive analysis, probability distributions, hypothesis testing, correlation, regression, and sampling.

Built for shared `commonMain` code and published as focused modules to Maven Central.

kstats gives Kotlin projects a focused statistics layer without pulling in a larger data-science stack. It covers the common workflow from describing a sample to fitting distributions, running hypothesis tests, measuring relationships, and resampling data.

## Why kstats?

- Kotlin Multiplatform-first APIs that work naturally in `commonMain`
- Small modules with a BOM, so you can add only what your project needs
- Consistent result types such as `DescriptiveStatistics`, `TestResult`, and `CorrelationResult`
- Broad target coverage across JVM, native targets, JS, and Wasm

## Modules

| Module | What it covers | Typical API |
| --- | --- | --- |
| `kstats-core` | Descriptive statistics, moments, quantiles, summary objects, streaming stats | `mean()`, `median()`, `standardDeviation()`, `describe()`, `OnlineStatistics` |
| `kstats-distributions` | 28 probability distributions with a shared API | `NormalDistribution`, `PoissonDistribution`, `cdf()`, `quantile()`, `sample(..., Random)` |
| `kstats-hypothesis` | Parametric, non-parametric, normality, and categorical tests | `tTest()`, `oneWayAnova()`, `shapiroWilkTest()`, `chiSquaredIndependenceTest()` |
| `kstats-correlation` | Correlation, covariance, matrices, and simple linear regression | `pearsonCorrelation()`, `spearmanCorrelation()`, `partialCorrelation()`, `simpleLinearRegression()` |
| `kstats-sampling` | Ranking, normalization, binning, bootstrap, and weighted sampling | `rank()`, `zScore()`, `bin()`, `bootstrapSample()`, `WeightedDice` |

<details>
<summary><strong>Included models and test families</strong></summary>

- Distributions: 18 continuous and 10 discrete models, including Normal, Student's t, Gamma, Weibull, Beta, Poisson, Binomial, Negative Binomial, Hypergeometric, and Zipf.
- Hypothesis tests: one-sample, two-sample, and paired t-tests, ANOVA, Mann-Whitney, Wilcoxon signed-rank, Kolmogorov-Smirnov, Shapiro-Wilk, Anderson-Darling, D'Agostino-Pearson, Jarque-Bera, chi-squared tests, Fisher exact, G-test, Levene, Bartlett, Fligner-Killeen, Friedman, and binomial testing.
- Correlation and regression: Pearson, Spearman, Kendall tau, point-biserial, partial correlation, covariance, correlation matrices, covariance matrices, and simple linear regression.
- Sampling and transforms: ranking with tie strategies, z-score and min-max normalization, binning, random sampling, bootstrap sampling, `WeightedCoin`, and `WeightedDice`.

</details>

## Installation

Use the BOM if you depend on more than one kstats module.

```kotlin
dependencies {
    implementation(platform("org.oremif:kstats-bom:0.3.0"))

    implementation("org.oremif:kstats-core")
    implementation("org.oremif:kstats-distributions")
    implementation("org.oremif:kstats-hypothesis")
    implementation("org.oremif:kstats-correlation")
    implementation("org.oremif:kstats-sampling")
}
```

For a Kotlin Multiplatform project, add dependencies in `commonMain`:

```kotlin
kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform("org.oremif:kstats-bom:0.3.0"))

            implementation("org.oremif:kstats-core")
            implementation("org.oremif:kstats-hypothesis")
        }
    }
}
```

If you only need one module, you can depend on it directly, for example `implementation("org.oremif:kstats-core:0.3.0")`.

## A Quick Feel for the API

```kotlin
import kotlin.random.Random
import org.oremif.kstats.correlation.pearsonCorrelation
import org.oremif.kstats.correlation.simpleLinearRegression
import org.oremif.kstats.descriptive.describe
import org.oremif.kstats.distributions.NormalDistribution
import org.oremif.kstats.hypothesis.tTest
import org.oremif.kstats.sampling.zScore

val sample = doubleArrayOf(2.0, 4.0, 4.0, 5.0, 7.0, 9.0)
val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1)

val summary = sample.describe() // descriptive snapshot with quartiles, spread, and shape
val standardized = sample.zScore() // z-score normalization
val normal = NormalDistribution(mu = summary.mean, sigma = summary.standardDeviation)
val test = tTest(sample, mu = 5.0) // one-sample t-test against mu = 5.0
val correlation = pearsonCorrelation(x, y) // correlation coefficient and p-value
val regression = simpleLinearRegression(x, y) // fitted line with prediction support
val draws = normal.sample(3, Random.Default) // three random draws
```

## Platform Support

kstats is built around `commonMain` and the multiplatform modules are configured for:

- JVM and Android
- Android Native
- iOS, macOS, watchOS, and tvOS
- Linux and Windows
- JS for browser and Node.js
- Wasm JS and Wasm WASI

<details>
<summary><strong>Exact Kotlin targets</strong></summary>

`jvm`, `android`, `androidNativeArm32`, `androidNativeArm64`, `androidNativeX86`, `androidNativeX64`, `iosX64`, `iosArm64`, `iosSimulatorArm64`, `macosArm64`, `macosX64`, `linuxArm64`, `linuxX64`, `mingwX64`, `watchosArm32`, `watchosArm64`, `watchosX64`, `watchosDeviceArm64`, `watchosSimulatorArm64`, `tvosArm64`, `tvosSimulatorArm64`, `js`, `wasmJs`, `wasmWasi`

</details>

## Documentation

- API reference: [Dokka site](https://oremif.github.io/kstats)
- Guides and docs source: [`docs/`](docs)

## Development

```bash
./gradlew jvmTest
./gradlew allTests
./gradlew build
```

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contribution workflow, issue guidelines, and project conventions.

## License

kstats is licensed under the [Apache License 2.0](LICENSE).
