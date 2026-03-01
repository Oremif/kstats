[![Maven Central](https://img.shields.io/maven-central/v/org.oremif/kstats)](https://central.sonatype.com/artifact/org.oremif/kstats)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# kstats

Kotlin Multiplatform statistics library.

## Features

| Module            | Description                                     | Highlights                                                                        |
|-------------------|-------------------------------------------------|-----------------------------------------------------------------------------------|
| **Descriptive**   | Central tendency, dispersion, shape, quantiles  | `mean()`, `standardDeviation()`, `describe()`, `skewness()`, `kurtosis()`         |
| **Distributions** | Continuous & discrete probability distributions | `NormalDistribution`, `StudentTDistribution`, `BinomialDistribution`, and 14 more |
| **Hypothesis**    | Parametric & non-parametric tests               | `tTest()`, `oneWayAnova()`, `mannWhitneyUTest()`, `shapiroWilkTest()`             |
| **Correlation**   | Correlation coefficients & regression           | `pearsonCorrelation()`, `spearmanCorrelation()`, `simpleLinearRegression()`       |
| **Sampling**      | Ranking, normalization, weighted sampling       | `zScore()`, `rank()`, `bootstrapSample()`, `WeightedDice`                         |

## Installation

Add the dependency to your `build.gradle.kts`:

```kotlin
dependencies {
    implementation("org.oremif:kstats:0.1.0")
}
```

For Kotlin Multiplatform projects:

```kotlin
kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation("org.oremif:kstats:0.1.0")
        }
    }
}
```

## Supported platforms

| Platform | Target                                    |
|----------|-------------------------------------------|
| JVM      | `jvm`                                     |
| Android  | `android`                                 |
| iOS      | `iosX64`, `iosArm64`, `iosSimulatorArm64` |
| Linux    | `linuxX64`                                |

## Quickstart

### Descriptive statistics

```kotlin
val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)

data.mean()                // 5.0
data.median()              // 4.5
data.standardDeviation()   // 2.138...
data.skewness()            // 0.656...

val summary = data.describe()  // count, mean, sd, min, Q1, median, Q3, max, ...
```

### Probability distributions

```kotlin
val normal = NormalDistribution(mu = 100.0, sigma = 15.0)
normal.cdf(115.0)       // ≈ 0.8413
normal.quantile(0.95)   // ≈ 124.67
normal.sample(1000)     // draw 1000 random values

val poisson = PoissonDistribution(lambda = 3.0)
poisson.pmf(5)           // P(X = 5)
```

### Hypothesis testing

```kotlin
val sample = doubleArrayOf(5.1, 4.9, 5.3, 5.0, 4.8)
val result = tTest(sample, mu = 5.0)
result.pValue            // p-value
result.isSignificant()   // true if p < 0.05
```

### Correlation & regression

```kotlin
val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1)

val r = pearsonCorrelation(x, y)
r.coefficient   // ≈ 0.999
r.pValue        // significance

val reg = simpleLinearRegression(x, y)
reg.slope       // ≈ 2.0
reg.rSquared    // ≈ 0.998
reg.predict(6.0)
```

## Building & testing

```bash
# Run all JVM tests
./gradlew :library:jvmTest

# Run all tests on the current platform
./gradlew :library:allTests

# Full build
./gradlew :library:build
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
