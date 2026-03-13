[![Maven Central](https://img.shields.io/maven-central/v/org.oremif/kstats-core)](https://central.sonatype.com/artifact/org.oremif/kstats-core)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

# kstats

Kotlin Multiplatform statistics library.

## Features

| Module                  | Description                                     | Highlights                                                                        |
|-------------------------|-------------------------------------------------|-----------------------------------------------------------------------------------|
| **kstats-core**         | Math foundations, descriptive statistics         | `mean()`, `standardDeviation()`, `describe()`, `skewness()`, `kurtosis()`         |
| **kstats-distributions**| Continuous & discrete probability distributions | `NormalDistribution`, `StudentTDistribution`, `BinomialDistribution`, and 14 more |
| **kstats-hypothesis**   | Parametric & non-parametric tests               | `tTest()`, `oneWayAnova()`, `mannWhitneyUTest()`, `shapiroWilkTest()`             |
| **kstats-correlation**  | Correlation coefficients & regression           | `pearsonCorrelation()`, `spearmanCorrelation()`, `simpleLinearRegression()`       |
| **kstats-sampling**     | Ranking, normalization, weighted sampling       | `zScore()`, `rank()`, `bootstrapSample()`, `WeightedDice`                         |

## Installation

### Using BOM (recommended)

```kotlin
dependencies {
    implementation(platform("org.oremif:kstats-bom:0.2.0"))

    implementation("org.oremif:kstats-core")
    implementation("org.oremif:kstats-distributions")
    implementation("org.oremif:kstats-hypothesis")
    implementation("org.oremif:kstats-correlation")
    implementation("org.oremif:kstats-sampling")
}
```

### Individual modules

```kotlin
dependencies {
    implementation("org.oremif:kstats-core:0.2.0")
    implementation("org.oremif:kstats-distributions:0.2.0")
    // add other modules as needed
}
```

### Kotlin Multiplatform

```kotlin
kotlin {
    sourceSets {
        commonMain.dependencies {
            implementation(project.dependencies.platform("org.oremif:kstats-bom:0.2.0"))

            implementation("org.oremif:kstats-core")
            implementation("org.oremif:kstats-distributions")
            // add other modules as needed
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
./gradlew jvmTest

# Run tests for a specific module
./gradlew :kstats-core:jvmTest

# Run all platform tests
./gradlew allTests

# Full build
./gradlew build
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
