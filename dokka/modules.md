# Module Kstats
<!---IMPORT org.oremif.kstats.samples.DokkaSamples-->
<!---IMPORT org.oremif.kstats.distributions.samples.DokkaSamples-->
<!---IMPORT org.oremif.kstats.hypothesis.samples.DokkaSamples-->
<!---IMPORT org.oremif.kstats.correlation.samples.DokkaSamples-->
<!---IMPORT org.oremif.kstats.sampling.samples.DokkaSamples-->

Zero-dependency Kotlin Multiplatform statistics library. All math is implemented from scratch
in pure Kotlin common code — no platform-specific dependencies, no JVM-only numerics.
Targets JVM, Android, iOS, macOS, watchOS, tvOS, Linux, Windows, JS, and Wasm.

Use the BOM for version alignment, then depend on the modules you need:

```kotlin
dependencies {
    // Version alignment
    implementation(platform("org.oremif:kstats-bom:<version>"))

    // Pick the modules you need
    implementation("org.oremif:kstats-core")
    implementation("org.oremif:kstats-distributions")
    implementation("org.oremif:kstats-hypothesis")
    implementation("org.oremif:kstats-correlation")
    implementation("org.oremif:kstats-sampling")
}
```

---

## kstats-core

Descriptive statistics, special math functions, and shared foundations for every other module.

- Extension functions on `DoubleArray` and `Iterable<Double>` for mean, median, variance, standard
  deviation, skewness, kurtosis, quantiles, and frequency tables.
- `describe()` produces a full `DescriptiveStatistics` summary in a single call.
- `OnlineStatistics` accumulates data in a single streaming pass (Welford's algorithm with
  Terriberry extension for skewness and kurtosis).
- Special functions: gamma, beta, erf/erfInv, digamma, combinatorics.
- Typed exception hierarchy (`InsufficientDataException`, `InvalidParameterException`,
  `ConvergenceException`, `DegenerateDataException`) and reusable validation helpers.

<!---FUN dokkaCoreDescriptive-->

```kotlin
val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)

val stats = data.describe()
stats.mean              // 5.0
stats.standardDeviation // 2.0
stats.median            // 4.5
stats.skewness          // 0.656...

// Streaming computation — no need to hold all data in memory
val online = OnlineStatistics()
online.addAll(data)
online.mean       // 5.0
online.variance() // 4.571...
```

<!---END-->

---

## kstats-distributions

28 probability distributions (18 continuous + 10 discrete) with a shared interface for density,
cumulative probability, quantiles, and random sampling.

**Continuous:** Normal, Student's t, Chi-Squared, F, Exponential, Gamma, Beta, Pareto, Weibull,
Laplace, LogNormal, Logistic, Cauchy, Gumbel, Levy, Nakagami, Uniform, Triangular.

**Discrete:** Binomial, BinomialBeta, Bernoulli, Poisson, Geometric, Negative Binomial,
Uniform Discrete, Hypergeometric, Logarithmic, Zipf.

Every distribution exposes `mean`, `variance`, `standardDeviation`, `skewness`, `kurtosis`,
and `entropy` as properties.

<!---FUN dokkaDistributions-->

```kotlin
val normal = NormalDistribution(mu = 0.0, sigma = 1.0)
normal.pdf(0.0)           // 0.3989...
normal.cdf(1.96)          // 0.975...
normal.quantile(0.975)    // 1.96
normal.sample(Random(42)) // a single random draw

val poisson = PoissonDistribution(rate = 4.0)
poisson.pmf(3)            // P(X = 3)
poisson.cdf(5)            // P(X <= 5)
poisson.mean              // 4.0
```

<!---END-->

---

## kstats-hypothesis

Statistical hypothesis tests returning a unified `TestResult` with statistic, p-value,
degrees of freedom, confidence interval, and `isSignificant(alpha)` helper.

- **Parametric:** one-sample / two-sample / paired t-test, one-way ANOVA.
- **Non-parametric:** Mann-Whitney U, Wilcoxon signed-rank, Friedman, Kolmogorov-Smirnov.
- **Normality:** Shapiro-Wilk, Anderson-Darling, D'Agostino-Pearson, Jarque-Bera.
- **Categorical:** chi-squared (goodness-of-fit & independence), G-test, Fisher exact, binomial test.
- **Homogeneity of variance:** Levene, Bartlett, Fligner-Killeen.

<!---FUN dokkaHypothesis-->

```kotlin
val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
val result = tTest(sample, mu = 5.0)
result.statistic          // t-statistic
result.pValue             // p-value
result.confidenceInterval // 95% CI for the mean
result.isSignificant()    // true if p < 0.05

// Normality check before choosing a test
val sw = shapiroWilkTest(sample)
if (!sw.isSignificant()) { /* data is consistent with normality */ }
```

<!---END-->

---

## kstats-correlation

Correlation coefficients, covariance/correlation matrices, and simple linear regression.

- Pearson, Spearman, Kendall (O(n log n)), point-biserial, and partial correlation — each
  returns a `CorrelationResult` with coefficient, p-value, and sample size.
- `covarianceMatrix()` and `correlationMatrix()` for multivariate analysis.
- `simpleLinearRegression()` fits y = intercept + slope · x and returns R², standard errors,
  residuals, and a `predict()` function.

<!---FUN dokkaCorrelation-->

```kotlin
val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1)

val r = pearsonCorrelation(x, y)
r.coefficient // 0.999...
r.pValue      // < 0.001

val reg = simpleLinearRegression(x, y)
reg.slope        // ~2.0
reg.intercept    // ~0.04
reg.rSquared     // 0.999...
reg.predict(6.0) // predicted y for x = 6
```

<!---END-->

---

## kstats-sampling

Ranking, normalization, binning, and weighted random sampling utilities.

- `rank()` with configurable `TieMethod` (AVERAGE, MIN, MAX, FIRST, LAST, RANDOM).
- `zScore()` standardization and `minMaxNormalize()` scaling.
- `bin()` / `frequencyTable()` for histogram construction.
- `randomSample()` (without replacement) and `bootstrapSample()` (with replacement).
- `WeightedDice<T>` and `WeightedCoin` for categorical and Bernoulli sampling.

<!---FUN dokkaSampling-->

```kotlin
val data = doubleArrayOf(3.0, 1.0, 4.0, 1.0, 5.0)

data.rank()              // [3.0, 1.5, 4.0, 1.5, 5.0] (average ties)
data.zScore()            // standardized to mean=0, sd=1
data.minMaxNormalize()   // scaled to [0.0, 1.0]

val die = WeightedDice(mapOf("A" to 0.7, "B" to 0.2, "C" to 0.1), Random(42))
die.roll()               // "A" (most likely)
```

<!---END-->

Pick the modules that match your analysis needs, or pull in the full set through the BOM.
