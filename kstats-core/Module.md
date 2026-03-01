# Module kstats-core
<!---IMPORT org.oremif.kstats.samples.DokkaSamples-->

Descriptive statistics, special math functions, and shared foundations for every other kstats module.

- Extension functions on `DoubleArray` and `Iterable<Double>` for mean, median, variance, standard
  deviation, skewness, kurtosis, quantiles, and frequency tables.
- `describe()` produces a full `DescriptiveStatistics` summary in a single call.
- `OnlineStatistics` accumulates data in a single streaming pass (Welford's algorithm with
  Terriberry extension for skewness and kurtosis).
- Special functions: gamma, beta, erf/erfInv, digamma, trigamma, combinatorics.
- Typed exception hierarchy (`InsufficientDataException`, `InvalidParameterException`,
  `ConvergenceException`, `DegenerateDataException`) and reusable validation helpers.

## Getting started

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

## Descriptive statistics

Central tendency, dispersion, and shape — all available as extension functions on `DoubleArray`,
`Iterable<Double>`, and `Sequence<Double>`.

**Central tendency:** `mean()`, `geometricMean()`, `harmonicMean()`, `median()`, `mode()`,
`trimmedMean(proportion)`, `weightedMean(weights)`.

**Dispersion:** `variance(kind)`, `standardDeviation(kind)`, `range()`, `interquartileRange()`,
`meanAbsoluteDeviation()`, `medianAbsoluteDeviation()`, `standardError()`,
`coefficientOfVariation(kind)`, `semiVariance(direction, kind)`.

**Shape:** `skewness(kind)`, `kurtosis(kind, excess)`, `centralMoment(order)`, `kStatistic(order)`.

**Quantiles:** `percentile(p)`, `quantile(p)`, `quartiles()`.

**Summary:** `describe()` returns a `DescriptiveStatistics` snapshot with all of the above.

Use `PopulationKind.SAMPLE` (default) or `PopulationKind.POPULATION` to switch between
sample and population formulas.

## Special functions

Pure-Kotlin implementations of mathematical special functions used internally by distributions
and hypothesis tests, and available for direct use:

- `lnGamma(x)`, `gamma(x)` — log-gamma and gamma function
- `lnBeta(a, b)`, `beta(a, b)` — log-beta and beta function
- `regularizedBeta(x, a, b)` — regularized incomplete beta
- `regularizedGammaP(a, x)`, `regularizedGammaQ(a, x)` — regularized incomplete gamma
- `erf(x)`, `erfc(x)`, `erfInv(x)`, `erfcInv(x)` — error functions and inverses
- `digamma(x)`, `trigamma(x)` — polygamma functions
- `lnFactorial(n)`, `lnCombination(n, k)`, `lnPermutation(n, k)` — combinatorics
- `generalizedHarmonic(n, s)` — generalized harmonic numbers
