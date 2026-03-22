# Module kstats-hypothesis

Statistical hypothesis tests returning a unified `TestResult` with statistic, p-value,
degrees of freedom, confidence interval, and `isSignificant(alpha)` helper.

## Getting started

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

## Available tests

**Parametric:**

- `tTest(sample, mu, alternative, alpha)` — one-sample t-test
- `tTest(sample1, sample2, alternative, alpha)` — two-sample t-test
- `pairedTTest(sample1, sample2, alternative, alpha)` — paired t-test
- `oneWayAnova(vararg groups)` — one-way ANOVA (returns `AnovaResult`)

**Non-parametric:**

- `mannWhitneyUTest(x, y, alternative)` — Mann-Whitney U test
- `wilcoxonSignedRankTest(sample, mu, alternative)` — Wilcoxon signed-rank test
- `friedmanTest(vararg groups)` — Friedman test
- `kolmogorovSmirnovTest(sample, cdf)` — one-sample Kolmogorov-Smirnov
- `kolmogorovSmirnovTest(sample1, sample2)` — two-sample Kolmogorov-Smirnov

**Normality:**

- `shapiroWilkTest(sample)` — Shapiro-Wilk test
- `andersonDarlingTest(sample)` — Anderson-Darling test
- `dagostinoPearsonTest(sample)` — D'Agostino-Pearson test
- `jarqueBeraTest(sample)` — Jarque-Bera test

**Categorical:**

- `chiSquaredTest(observed, expected)` — goodness-of-fit chi-squared
- `chiSquaredIndependenceTest(contingencyTable)` — chi-squared independence test
- `gTest(observed, expected)` — G-test (goodness-of-fit)
- `gIndependenceTest(contingencyTable)` — G-test (independence)
- `fisherExactTest(a, b, c, d)` — Fisher exact test (2x2)
- `binomialTest(successes, trials, p, alternative)` — binomial test

**Homogeneity of variance:**

- `leveneTest(groups)` — Levene test
- `bartlettTest(vararg groups)` — Bartlett test
- `flignerKilleenTest(vararg groups)` — Fligner-Killeen test

## Result types

- `TestResult` — statistic, pValue, degreesOfFreedom, alternative, confidenceInterval,
  additionalInfo, `isSignificant(alpha)`.
- `AnovaResult` — F-statistic, p-value, between/within sum of squares and degrees of freedom.
- `Alternative` — enum: `TWO_SIDED`, `LESS`, `GREATER`.
