# Module kstats-correlation
<!---IMPORT org.oremif.kstats.correlation.samples.DokkaSamples-->

Correlation coefficients, covariance/correlation matrices, and simple linear regression.

- Pearson, Spearman, Kendall (O(n log n)), point-biserial, and partial correlation — each
  returns a `CorrelationResult` with coefficient, p-value, and sample size.
- `covarianceMatrix()` and `correlationMatrix()` for multivariate analysis.
- `simpleLinearRegression()` fits y = intercept + slope * x and returns R², standard errors,
  residuals, and a `predict()` function.

## Getting started

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

## Correlation functions

- `pearsonCorrelation(x, y)` — Pearson product-moment correlation
- `spearmanCorrelation(x, y)` — Spearman rank correlation
- `kendallTau(x, y)` — Kendall rank correlation (O(n log n) merge-sort algorithm)
- `pointBiserialCorrelation(x, y)` — point-biserial correlation (overloads for `DoubleArray`,
  `BooleanArray`, `IntArray`)
- `partialCorrelation(target, control)` — partial correlation controlling for a third variable

## Matrices

- `correlationMatrix(vararg variables)` — Pearson correlation matrix
- `covarianceMatrix(vararg variables, kind)` — covariance matrix (sample or population)
- `covariance(x, y, kind)` — pairwise covariance

## Regression

- `simpleLinearRegression(x, y)` — returns `SimpleLinearRegressionResult` with `intercept`,
  `slope`, `rSquared`, `residuals`, `standardErrors`, `residualStandardError`, and `predict(x)`.
