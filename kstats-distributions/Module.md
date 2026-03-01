# Module kstats-distributions
<!---IMPORT org.oremif.kstats.distributions.samples.DokkaSamples-->

28 probability distributions (18 continuous + 10 discrete) with a shared interface for density,
cumulative probability, quantiles, and random sampling.

Every distribution exposes `mean`, `variance`, `standardDeviation`, `skewness`, `kurtosis`,
and `entropy` as properties.

## Getting started

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

## Interfaces

- `Distribution` — sealed interface with `mean`, `variance`, `standardDeviation`, `skewness`,
  `kurtosis`, `entropy`, `cdf(x)`, `sf(x)`, `quantile(p)`.
- `ContinuousDistribution` — extends `Distribution`, adds `pdf(x)` and `sample(random)`.
- `DiscreteDistribution` — extends `Distribution`, adds `pmf(k)` and `sample(random)`.

## Continuous distributions

| Distribution | Parameters |
|---|---|
| `NormalDistribution` | mu, sigma |
| `StudentTDistribution` | df |
| `ChiSquaredDistribution` | k |
| `FDistribution` | df1, df2 |
| `ExponentialDistribution` | lambda |
| `GammaDistribution` | shape, scale |
| `BetaDistribution` | alpha, beta |
| `ParetoDistribution` | xm, alpha |
| `WeibullDistribution` | shape, scale |
| `LaplaceDistribution` | mu, b |
| `LogNormalDistribution` | mu, sigma |
| `LogisticDistribution` | mu, s |
| `CauchyDistribution` | location, scale |
| `GumbelDistribution` | location, scale |
| `LevyDistribution` | location, scale |
| `NakagamiDistribution` | m, omega |
| `UniformDistribution` | min, max |
| `TriangularDistribution` | min, mode, max |

## Discrete distributions

| Distribution | Parameters |
|---|---|
| `BinomialDistribution` | n, p |
| `BetaBinomialDistribution` | n, alpha, beta |
| `BernoulliDistribution` | p |
| `PoissonDistribution` | lambda |
| `GeometricDistribution` | p |
| `NegativeBinomialDistribution` | r, p |
| `UniformDiscreteDistribution` | min, max |
| `HypergeometricDistribution` | N, K, n |
| `LogarithmicDistribution` | p |
| `ZipfDistribution` | n, s |
