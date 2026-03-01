# Module kstats-sampling
<!---IMPORT org.oremif.kstats.sampling.samples.DokkaSamples-->

Ranking, normalization, binning, and weighted random sampling utilities.

## Getting started

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

## Ranking

- `rank(tieMethod)` — assign ranks to values with configurable tie-breaking.
  `TieMethod`: `AVERAGE`, `MIN`, `MAX`, `DENSE`, `ORDINAL`.
- `percentileRank()` — percentile rank of each element.

## Normalization

- `zScore()` — standardize to mean = 0, standard deviation = 1.
- `minMaxNormalize()` — scale to [0, 1].
- `minMaxNormalize(newMin, newMax)` — scale to a custom range.

## Binning

- `bin(binCount)`, `bin(binSize)` — partition data into `Bin<Double>` entries.
- `frequencyTable(binCount)`, `frequencyTable(binSize)` — partition into `FrequencyBin` entries
  with counts.

## Sampling

- `randomSample(n, random)` — sample without replacement.
- `bootstrapSample(n, random)` — sample with replacement.

## Weighted sampling

- `WeightedDice<T>(weights)` — categorical sampler from a map of outcomes to probabilities.
- `WeightedCoin(probability)` — Bernoulli sampler with `roll()`.
