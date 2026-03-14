package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.distributions.BetaDistribution
import org.oremif.kstats.distributions.BinomialDistribution

/**
 * Performs an exact binomial test for whether the proportion of successes equals [probability].
 *
 * The null hypothesis is that the true probability of success in each trial equals [probability].
 * Unlike asymptotic tests, this test computes exact p-values using the binomial distribution,
 * making it appropriate for any sample size. The confidence interval is a Clopper-Pearson
 * (exact) interval based on the Beta distribution.
 *
 * ### Example:
 * ```kotlin
 * val result = binomialTest(successes = 7, trials = 10, probability = 0.5)
 * result.statistic          // 0.7 (observed proportion)
 * result.pValue             // two-sided exact p-value
 * result.confidenceInterval // 95% Clopper-Pearson CI for the proportion
 * result.isSignificant()    // true if p < 0.05
 * ```
 *
 * @param successes the number of observed successes. Must be in `[0, trials]`.
 * @param trials the total number of trials. Must be non-negative.
 * @param probability the hypothesized probability of success per trial. Must be in `[0, 1]`.
 *   Defaults to `0.5`.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED].
 *   [Alternative.LESS] tests if the true proportion is less than [probability],
 *   [Alternative.GREATER] tests if it is greater.
 * @param confidenceLevel the confidence level for the Clopper-Pearson confidence interval.
 *   Must be in `(0, 1)`. Defaults to `0.95` (95%).
 * @return a [TestResult] containing the observed proportion as the statistic, the exact p-value,
 *   a Clopper-Pearson confidence interval, and additional info with "successes", "trials",
 *   and "hypothesizedProbability".
 */
public fun binomialTest(
    successes: Int,
    trials: Int,
    probability: Double = 0.5,
    alternative: Alternative = Alternative.TWO_SIDED,
    confidenceLevel: Double = 0.95
): TestResult {
    if (trials < 0) throw InvalidParameterException("trials must be non-negative, got $trials")
    if (successes < 0 || successes > trials) throw InvalidParameterException(
        "successes must be in [0, trials], got successes=$successes, trials=$trials"
    )
    if (probability !in 0.0..1.0) throw InvalidParameterException(
        "probability must be in [0, 1], got $probability"
    )
    if (confidenceLevel <= 0.0 || confidenceLevel >= 1.0) throw InvalidParameterException(
        "confidenceLevel must be in (0, 1), got $confidenceLevel"
    )

    val k = successes
    val n = trials
    val p = probability

    val statistic = if (n == 0) Double.NaN else k.toDouble() / n

    if (n == 0) {
        return TestResult(
            testName = "Binomial Test",
            statistic = statistic,
            pValue = 1.0,
            alternative = alternative,
            confidenceInterval = Pair(0.0, 1.0),
            additionalInfo = mapOf(
                "successes" to k.toDouble(),
                "trials" to n.toDouble(),
                "hypothesizedProbability" to p
            )
        )
    }

    val dist = BinomialDistribution(n, p)

    val pValue = when (alternative) {
        Alternative.LESS -> dist.cdf(k)
        Alternative.GREATER -> dist.sf(k - 1)
        Alternative.TWO_SIDED -> {
            val pObserved = dist.pmf(k)
            var sum = 0.0
            for (j in 0..n) {
                val pj = dist.pmf(j)
                if (pj <= pObserved * (1.0 + 1e-7)) {
                    sum += pj
                }
            }
            sum
        }
    }

    // Clopper-Pearson exact confidence interval
    val alpha = 1.0 - confidenceLevel
    val lower = if (k == 0) 0.0 else BetaDistribution(k.toDouble(), (n - k + 1).toDouble()).quantile(alpha / 2.0)
    val upper = if (k == n) 1.0 else BetaDistribution((k + 1).toDouble(), (n - k).toDouble()).quantile(1.0 - alpha / 2.0)

    return TestResult(
        testName = "Binomial Test",
        statistic = statistic,
        pValue = pValue.coerceIn(0.0, 1.0),
        alternative = alternative,
        confidenceInterval = Pair(lower, upper),
        additionalInfo = mapOf(
            "successes" to k.toDouble(),
            "trials" to n.toDouble(),
            "hypothesizedProbability" to p
        )
    )
}
