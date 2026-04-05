package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.ConfidenceInterval
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.distributions.BetaDistribution
import org.oremif.kstats.distributions.BinomialDistribution
import org.oremif.kstats.distributions.NormalDistribution
import kotlin.math.sqrt

private val standardNormal = NormalDistribution(0.0, 1.0)

/**
 * Performs an exact binomial test for whether the proportion of successes equals [probability].
 *
 * The null hypothesis is that the true probability of success in each trial equals [probability].
 * Unlike asymptotic tests, this test computes exact p-values using the binomial distribution,
 * making it appropriate for any sample size. The confidence interval method can be selected
 * via [ciMethod]: Clopper-Pearson (exact, conservative), Wilson (score-based, narrower),
 * or Agresti-Coull (adjusted Wald, simple approximation).
 *
 * ### Example:
 * ```kotlin
 * val result = binomialTest(successes = 7, trials = 10, probability = 0.5)
 * result.statistic          // 0.7 (observed proportion)
 * result.pValue             // two-sided exact p-value
 * result.confidenceInterval // 95% Clopper-Pearson CI for the proportion
 * result.isSignificant()    // true if p < 0.05
 *
 * // Use Wilson score interval for narrower CI
 * val wilson = binomialTest(successes = 45, trials = 100, ciMethod = CIMethod.WILSON)
 * wilson.confidenceInterval // Wilson score interval
 * ```
 *
 * @param successes the number of observed successes. Must be in `[0, trials]`.
 * @param trials the total number of trials. Must be non-negative.
 * @param probability the hypothesized probability of success per trial. Must be in `[0, 1]`.
 *   Defaults to `0.5`.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED].
 *   [Alternative.LESS] tests if the true proportion is less than [probability],
 *   [Alternative.GREATER] tests if it is greater.
 * @param confidenceLevel the confidence level for the confidence interval.
 *   Must be in `(0, 1)`. Defaults to `0.95` (95%).
 * @param ciMethod the method used to compute the confidence interval for the proportion.
 *   Defaults to [CIMethod.CLOPPER_PEARSON] (exact interval). Use [CIMethod.WILSON] for
 *   narrower intervals recommended in A/B testing, or [CIMethod.AGRESTI_COULL] for a
 *   simple approximation.
 * @return a [TestResult] containing the observed proportion as the statistic, the exact p-value,
 *   a confidence interval computed using [ciMethod], and additional info with "successes",
 *   "trials", and "hypothesizedProbability".
 */
public fun binomialTest(
    successes: Int,
    trials: Int,
    probability: Double = 0.5,
    alternative: Alternative = Alternative.TWO_SIDED,
    confidenceLevel: Double = 0.95,
    ciMethod: CIMethod = CIMethod.CLOPPER_PEARSON
): TestResult {
    if (trials < 0) throw InvalidParameterException("trials must be non-negative, got $trials")
    if (successes !in 0..trials) throw InvalidParameterException(
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
            confidenceInterval = ConfidenceInterval(0.0, 1.0),
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

    val alpha = 1.0 - confidenceLevel
    val ci = when (ciMethod) {
        CIMethod.CLOPPER_PEARSON -> {
            // Clopper-Pearson exact confidence interval
            val cpLower = if (k == 0) 0.0
            else BetaDistribution(k.toDouble(), (n - k + 1).toDouble()).quantile(alpha / 2.0)
            val cpUpper = if (k == n) 1.0
            else BetaDistribution((k + 1).toDouble(), (n - k).toDouble()).quantile(1.0 - alpha / 2.0)
            ConfidenceInterval(cpLower, cpUpper)
        }

        CIMethod.WILSON -> {
            // Wilson score interval — algebraically guaranteed to stay in [0, 1]
            val z = standardNormal.quantile(1.0 - alpha / 2.0)
            val z2 = z * z
            val denom = 1.0 + z2 / n
            val center = (statistic + z2 / (2.0 * n)) / denom
            val halfWidth = (z / denom) * sqrt(statistic * (1.0 - statistic) / n + z2 / (4.0 * n * n))
            ConfidenceInterval(center - halfWidth, center + halfWidth)
        }

        CIMethod.AGRESTI_COULL -> {
            // Agresti-Coull adjusted Wald interval
            val z = standardNormal.quantile(1.0 - alpha / 2.0)
            val z2 = z * z
            val nTilde = n + z2
            val pTilde = (k + z2 / 2.0) / nTilde
            val se = sqrt(pTilde * (1.0 - pTilde) / nTilde)
            ConfidenceInterval(
                (pTilde - z * se).coerceIn(0.0, 1.0),
                (pTilde + z * se).coerceIn(0.0, 1.0)
            )
        }
    }

    return TestResult(
        testName = "Binomial Test",
        statistic = statistic,
        pValue = pValue.coerceIn(0.0, 1.0),
        alternative = alternative,
        confidenceInterval = ci,
        additionalInfo = mapOf(
            "successes" to k.toDouble(),
            "trials" to n.toDouble(),
            "hypothesizedProbability" to p
        )
    )
}
