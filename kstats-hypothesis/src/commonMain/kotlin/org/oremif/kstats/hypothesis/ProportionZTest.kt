package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.distributions.NormalDistribution
import kotlin.math.abs
import kotlin.math.sqrt

// Standard normal distribution used for all p-value and CI computations.
private val standardNormal = NormalDistribution(0.0, 1.0)

/**
 * Performs a one-sample proportion z-test for whether the true proportion equals [p0].
 *
 * The null hypothesis is that the probability of success in each trial equals [p0]. The test
 * computes a z-statistic using the standard error under the null hypothesis and compares it
 * to the standard normal distribution. This is an asymptotic test appropriate when the sample
 * size is large enough for the normal approximation to hold (commonly n * p0 >= 5 and
 * n * (1 - p0) >= 5). For small samples, consider [binomialTest] instead.
 *
 * The confidence interval is a Wald interval based on the observed proportion.
 *
 * ### Example:
 * ```kotlin
 * val result = proportionZTest(successes = 60, trials = 100, p0 = 0.5)
 * result.statistic          // 2.0 (z-statistic)
 * result.pValue             // 0.0455 (two-sided)
 * result.confidenceInterval // Wald 95% CI for the observed proportion
 * result.isSignificant()    // true at alpha = 0.05
 * ```
 *
 * @param successes the number of observed successes. Must be in `[0, trials]`.
 * @param trials the total number of trials. Must be positive.
 * @param p0 the hypothesized proportion under the null hypothesis. Must be in `(0, 1)`.
 *   Defaults to `0.5`.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED],
 *   which tests whether the true proportion differs from [p0] in either direction.
 * @param confidenceLevel the confidence level for the Wald confidence interval.
 *   Must be in `(0, 1)`. Defaults to `0.95` (95%).
 * @return a [TestResult] containing the z-statistic, p-value, a Wald confidence interval for
 *   the observed proportion, and additional info with "observedProportion",
 *   "hypothesizedProportion", and "standardError".
 * @see binomialTest for an exact test suitable for small sample sizes.
 */
public fun proportionZTest(
    successes: Int,
    trials: Int,
    p0: Double = 0.5,
    alternative: Alternative = Alternative.TWO_SIDED,
    confidenceLevel: Double = 0.95
): TestResult {
    if (trials <= 0) throw InsufficientDataException("trials must be positive, got $trials")
    if (successes !in 0..trials) throw InvalidParameterException(
        "successes must be in [0, trials], got successes=$successes, trials=$trials"
    )
    if (p0 <= 0.0 || p0 >= 1.0) throw InvalidParameterException(
        "p0 must be in (0, 1), got $p0"
    )
    if (confidenceLevel <= 0.0 || confidenceLevel >= 1.0) throw InvalidParameterException(
        "confidenceLevel must be in (0, 1), got $confidenceLevel"
    )

    val n = trials.toDouble()
    val pHat = successes.toDouble() / n

    // z-statistic under H0: p = p0
    val se0 = sqrt(p0 * (1.0 - p0) / n)
    val z = (pHat - p0) / se0

    val pValue = when (alternative) {
        Alternative.TWO_SIDED -> 2.0 * standardNormal.sf(abs(z))
        Alternative.LESS -> standardNormal.cdf(z)
        Alternative.GREATER -> standardNormal.sf(z)
    }

    // Wald confidence interval using the observed proportion
    val seWald = sqrt(pHat * (1.0 - pHat) / n)
    val alpha = 1.0 - confidenceLevel
    val ci = if (alpha.isNaN()) {
        Pair(Double.NaN, Double.NaN)
    } else when (alternative) {
        Alternative.TWO_SIDED -> {
            val zCrit = standardNormal.quantile(1.0 - alpha / 2.0)
            Pair(pHat - zCrit * seWald, pHat + zCrit * seWald)
        }
        Alternative.LESS -> {
            val zCrit = standardNormal.quantile(1.0 - alpha)
            Pair(0.0, pHat + zCrit * seWald)
        }
        Alternative.GREATER -> {
            val zCrit = standardNormal.quantile(1.0 - alpha)
            Pair(pHat - zCrit * seWald, 1.0)
        }
    }

    return TestResult(
        testName = "One-Sample Proportion z-Test",
        statistic = z,
        pValue = pValue.coerceIn(0.0, 1.0),
        alternative = alternative,
        confidenceInterval = ci,
        additionalInfo = mapOf(
            "observedProportion" to pHat,
            "hypothesizedProportion" to p0,
            "standardError" to se0
        )
    )
}

/**
 * Performs a two-sample proportion z-test for whether two populations have the same proportion.
 *
 * The null hypothesis is that the true proportions in both populations are equal. The test
 * pools the two samples to estimate the common proportion, computes a z-statistic using the
 * pooled standard error, and compares it to the standard normal distribution. This is an
 * asymptotic test appropriate when both samples are large enough for the normal approximation.
 *
 * The confidence interval for the difference in proportions uses the unpooled standard error
 * (each sample's own observed proportion), which is the standard approach for proportion
 * difference intervals.
 *
 * ### Example:
 * ```kotlin
 * val result = proportionZTest(
 *     successes1 = 45, trials1 = 100,
 *     successes2 = 30, trials2 = 80
 * )
 * result.statistic          // z-statistic
 * result.pValue             // two-sided p-value
 * result.confidenceInterval // 95% CI for the difference (p1 - p2)
 * result.isSignificant()    // true if p < 0.05
 * ```
 *
 * @param successes1 the number of observed successes in the first sample. Must be in `[0, trials1]`.
 * @param trials1 the total number of trials in the first sample. Must be positive.
 * @param successes2 the number of observed successes in the second sample. Must be in `[0, trials2]`.
 * @param trials2 the total number of trials in the second sample. Must be positive.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED],
 *   which tests whether the two proportions differ in either direction.
 * @param confidenceLevel the confidence level for the confidence interval on the proportion
 *   difference. Must be in `(0, 1)`. Defaults to `0.95` (95%).
 * @return a [TestResult] containing the z-statistic, p-value, a confidence interval for the
 *   proportion difference (p1 - p2), and additional info with "proportion1", "proportion2",
 *   "proportionDifference", "pooledProportion", and "standardError".
 */
public fun proportionZTest(
    successes1: Int,
    trials1: Int,
    successes2: Int,
    trials2: Int,
    alternative: Alternative = Alternative.TWO_SIDED,
    confidenceLevel: Double = 0.95
): TestResult {
    if (trials1 <= 0) throw InsufficientDataException("trials1 must be positive, got $trials1")
    if (trials2 <= 0) throw InsufficientDataException("trials2 must be positive, got $trials2")
    if (successes1 !in 0..trials1) throw InvalidParameterException(
        "successes1 must be in [0, trials1], got successes1=$successes1, trials1=$trials1"
    )
    if (successes2 !in 0..trials2) throw InvalidParameterException(
        "successes2 must be in [0, trials2], got successes2=$successes2, trials2=$trials2"
    )
    if (confidenceLevel <= 0.0 || confidenceLevel >= 1.0) throw InvalidParameterException(
        "confidenceLevel must be in (0, 1), got $confidenceLevel"
    )

    val n1 = trials1.toDouble()
    val n2 = trials2.toDouble()
    val pHat1 = successes1.toDouble() / n1
    val pHat2 = successes2.toDouble() / n2

    // Pooled proportion under H0: p1 = p2
    val pPool = (successes1 + successes2).toDouble() / (n1 + n2)

    // z-statistic using pooled SE
    val sePool = sqrt(pPool * (1.0 - pPool) * (1.0 / n1 + 1.0 / n2))

    val z = if (sePool == 0.0) {
        val diff = pHat1 - pHat2
        if (diff == 0.0) 0.0 else if (diff > 0) Double.POSITIVE_INFINITY else Double.NEGATIVE_INFINITY
    } else {
        (pHat1 - pHat2) / sePool
    }

    val pValue = if (sePool == 0.0) {
        val diff = pHat1 - pHat2
        if (diff == 0.0) 1.0 else when (alternative) {
            Alternative.TWO_SIDED -> 0.0
            Alternative.LESS -> if (diff < 0) 0.0 else 1.0
            Alternative.GREATER -> if (diff > 0) 0.0 else 1.0
        }
    } else {
        when (alternative) {
            Alternative.TWO_SIDED -> 2.0 * standardNormal.sf(abs(z))
            Alternative.LESS -> standardNormal.cdf(z)
            Alternative.GREATER -> standardNormal.sf(z)
        }
    }

    // CI uses unpooled SE (based on each sample's observed proportion)
    val diff = pHat1 - pHat2
    val seUnpooled = sqrt(pHat1 * (1.0 - pHat1) / n1 + pHat2 * (1.0 - pHat2) / n2)
    val alpha = 1.0 - confidenceLevel
    val ci = if (alpha.isNaN()) {
        Pair(Double.NaN, Double.NaN)
    } else when (alternative) {
        Alternative.TWO_SIDED -> {
            val zCrit = standardNormal.quantile(1.0 - alpha / 2.0)
            Pair(diff - zCrit * seUnpooled, diff + zCrit * seUnpooled)
        }
        Alternative.LESS -> {
            val zCrit = standardNormal.quantile(1.0 - alpha)
            Pair(Double.NEGATIVE_INFINITY, diff + zCrit * seUnpooled)
        }
        Alternative.GREATER -> {
            val zCrit = standardNormal.quantile(1.0 - alpha)
            Pair(diff - zCrit * seUnpooled, Double.POSITIVE_INFINITY)
        }
    }

    return TestResult(
        testName = "Two-Sample Proportion z-Test",
        statistic = z,
        pValue = pValue.coerceIn(0.0, 1.0),
        alternative = alternative,
        confidenceInterval = ci,
        additionalInfo = mapOf(
            "proportion1" to pHat1,
            "proportion2" to pHat2,
            "proportionDifference" to diff,
            "pooledProportion" to pPool,
            "standardError" to sePool
        )
    )
}
