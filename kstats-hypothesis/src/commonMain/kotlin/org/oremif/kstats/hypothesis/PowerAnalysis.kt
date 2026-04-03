package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.distributions.NormalDistribution
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.sqrt

/**
 * Specifies the design of the t-test for power analysis calculations.
 *
 * The test type determines how sample size [n] is interpreted and how the noncentrality
 * parameter is computed. For [TWO_SAMPLE], [n] is the per-group sample size and the
 * noncentrality scales by the square root of n/2. For [ONE_SAMPLE] and [PAIRED], [n]
 * is the total sample size (or number of pairs) and the noncentrality scales by the
 * square root of n.
 */
public enum class TTestType {
    /**
     * A one-sample t-test comparing a single sample mean to a hypothesized value.
     *
     * The parameter [n] represents the total sample size.
     */
    ONE_SAMPLE,

    /**
     * A two-sample (independent) t-test comparing the means of two independent groups.
     *
     * The parameter [n] represents the per-group sample size, assuming equal-sized groups.
     */
    TWO_SAMPLE,

    /**
     * A paired t-test comparing matched observations (e.g. before/after measurements).
     *
     * The parameter [n] represents the number of pairs. Uses the same noncentrality
     * factor as [ONE_SAMPLE].
     */
    PAIRED
}

private val standardNormal = NormalDistribution(0.0, 1.0)

// ── T-test power analysis ──────────────────────────────────────────────────

/**
 * Computes the statistical power of a t-test using the normal approximation.
 *
 * Power is the probability of correctly rejecting the null hypothesis when the true
 * effect size equals [effectSize]. Higher power means the test is more likely to detect
 * a real effect. A power of 0.8 (80%) is the most common target in study design.
 *
 * Uses Cohen's d as the effect size metric. Conventional thresholds: 0.2 (small),
 * 0.5 (medium), 0.8 (large). Negative effect sizes are treated as their absolute value.
 * When [effectSize] is zero, the returned power equals [alpha] (the Type I error rate).
 *
 * ### Example:
 * ```kotlin
 * // Power of a two-sample t-test with medium effect, 64 per group
 * tTestPower(effectSize = 0.5, n = 64)                    // ~0.807
 * tTestPower(effectSize = 0.5, n = 64, type = TTestType.ONE_SAMPLE) // ~0.979
 * tTestPower(effectSize = 0.5, n = 64, alternative = Alternative.GREATER) // ~0.882
 * ```
 *
 * @param effectSize the expected Cohen's d effect size. The sign is ignored.
 * @param n the sample size. For [TTestType.TWO_SAMPLE], this is the per-group size
 *   (assuming equal groups). For [TTestType.ONE_SAMPLE] and [TTestType.PAIRED], this is
 *   the total sample size or number of pairs. Must be at least 2.
 * @param alpha the significance level (Type I error rate). Defaults to `0.05` (5%).
 *   Must be in `(0, 1)`.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED].
 *   One-sided tests have higher power for the same parameters.
 * @param type the t-test design. Defaults to [TTestType.TWO_SAMPLE].
 * @return the statistical power, a value in `[0, 1]`.
 * @see tTestRequiredN to find the sample size needed for a target power.
 * @see tTestMinimumEffect to find the smallest detectable effect at a given sample size.
 */
public fun tTestPower(
    effectSize: Double,
    n: Int,
    alpha: Double = 0.05,
    alternative: Alternative = Alternative.TWO_SIDED,
    type: TTestType = TTestType.TWO_SAMPLE,
): Double {
    if (n < 2) throw InvalidParameterException("n must be at least 2, got $n")
    if (alpha <= 0.0 || alpha >= 1.0) throw InvalidParameterException("alpha must be in (0, 1), got $alpha")

    val d = abs(effectSize)
    val lambda = d * tTestNoncentralityFactor(n, type)
    return computePower(lambda, alpha, alternative)
}

/**
 * Computes the minimum sample size needed for a t-test to achieve the desired power.
 *
 * Given a target [power] and expected [effectSize] (Cohen's d), returns the smallest
 * integer sample size that meets or exceeds the requested power at significance level
 * [alpha]. The result is rounded up to the next whole number, with a minimum of 2.
 *
 * ### Example:
 * ```kotlin
 * // Sample size per group for a two-sample t-test
 * tTestRequiredN(effectSize = 0.5, power = 0.8)             // 64
 * tTestRequiredN(effectSize = 0.5, power = 0.8, type = TTestType.ONE_SAMPLE) // 32
 * tTestRequiredN(effectSize = 0.8, power = 0.9, alpha = 0.01) // 38
 * ```
 *
 * @param effectSize the expected Cohen's d effect size. Must be non-zero. The sign is ignored.
 * @param power the target statistical power (probability of detecting the effect).
 *   Defaults to `0.8` (80%). Must be in `(0, 1)`.
 * @param alpha the significance level (Type I error rate). Defaults to `0.05` (5%).
 *   Must be in `(0, 1)`.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED].
 * @param type the t-test design. Defaults to [TTestType.TWO_SAMPLE].
 * @return the minimum required sample size (per group for [TTestType.TWO_SAMPLE], total for
 *   [TTestType.ONE_SAMPLE] and [TTestType.PAIRED]). Always at least 2.
 * @see tTestPower to compute power at a given sample size.
 * @see tTestMinimumEffect to find the smallest detectable effect at a given sample size.
 */
public fun tTestRequiredN(
    effectSize: Double,
    power: Double = 0.8,
    alpha: Double = 0.05,
    alternative: Alternative = Alternative.TWO_SIDED,
    type: TTestType = TTestType.TWO_SAMPLE,
): Int {
    if (effectSize == 0.0) throw InvalidParameterException("effectSize must be non-zero")
    if (power <= 0.0 || power >= 1.0) throw InvalidParameterException("power must be in (0, 1), got $power")
    if (alpha <= 0.0 || alpha >= 1.0) throw InvalidParameterException("alpha must be in (0, 1), got $alpha")

    val d = abs(effectSize)
    val lambda = computeNoncentrality(power, alpha, alternative)
    val nRaw = when (type) {
        TTestType.ONE_SAMPLE, TTestType.PAIRED -> (lambda / d) * (lambda / d)
        TTestType.TWO_SAMPLE -> 2.0 * (lambda / d) * (lambda / d)
    }
    return maxOf(2, ceil(nRaw).toInt())
}

/**
 * Computes the minimum detectable effect size (MDE) for a t-test at the given sample size.
 *
 * Returns the smallest Cohen's d that a t-test with [n] observations can detect at the
 * specified [power] and [alpha]. This is useful for assessing whether an existing dataset
 * is large enough to detect practically meaningful effects.
 *
 * ### Example:
 * ```kotlin
 * // Smallest effect detectable with 100 subjects per group
 * tTestMinimumEffect(n = 100, power = 0.8)             // ~0.398
 * tTestMinimumEffect(n = 100, type = TTestType.ONE_SAMPLE) // ~0.281
 * ```
 *
 * @param n the sample size. For [TTestType.TWO_SAMPLE], this is the per-group size.
 *   For [TTestType.ONE_SAMPLE] and [TTestType.PAIRED], this is the total size or number
 *   of pairs. Must be at least 2.
 * @param power the target statistical power. Defaults to `0.8` (80%). Must be in `(0, 1)`.
 * @param alpha the significance level (Type I error rate). Defaults to `0.05` (5%).
 *   Must be in `(0, 1)`.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED].
 * @param type the t-test design. Defaults to [TTestType.TWO_SAMPLE].
 * @return the minimum detectable Cohen's d effect size. Always positive.
 * @see tTestPower to compute power at a given effect size.
 * @see tTestRequiredN to find the sample size needed for a target power.
 */
public fun tTestMinimumEffect(
    n: Int,
    power: Double = 0.8,
    alpha: Double = 0.05,
    alternative: Alternative = Alternative.TWO_SIDED,
    type: TTestType = TTestType.TWO_SAMPLE,
): Double {
    if (n < 2) throw InvalidParameterException("n must be at least 2, got $n")
    if (power <= 0.0 || power >= 1.0) throw InvalidParameterException("power must be in (0, 1), got $power")
    if (alpha <= 0.0 || alpha >= 1.0) throw InvalidParameterException("alpha must be in (0, 1), got $alpha")

    val lambda = computeNoncentrality(power, alpha, alternative)
    return lambda / tTestNoncentralityFactor(n, type)
}

// ── Proportion z-test power analysis ───────────────────────────────────────

/**
 * Computes the statistical power of a proportion z-test using the normal approximation.
 *
 * Power is the probability of correctly rejecting the null hypothesis when the true
 * effect size equals [effectSize]. Uses Cohen's h as the effect size metric, which is
 * based on the arcsine transformation of proportions. Conventional thresholds: 0.2 (small),
 * 0.5 (medium), 0.8 (large). Negative effect sizes are treated as their absolute value.
 *
 * ### Example:
 * ```kotlin
 * // Power of a two-sample proportion z-test with 64 per group
 * proportionZTestPower(effectSize = 0.5, n = 64)                     // ~0.807
 * proportionZTestPower(effectSize = 0.5, n = 64, twoSample = false)  // ~0.979
 * ```
 *
 * @param effectSize the expected Cohen's h effect size. The sign is ignored.
 * @param n the sample size. For two-sample tests ([twoSample] = `true`), this is the
 *   per-group size assuming equal groups. For one-sample tests, this is the total sample
 *   size. Must be at least 2.
 * @param alpha the significance level (Type I error rate). Defaults to `0.05` (5%).
 *   Must be in `(0, 1)`.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED].
 * @param twoSample whether this is a two-sample test. Defaults to `true`. When `false`,
 *   uses the one-sample noncentrality factor (higher power for the same n).
 * @return the statistical power, a value in `[0, 1]`.
 * @see proportionZTestRequiredN to find the sample size needed for a target power.
 * @see proportionZTestMinimumEffect to find the smallest detectable effect at a given sample size.
 * @see cohensH to compute Cohen's h from two proportions.
 */
public fun proportionZTestPower(
    effectSize: Double,
    n: Int,
    alpha: Double = 0.05,
    alternative: Alternative = Alternative.TWO_SIDED,
    twoSample: Boolean = true,
): Double {
    if (n < 2) throw InvalidParameterException("n must be at least 2, got $n")
    if (alpha <= 0.0 || alpha >= 1.0) throw InvalidParameterException("alpha must be in (0, 1), got $alpha")

    val h = abs(effectSize)
    val lambda = h * proportionNoncentralityFactor(n, twoSample)
    return computePower(lambda, alpha, alternative)
}

/**
 * Computes the minimum sample size needed for a proportion z-test to achieve the desired power.
 *
 * Given a target [power] and expected [effectSize] (Cohen's h), returns the smallest
 * integer sample size that meets or exceeds the requested power at significance level
 * [alpha]. The result is rounded up to the next whole number, with a minimum of 2.
 *
 * ### Example:
 * ```kotlin
 * // Per-group sample size for a two-sample proportion z-test
 * proportionZTestRequiredN(effectSize = 0.5, power = 0.8) // 63
 * // Total sample size for a one-sample proportion z-test
 * proportionZTestRequiredN(effectSize = 0.5, power = 0.8, twoSample = false) // 32
 * ```
 *
 * @param effectSize the expected Cohen's h effect size. Must be non-zero. The sign is ignored.
 * @param power the target statistical power (probability of detecting the effect).
 *   Defaults to `0.8` (80%). Must be in `(0, 1)`.
 * @param alpha the significance level (Type I error rate). Defaults to `0.05` (5%).
 *   Must be in `(0, 1)`.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED].
 * @param twoSample whether this is a two-sample test. Defaults to `true`. When `true`,
 *   the returned value is the per-group sample size. When `false`, it is the total sample size.
 * @return the minimum required sample size. Always at least 2.
 * @see proportionZTestPower to compute power at a given sample size.
 * @see proportionZTestMinimumEffect to find the smallest detectable effect at a given sample size.
 * @see cohensH to compute Cohen's h from two proportions.
 */
public fun proportionZTestRequiredN(
    effectSize: Double,
    power: Double = 0.8,
    alpha: Double = 0.05,
    alternative: Alternative = Alternative.TWO_SIDED,
    twoSample: Boolean = true,
): Int {
    if (effectSize == 0.0) throw InvalidParameterException("effectSize must be non-zero")
    if (power <= 0.0 || power >= 1.0) throw InvalidParameterException("power must be in (0, 1), got $power")
    if (alpha <= 0.0 || alpha >= 1.0) throw InvalidParameterException("alpha must be in (0, 1), got $alpha")

    val h = abs(effectSize)
    val lambda = computeNoncentrality(power, alpha, alternative)
    val lambdaOverH = lambda / h
    val nRaw = if (twoSample) {
        2.0 * lambdaOverH * lambdaOverH
    } else {
        lambdaOverH * lambdaOverH
    }
    return maxOf(2, ceil(nRaw).toInt())
}

/**
 * Computes the minimum detectable effect size (MDE) for a proportion z-test at the given sample size.
 *
 * Returns the smallest Cohen's h that a proportion z-test with [n] observations can detect
 * at the specified [power] and [alpha]. This is useful for assessing whether an existing
 * dataset or planned study is large enough to detect practically meaningful differences
 * between proportions.
 *
 * ### Example:
 * ```kotlin
 * // Smallest effect detectable with 100 subjects per group (two-sample)
 * proportionZTestMinimumEffect(n = 100, power = 0.8)                    // ~0.398
 * proportionZTestMinimumEffect(n = 100, power = 0.8, twoSample = false) // ~0.281
 * ```
 *
 * @param n the sample size. For two-sample tests ([twoSample] = `true`), this is the
 *   per-group size. For one-sample tests, this is the total sample size. Must be at least 2.
 * @param power the target statistical power. Defaults to `0.8` (80%). Must be in `(0, 1)`.
 * @param alpha the significance level (Type I error rate). Defaults to `0.05` (5%).
 *   Must be in `(0, 1)`.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED].
 * @param twoSample whether this is a two-sample test. Defaults to `true`.
 * @return the minimum detectable Cohen's h effect size. Always positive.
 * @see proportionZTestPower to compute power at a given effect size.
 * @see proportionZTestRequiredN to find the sample size needed for a target power.
 * @see cohensH to compute Cohen's h from two proportions.
 */
public fun proportionZTestMinimumEffect(
    n: Int,
    power: Double = 0.8,
    alpha: Double = 0.05,
    alternative: Alternative = Alternative.TWO_SIDED,
    twoSample: Boolean = true,
): Double {
    if (n < 2) throw InvalidParameterException("n must be at least 2, got $n")
    if (power <= 0.0 || power >= 1.0) throw InvalidParameterException("power must be in (0, 1), got $power")
    if (alpha <= 0.0 || alpha >= 1.0) throw InvalidParameterException("alpha must be in (0, 1), got $alpha")

    val lambda = computeNoncentrality(power, alpha, alternative)
    return lambda / proportionNoncentralityFactor(n, twoSample)
}

// ── Internal helpers ───────────────────────────────────────────────────────

// λ = d * factor, where factor depends on test type and sample size
private fun tTestNoncentralityFactor(n: Int, type: TTestType): Double = when (type) {
    TTestType.ONE_SAMPLE, TTestType.PAIRED -> sqrt(n.toDouble())
    TTestType.TWO_SAMPLE -> sqrt(n.toDouble() / 2.0)
}

// λ = h * factor, where factor depends on one-sample vs two-sample
private fun proportionNoncentralityFactor(n: Int, twoSample: Boolean): Double =
    if (twoSample) sqrt(n.toDouble() / 2.0) else sqrt(n.toDouble())

// Normal approximation: power = Φ(λ - z_crit) + Φ(-λ - z_crit) for two-sided
private fun computePower(lambda: Double, alpha: Double, alternative: Alternative): Double {
    return when (alternative) {
        Alternative.TWO_SIDED -> {
            val zCrit = standardNormal.quantile(1.0 - alpha / 2.0)
            (standardNormal.cdf(lambda - zCrit) + standardNormal.cdf(-lambda - zCrit))
                .coerceIn(0.0, 1.0)
        }

        Alternative.LESS, Alternative.GREATER -> {
            val zCrit = standardNormal.quantile(1.0 - alpha)
            standardNormal.cdf(lambda - zCrit).coerceIn(0.0, 1.0)
        }
    }
}

// Inverse: given desired power, compute noncentrality λ ≈ z_crit + Φ⁻¹(power)
private fun computeNoncentrality(power: Double, alpha: Double, alternative: Alternative): Double {
    val zBeta = standardNormal.quantile(power)
    val zCrit = when (alternative) {
        Alternative.TWO_SIDED -> standardNormal.quantile(1.0 - alpha / 2.0)
        Alternative.LESS, Alternative.GREATER -> standardNormal.quantile(1.0 - alpha)
    }
    return zCrit + zBeta
}
