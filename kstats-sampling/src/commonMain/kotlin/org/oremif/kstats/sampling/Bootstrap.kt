package org.oremif.kstats.sampling

import org.oremif.kstats.core.ConfidenceInterval
import org.oremif.kstats.core.erf
import org.oremif.kstats.core.erfInv
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * The result of a bootstrap confidence interval computation, containing three CI methods.
 *
 * All three intervals are computed from the same set of bootstrap resamples, so they can
 * be compared directly without re-running the bootstrap.
 *
 * ### Example:
 * ```kotlin
 * val result = bootstrapCI(
 *     doubleArrayOf(23.0, 25.0, 27.0, 30.0, 32.0, 28.0),
 *     nResamples = 10_000,
 *     random = Random(42),
 * ) { it.average() }
 * result.percentile // ConfidenceInterval(lower=24.83..., upper=30.33...)
 * result.bca        // ConfidenceInterval(lower=24.96..., upper=30.49...)
 * result.basic      // ConfidenceInterval(lower=24.66..., upper=30.16...)
 * ```
 *
 * @property percentile the percentile confidence interval, computed from quantiles of the
 * bootstrap distribution. Simple and intuitive but does not correct for bias or skewness.
 * @property basic the basic (pivotal) confidence interval, which reflects the bootstrap
 * distribution around the observed statistic. Defined as twice the observed statistic minus
 * the opposite-tail quantile of the bootstrap distribution.
 * @property bca the bias-corrected and accelerated (BCa) confidence interval. Adjusts for
 * both median bias and skewness of the bootstrap distribution using a jackknife acceleration
 * factor. Generally the most accurate of the three methods.
 * @property observedStatistic the value of the statistic computed on the original data.
 * @property nResamples the number of bootstrap resamples that were generated.
 * @property confidenceLevel the confidence level used to compute the intervals.
 */
public data class BootstrapCIResult(
    public val percentile: ConfidenceInterval,
    public val basic: ConfidenceInterval,
    public val bca: ConfidenceInterval,
    public val observedStatistic: Double,
    public val nResamples: Int,
    public val confidenceLevel: Double,
)

/**
 * Computes bootstrap confidence intervals for a statistic using three methods: percentile,
 * basic (pivotal), and BCa (bias-corrected and accelerated).
 *
 * The bootstrap works by repeatedly resampling the data with replacement, computing the
 * statistic on each resample, and using the distribution of those values to estimate the
 * uncertainty of the statistic. The BCa method additionally uses a jackknife (leave-one-out)
 * procedure to correct for bias and skewness.
 *
 * ### Example:
 * ```kotlin
 * val data = doubleArrayOf(23.0, 25.0, 27.0, 30.0, 32.0, 28.0)
 * val ci = bootstrapCI(data, nResamples = 10_000, random = Random(42)) { it.average() }
 * ci.percentile // percentile CI
 * ci.bca        // BCa CI (most accurate for skewed data)
 * ci.basic      // basic (pivotal) CI
 * ```
 *
 * @param data the observed data to bootstrap from.
 * @param nResamples the number of bootstrap resamples to generate. Defaults to `10_000`.
 * More resamples produce more stable intervals at the cost of computation time.
 * @param confidenceLevel the confidence level for the intervals, in (0, 1) exclusive.
 * Defaults to `0.95` (95%).
 * @param random the random number generator used for resampling. Defaults to [Random].
 * Pass a seeded instance (e.g. `Random(42)`) for reproducible results.
 * @param statistic a function that computes the statistic of interest from a [DoubleArray].
 * @return a [BootstrapCIResult] containing all three confidence intervals, the observed
 * statistic, and metadata.
 */
public fun bootstrapCI(
    data: DoubleArray,
    nResamples: Int = 10_000,
    confidenceLevel: Double = 0.95,
    random: Random = Random,
    statistic: (DoubleArray) -> Double,
): BootstrapCIResult {
    if (data.isEmpty()) throw InsufficientDataException("data must not be empty")
    if (nResamples < 1) throw InvalidParameterException("nResamples must be at least 1, got $nResamples")
    if (confidenceLevel.isNaN() || confidenceLevel <= 0.0 || confidenceLevel >= 1.0) throw InvalidParameterException(
        "confidenceLevel must be in (0, 1), got $confidenceLevel"
    )

    val n = data.size
    val observed = statistic(data)

    // Generate bootstrap distribution
    val bootstrapStats = DoubleArray(nResamples)
    val resample = DoubleArray(n)
    for (b in 0 until nResamples) {
        for (i in 0 until n) {
            resample[i] = data[random.nextInt(n)]
        }
        bootstrapStats[b] = statistic(resample)
    }
    bootstrapStats.sort()

    // Jackknife leave-one-out estimates for BCa acceleration factor
    val jackknife = DoubleArray(n)
    val leaveOneOut = DoubleArray(n - 1)
    for (i in 0 until n) {
        var k = 0
        for (j in 0 until n) {
            if (j != i) leaveOneOut[k++] = data[j]
        }
        jackknife[i] = statistic(leaveOneOut)
    }

    return buildResult(bootstrapStats, observed, confidenceLevel, jackknife, nResamples)
}

/**
 * Computes bootstrap confidence intervals for a statistic on a list of arbitrary elements,
 * using three methods: percentile, basic (pivotal), and BCa (bias-corrected and accelerated).
 *
 * This overload accepts any `List<T>`, making it suitable for bootstrapping statistics on
 * structured data (e.g. weighted measurements, records). The [statistic] function receives
 * a `List<T>` resample and must return a scalar `Double`.
 *
 * ### Example:
 * ```kotlin
 * data class Obs(val value: Double, val weight: Double)
 * val data = listOf(Obs(1.0, 0.5), Obs(2.0, 1.0), Obs(3.0, 1.5))
 * val ci = bootstrapCI(data, nResamples = 10_000, random = Random(42)) { sample ->
 *     sample.sumOf { it.value * it.weight } / sample.sumOf { it.weight }
 * }
 * ci.bca // BCa CI for the weighted mean
 * ```
 *
 * @param T the element type of the data list.
 * @param data the observed data to bootstrap from.
 * @param nResamples the number of bootstrap resamples to generate. Defaults to `10_000`.
 * More resamples produce more stable intervals at the cost of computation time.
 * @param confidenceLevel the confidence level for the intervals, in (0, 1) exclusive.
 * Defaults to `0.95` (95%).
 * @param random the random number generator used for resampling. Defaults to [Random].
 * Pass a seeded instance (e.g. `Random(42)`) for reproducible results.
 * @param statistic a function that computes the statistic of interest from a `List<T>`.
 * @return a [BootstrapCIResult] containing all three confidence intervals, the observed
 * statistic, and metadata.
 */
public fun <T> bootstrapCI(
    data: List<T>,
    nResamples: Int = 10_000,
    confidenceLevel: Double = 0.95,
    random: Random = Random,
    statistic: (List<T>) -> Double,
): BootstrapCIResult {
    if (data.isEmpty()) throw InsufficientDataException("data must not be empty")
    if (nResamples < 1) throw InvalidParameterException("nResamples must be at least 1, got $nResamples")
    if (confidenceLevel.isNaN() || confidenceLevel <= 0.0 || confidenceLevel >= 1.0) throw InvalidParameterException(
        "confidenceLevel must be in (0, 1), got $confidenceLevel"
    )

    val n = data.size
    val observed = statistic(data)

    // Generate bootstrap distribution
    val bootstrapStats = DoubleArray(nResamples)
    for (b in 0 until nResamples) {
        val resample = List(n) { data[random.nextInt(n)] }
        bootstrapStats[b] = statistic(resample)
    }
    bootstrapStats.sort()

    // Jackknife leave-one-out estimates for BCa acceleration factor
    val jackknife = DoubleArray(n)
    for (i in 0 until n) {
        val leaveOneOut = ArrayList<T>(n - 1)
        for (j in 0 until n) {
            if (j != i) leaveOneOut.add(data[j])
        }
        jackknife[i] = statistic(leaveOneOut)
    }

    return buildResult(bootstrapStats, observed, confidenceLevel, jackknife, nResamples)
}

// ── shared CI computation ──────────────────────────────────────────────────

private fun buildResult(
    sortedBootstrapStats: DoubleArray,
    observed: Double,
    confidenceLevel: Double,
    jackknife: DoubleArray,
    nResamples: Int,
): BootstrapCIResult {
    val alpha = 1.0 - confidenceLevel

    // Percentile CI: quantiles of the bootstrap distribution
    val pLower = sortedQuantile(sortedBootstrapStats, alpha / 2.0)
    val pUpper = sortedQuantile(sortedBootstrapStats, 1.0 - alpha / 2.0)

    // Basic (pivotal) CI: 2θ̂ − θ*(1−α/2), 2θ̂ − θ*(α/2)
    val bLower = 2.0 * observed - pUpper
    val bUpper = 2.0 * observed - pLower

    // BCa CI: bias-corrected and accelerated
    val bca = computeBca(sortedBootstrapStats, observed, alpha, jackknife)

    return BootstrapCIResult(
        percentile = ConfidenceInterval(pLower, pUpper),
        basic = ConfidenceInterval(bLower, bUpper),
        bca = bca,
        observedStatistic = observed,
        nResamples = nResamples,
        confidenceLevel = confidenceLevel,
    )
}

// ── BCa ────────────────────────────────────────────────────────────────────

private fun computeBca(
    sortedStats: DoubleArray,
    observed: Double,
    alpha: Double,
    jackknife: DoubleArray,
): ConfidenceInterval {
    val nBoot = sortedStats.size
    val n = jackknife.size

    // Bias correction: z0 = Φ⁻¹(proportion of bootstrap stats < observed)
    var countBelow = 0
    for (stat in sortedStats) {
        if (stat < observed) countBelow++
    }
    val proportion = countBelow.toDouble() / nBoot

    // If all stats are on one side, BCa is undefined — fall back to percentile
    if (proportion == 0.0 || proportion == 1.0) {
        return ConfidenceInterval(
            lower = sortedQuantile(sortedStats, alpha / 2.0),
            upper = sortedQuantile(sortedStats, 1.0 - alpha / 2.0),
        )
    }
    val z0 = normalQuantile(proportion)

    // Acceleration factor via jackknife (Neumaier compensated mean)
    var jackSum = 0.0
    var compensation = 0.0
    for (v in jackknife) {
        val t = jackSum + v
        compensation += if (abs(jackSum) >= abs(v)) (jackSum - t) + v else (v - t) + jackSum
        jackSum = t
    }
    val jackMean = (jackSum + compensation) / n

    var sumCubed = 0.0
    var sumSquared = 0.0
    for (v in jackknife) {
        val diff = jackMean - v
        val diffSq = diff * diff
        sumSquared += diffSq
        sumCubed += diffSq * diff
    }

    // â = Σ(θ̄ − θ̂₍₋ᵢ₎)³ / [6 · (Σ(θ̄ − θ̂₍₋ᵢ₎)²)^(3/2)]
    // If all jackknife estimates are equal, acceleration is zero (no skewness)
    val aHat = if (sumSquared == 0.0) 0.0 else sumCubed / (6.0 * sumSquared * sqrt(sumSquared))

    // Adjusted quantile levels
    val zLow = normalQuantile(alpha / 2.0)
    val zHigh = normalQuantile(1.0 - alpha / 2.0)

    val alpha1 = normalCdf(z0 + (z0 + zLow) / (1.0 - aHat * (z0 + zLow)))
    val alpha2 = normalCdf(z0 + (z0 + zHigh) / (1.0 - aHat * (z0 + zHigh)))

    return ConfidenceInterval(
        lower = sortedQuantile(sortedStats, alpha1.coerceIn(0.0, 1.0)),
        upper = sortedQuantile(sortedStats, alpha2.coerceIn(0.0, 1.0)),
    )
}

// ── internal helpers ───────────────────────────────────────────────────────

private val SQRT_2 = sqrt(2.0)

/** Standard normal CDF: Φ(x) = 0.5 · (1 + erf(x / √2)) */
private fun normalCdf(x: Double): Double = 0.5 * (1.0 + erf(x / SQRT_2))

/** Standard normal quantile (probit): Φ⁻¹(p) = √2 · erfInv(2p − 1) */
private fun normalQuantile(p: Double): Double = SQRT_2 * erfInv(2.0 * p - 1.0)

/** Linear interpolation from a sorted array. */
private fun sortedQuantile(sorted: DoubleArray, q: Double): Double {
    if (sorted.size == 1) return sorted[0]
    val pos = q * (sorted.size - 1)
    val lo = floor(pos).toInt().coerceIn(0, sorted.lastIndex)
    val hi = minOf(lo + 1, sorted.lastIndex)
    val frac = pos - lo
    return sorted[lo] + frac * (sorted[hi] - sorted[lo])
}
