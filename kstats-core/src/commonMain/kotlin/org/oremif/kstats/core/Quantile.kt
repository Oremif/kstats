package org.oremif.kstats.core

import kotlin.math.abs
import kotlin.math.max

private const val NEWTON_MAX_ITERATIONS = 50
private const val BISECTION_MAX_ITERATIONS = 100
private const val RELATIVE_TOLERANCE = 1e-12

/**
 * Finds the quantile value x such that cdf(x) = [p], using Newton-Raphson with bisection fallback.
 *
 * This is the shared numerical root-finder used by distribution quantile methods. It first
 * attempts Newton-Raphson iteration (up to 50 steps) for fast quadratic convergence, then
 * falls back to bisection (up to 100 steps) if Newton stalls or the PDF is zero at the
 * current estimate. For unbounded distributions, the bisection phase automatically expands
 * the search bracket until it contains the target probability.
 *
 * ### Example:
 * ```kotlin
 * // Find the median of the standard normal distribution
 * val median = findQuantile(
 *     p = 0.5,
 *     cdf = { x -> 0.5 * (1.0 + erf(x / sqrt(2.0))) },
 *     pdf = { x -> exp(-x * x / 2.0) / sqrt(2.0 * PI) },
 *     initialGuess = 0.0,
 * ) // 0.0
 * ```
 *
 * @param p the target cumulative probability, typically in (0, 1).
 * @param cdf the cumulative distribution function.
 * @param pdf the probability density function, used by the Newton-Raphson step.
 * @param initialGuess a starting estimate for x. A good guess improves convergence speed.
 * @param lowerBound the lower bound of the distribution's support. Defaults to negative infinity.
 * @param upperBound the upper bound of the distribution's support. Defaults to positive infinity.
 * @return the value x such that cdf(x) is approximately equal to [p].
 * @throws org.oremif.kstats.core.exceptions.ConvergenceException if neither Newton-Raphson nor bisection converges within the
 * iteration limits.
 */
public fun findQuantile(
    p: Double,
    cdf: (Double) -> Double,
    pdf: (Double) -> Double,
    initialGuess: Double,
    lowerBound: Double = Double.NEGATIVE_INFINITY,
    upperBound: Double = Double.POSITIVE_INFINITY,
): Double {
    var x = initialGuess
    var totalIterations = 0

    // Phase 1 — Newton-Raphson
    for (i in 0 until NEWTON_MAX_ITERATIONS) {
        totalIterations++
        val pdfVal = pdf(x)
        if (pdfVal == 0.0) break // switch to bisection

        val delta = (cdf(x) - p) / pdfVal
        x -= delta

        // Clamp to bounds
        if (x < lowerBound) x = lowerBound
        if (x > upperBound) x = upperBound

        if (abs(delta) < RELATIVE_TOLERANCE * max(abs(x), 1.0)) {
            return x
        }
    }

    // Phase 2 — Bisection fallback
    var lo = lowerBound
    var hi = upperBound

    // Establish finite bracket if bounds are infinite
    if (lo.isInfinite() || hi.isInfinite()) {
        // Widen from current best x
        var width = max(abs(x), 1.0)
        if (lo.isInfinite()) lo = x - width
        if (hi.isInfinite()) hi = x + width

        // Expand until bracket contains p
        for (expansion in 0 until 50) {
            val cdfLo = cdf(lo)
            val cdfHi = cdf(hi)
            if (cdfLo <= p && cdfHi >= p) break
            width *= 2.0
            if (cdfLo > p) lo -= width
            if (cdfHi < p) hi += width
        }
    }

    // Ensure bracket is valid: cdf(lo) <= p <= cdf(hi)
    if (cdf(lo) > p) lo = lowerBound.coerceAtLeast(lo - abs(lo) - 1.0)
    if (cdf(hi) < p) hi = upperBound.coerceAtMost(hi + abs(hi) + 1.0)

    for (i in 0 until BISECTION_MAX_ITERATIONS) {
        totalIterations++
        val mid = (lo + hi) / 2.0

        if (abs(hi - lo) < RELATIVE_TOLERANCE * max(max(abs(lo), abs(hi)), 1.0)) {
            return mid
        }

        if (cdf(mid) < p) {
            lo = mid
        } else {
            hi = mid
        }
    }

    // Phase 3 — Failure
    val bestEstimate = (lo + hi) / 2.0
    checkConvergence(false, totalIterations, bestEstimate) {
        "findQuantile did not converge for p=$p after $totalIterations iterations"
    }

    // unreachable, checkConvergence throws
    return bestEstimate
}
