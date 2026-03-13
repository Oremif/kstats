package org.oremif.kstats.core

import org.oremif.kstats.core.exceptions.ConvergenceException
import kotlin.math.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class QuantileTest {

    private val TOLERANCE = 1e-10

    // ── Helper: Standard Normal cdf/pdf ──────────────────────────────────

    private fun normalCdf(x: Double): Double = 0.5 * (1.0 + erf(x / sqrt(2.0)))

    private fun normalPdf(x: Double): Double = exp(-x * x / 2.0) / sqrt(2.0 * PI)

    /** Analytical normal quantile via erfInv for reference comparison. */
    private fun normalQuantileAnalytical(p: Double): Double = sqrt(2.0) * erfInv(2.0 * p - 1.0)

    // ── Normal quantile via findQuantile ─────────────────────────────────

    @Test
    fun testNormalQuantileStandardValues() {
        val pValues = doubleArrayOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        for (p in pValues) {
            val expected = normalQuantileAnalytical(p)
            val result = findQuantile(
                p = p,
                cdf = ::normalCdf,
                pdf = ::normalPdf,
                initialGuess = 0.0,
            )
            assertEquals(expected, result, TOLERANCE, "Normal quantile mismatch for p=$p")
        }
    }

    @Test
    fun testNormalQuantileMedian() {
        val result = findQuantile(
            p = 0.5,
            cdf = ::normalCdf,
            pdf = ::normalPdf,
            initialGuess = 0.0,
        )
        assertEquals(0.0, result, TOLERANCE)
    }

    // ── Bounded distribution (Beta-like on [0, 1]) ──────────────────────

    /** Simple Beta(2,2)-like: pdf(x) = 6x(1-x), cdf(x) = 3x^2 - 2x^3 on [0,1]. */
    private fun betaCdf(x: Double): Double {
        if (x <= 0.0) return 0.0
        if (x >= 1.0) return 1.0
        return 3.0 * x * x - 2.0 * x * x * x
    }

    private fun betaPdf(x: Double): Double {
        if (x <= 0.0 || x >= 1.0) return 0.0
        return 6.0 * x * (1.0 - x)
    }

    @Test
    fun testBoundedDistributionConvergence() {
        // Beta(2,2) median is exactly 0.5
        val result = findQuantile(
            p = 0.5,
            cdf = ::betaCdf,
            pdf = ::betaPdf,
            initialGuess = 0.5,
            lowerBound = 0.0,
            upperBound = 1.0,
        )
        assertEquals(0.5, result, TOLERANCE)
    }

    @Test
    fun testBoundedDistributionVariousQuantiles() {
        // For cdf(x) = 3x^2 - 2x^3 = p, solve numerically and verify round-trip
        val pValues = doubleArrayOf(0.1, 0.25, 0.5, 0.75, 0.9)
        for (p in pValues) {
            val x = findQuantile(
                p = p,
                cdf = ::betaCdf,
                pdf = ::betaPdf,
                initialGuess = 0.5,
                lowerBound = 0.0,
                upperBound = 1.0,
            )
            // Round-trip: cdf(quantile(p)) should equal p
            assertEquals(p, betaCdf(x), TOLERANCE, "Round-trip failed for p=$p")
        }
    }

    // ── Edge cases: p near 0 and 1 ──────────────────────────────────────

    @Test
    fun testNormalQuantileExtremeP() {
        for (p in doubleArrayOf(1e-10, 1.0 - 1e-10)) {
            val expected = normalQuantileAnalytical(p)
            val result = findQuantile(
                p = p,
                cdf = ::normalCdf,
                pdf = ::normalPdf,
                initialGuess = 0.0,
            )
            // Relaxed tolerance for extreme tails
            assertEquals(expected, result, 1e-6, "Extreme quantile mismatch for p=$p")
        }
    }

    @Test
    fun testBoundedDistributionEdgeP() {
        // Very close to lower bound
        val lowResult = findQuantile(
            p = 0.001,
            cdf = ::betaCdf,
            pdf = ::betaPdf,
            initialGuess = 0.5,
            lowerBound = 0.0,
            upperBound = 1.0,
        )
        assertEquals(0.001, betaCdf(lowResult), TOLERANCE)

        // Very close to upper bound
        val highResult = findQuantile(
            p = 0.999,
            cdf = ::betaCdf,
            pdf = ::betaPdf,
            initialGuess = 0.5,
            lowerBound = 0.0,
            upperBound = 1.0,
        )
        assertEquals(0.999, betaCdf(highResult), TOLERANCE)
    }

    // ── Bisection fallback (pdf returns 0) ──────────────────────────────

    @Test
    fun testBisectionFallbackWhenPdfIsZero() {
        // Use a cdf that is valid but pdf always returns 0 → forces bisection
        val result = findQuantile(
            p = 0.5,
            cdf = ::betaCdf,
            pdf = { 0.0 }, // always zero → Newton cannot work
            initialGuess = 0.5,
            lowerBound = 0.0,
            upperBound = 1.0,
        )
        assertEquals(0.5, result, 1e-10, "Bisection should find median of Beta(2,2)")
    }

    @Test
    fun testBisectionFallbackWithUnboundedDistribution() {
        // Normal with pdf=0 → forces bisection with bracket expansion
        val result = findQuantile(
            p = 0.5,
            cdf = ::normalCdf,
            pdf = { 0.0 },
            initialGuess = 0.0,
        )
        assertEquals(0.0, result, 1e-6, "Bisection should find normal median")
    }

    // ── Convergence exception ───────────────────────────────────────────

    @Test
    fun testConvergenceExceptionOnPathologicalFunction() {
        // Step function with extremely wide finite bounds — bisection cannot narrow
        // the bracket to relative tolerance within 100 iterations at this scale.
        val exception = assertFailsWith<ConvergenceException> {
            findQuantile(
                p = 0.5,
                cdf = { x -> if (x < 0.0) 0.0 else 1.0 },
                pdf = { 0.0 },
                initialGuess = -1e200,
                lowerBound = -1e300,
                upperBound = 1e300,
            )
        }
        kotlin.test.assertTrue(exception.iterations > 0, "Expected positive iteration count")
        kotlin.test.assertTrue(exception.lastEstimate.isFinite(), "Expected finite lastEstimate")
    }

    // ── Exponential distribution (one-sided bound) ──────────────────────

    @Test
    fun testExponentialDistribution() {
        // Exponential(1): cdf(x) = 1 - e^(-x), pdf(x) = e^(-x), quantile(p) = -ln(1-p)
        val expCdf = { x: Double -> if (x <= 0.0) 0.0 else 1.0 - exp(-x) }
        val expPdf = { x: Double -> if (x <= 0.0) 0.0 else exp(-x) }

        val pValues = doubleArrayOf(0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        for (p in pValues) {
            val expected = -ln(1.0 - p)
            val result = findQuantile(
                p = p,
                cdf = expCdf,
                pdf = expPdf,
                initialGuess = 1.0,
                lowerBound = 0.0,
            )
            assertEquals(expected, result, TOLERANCE, "Exponential quantile mismatch for p=$p")
        }
    }
}
