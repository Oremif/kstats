package org.oremif.kstats.distributions

import kotlin.math.abs
import kotlin.math.exp
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Base class providing property-based tests common to all [ContinuousDistribution] implementations.
 *
 * Subclasses supply a distribution instance and test points; the base class runs:
 * - `exp(logPdf(x)) ≈ pdf(x)` consistency
 * - `sf(x) + cdf(x) ≈ 1`
 * - `cdf(quantile(p)) ≈ p` roundtrip
 * - CDF monotonicity
 * - Sample mean/variance match theoretical moments
 * - Trapezoidal PDF integration ≈ CDF difference
 */
abstract class ContinuousDistributionPropertyTests {

    /** Distribution instance used for all property-based tests. */
    abstract fun createDistribution(): ContinuousDistribution

    /** Points in the distribution's support for consistency checks. */
    abstract val testPoints: List<Double>

    /** Probabilities for CDF-quantile roundtrip. */
    open val pValues: List<Double> = listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)

    /** Tolerance for exp(logPdf) ≈ pdf and sf + cdf ≈ 1 checks. */
    open val consistencyTol: Double = 1e-12

    /** Tolerance for CDF-quantile roundtrip. */
    open val roundTripTol: Double = 1e-10

    /** Epsilon for PDF integration quantile bounds (wider for heavy-tailed distributions). */
    open val integrationEpsilon: Double = 1e-6

    @Test
    fun logPdfConsistency() {
        val d = createDistribution()
        for (x in testPoints) {
            assertEquals(
                d.pdf(x), exp(d.logPdf(x)), consistencyTol,
                "exp(logPdf($x)) ≈ pdf($x)"
            )
        }
    }

    @Test
    fun sfPlusCdfEqualsOne() {
        val d = createDistribution()
        for (x in testPoints) {
            assertEquals(
                1.0, d.sf(x) + d.cdf(x), consistencyTol,
                "sf($x) + cdf($x) ≈ 1"
            )
        }
    }

    @Test
    fun cdfQuantileRoundTrip() {
        val d = createDistribution()
        for (p in pValues) {
            assertEquals(
                p, d.cdf(d.quantile(p)), roundTripTol,
                "cdf(quantile($p)) ≈ $p"
            )
        }
    }

    @Test
    fun cdfMonotonicity() {
        val d = createDistribution()
        var prev = 0.0
        for (x in testPoints.sorted()) {
            val cdfVal = d.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing at $x")
            prev = cdfVal
        }
    }

    @Test
    open fun sampleStatistics() {
        val d = createDistribution()
        if (d.mean.isNaN() || !d.mean.isFinite() || d.variance.isNaN() || !d.variance.isFinite()) return
        val samples = d.sample(100_000, Random(42))
        val sampleMean = samples.average()
        assertEquals(
            d.mean, sampleMean, maxOf(abs(d.mean) * 0.05, 0.15),
            "sample mean ≈ ${d.mean}"
        )
        val sampleVar = samples.sumOf { (it - sampleMean) * (it - sampleMean) } / (samples.size - 1)
        assertEquals(
            d.variance, sampleVar, maxOf(d.variance * 0.1, 0.15),
            "sample variance ≈ ${d.variance}"
        )
    }

    @Test
    open fun pdfIntegration() {
        val d = createDistribution()
        val lower = d.quantile(integrationEpsilon)
        val upper = d.quantile(1.0 - integrationEpsilon)
        val integral = trapezoidalIntegral({ d.pdf(it) }, lower, upper)
        assertEquals(d.cdf(upper) - d.cdf(lower), integral, 1e-4)
    }
}
