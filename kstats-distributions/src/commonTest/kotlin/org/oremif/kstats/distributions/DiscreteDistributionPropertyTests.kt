package org.oremif.kstats.distributions

import kotlin.math.abs
import kotlin.math.exp
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Base class providing property-based tests common to all [DiscreteDistribution] implementations.
 *
 * Subclasses supply a distribution instance and a k-range; the base class runs:
 * - `exp(logPmf(k)) ≈ pmf(k)` consistency
 * - `sf(k) + cdf(k) ≈ 1`
 * - `cdf(quantileInt(p)) >= p` roundtrip
 * - CDF monotonicity
 * - Sample mean/variance match theoretical moments
 * - PMF sums to 1
 */
abstract class DiscreteDistributionPropertyTests {

    /** Distribution instance used for all property-based tests. */
    abstract fun createDistribution(): DiscreteDistribution

    /** Range of k values for consistency checks (should include outside-support values). */
    abstract val testKRange: IntRange

    /** Minimum support value for quantile roundtrip lower-bound guard. */
    open val supportMin: Int = 0

    /** Probabilities for CDF-quantile roundtrip. */
    open val pValues: List<Double> = listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)

    /** Tolerance for exp(logPmf) ≈ pmf consistency. */
    open val consistencyTol: Double = 1e-12

    /** Tolerance for sf + cdf ≈ 1 and PMF sums to 1. */
    open val sfCdfTol: Double = 1e-10

    @Test
    fun logPmfConsistency() {
        val d = createDistribution()
        for (k in testKRange) {
            assertEquals(
                d.pmf(k), exp(d.logPmf(k)), consistencyTol,
                "exp(logPmf($k)) ≈ pmf($k)"
            )
        }
    }

    @Test
    fun sfPlusCdfEqualsOne() {
        val d = createDistribution()
        for (k in testKRange) {
            assertEquals(
                1.0, d.sf(k) + d.cdf(k), sfCdfTol,
                "sf($k) + cdf($k) ≈ 1"
            )
        }
    }

    @Test
    fun cdfQuantileRoundTrip() {
        val d = createDistribution()
        for (p in pValues) {
            val k = d.quantileInt(p)
            assertTrue(d.cdf(k) >= p, "cdf(quantileInt($p)) >= $p")
            if (k > supportMin) assertTrue(
                d.cdf(k - 1) < p,
                "cdf(quantileInt($p)-1) < $p"
            )
        }
    }

    @Test
    fun cdfMonotonicity() {
        val d = createDistribution()
        var prev = 0.0
        for (k in testKRange) {
            val cdfVal = d.cdf(k)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing at $k")
            prev = cdfVal
        }
    }

    @Test
    open fun sampleStatistics() {
        val d = createDistribution()
        if (d.mean.isNaN() || !d.mean.isFinite() || d.variance.isNaN() || !d.variance.isFinite()) return
        val samples = d.sample(100_000, Random(42))
        val doubles = samples.map { it.toDouble() }
        val sampleMean = doubles.average()
        assertEquals(
            d.mean, sampleMean, maxOf(abs(d.mean) * 0.05, 0.15),
            "sample mean ≈ ${d.mean}"
        )
        val sampleVar = doubles.sumOf { (it - sampleMean) * (it - sampleMean) } / (doubles.size - 1)
        assertEquals(
            d.variance, sampleVar, maxOf(d.variance * 0.1, 0.15),
            "sample variance ≈ ${d.variance}"
        )
    }

    @Test
    fun pmfSumsToOne() {
        val d = createDistribution()
        val lower = d.quantileInt(0.0)
        val upper = d.quantileInt(1.0).let { q ->
            if (q == Int.MAX_VALUE) d.quantileInt(1.0 - 1e-10) else q
        }
        val total = (lower..upper).sumOf { d.pmf(it) }
        assertEquals(1.0, total, sfCdfTol)
    }
}
