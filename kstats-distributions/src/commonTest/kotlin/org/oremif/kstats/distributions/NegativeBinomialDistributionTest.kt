package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class NegativeBinomialDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPmfKnownValues() {
        val d = NegativeBinomialDistribution(5, 0.4)
        // scipy: stats.nbinom(5, 0.4).pmf(k)
        assertEquals(0.01024, d.pmf(0), 1e-10)
        assertEquals(0.0774144, d.pmf(3), 1e-10)
        assertEquals(0.1003290624, d.pmf(5), 1e-10)
        assertEquals(0.061979281588224, d.pmf(10), 1e-10)
        assertEquals(0.0, d.pmf(-1), 1e-15)
    }

    @Test
    fun testCdfKnownValues() {
        val d = NegativeBinomialDistribution(5, 0.4)
        assertEquals(0.01024, d.cdf(0), 1e-10)
        assertEquals(0.1736704, d.cdf(3), 1e-10)
        assertEquals(0.3668967424, d.cdf(5), 1e-10)
        assertEquals(0.782722294349824, d.cdf(10), 1e-10)
        assertEquals(0.0, d.cdf(-1), 1e-15)
    }

    @Test
    fun testCdfLargeK() {
        // Large k values that would be slow/inaccurate with naive PMF summation
        val d = NegativeBinomialDistribution(5, 0.4)
        assertEquals(9.999999523331493e-01, d.cdf(50), 1e-10)

        val d2 = NegativeBinomialDistribution(10, 0.3)
        assertEquals(9.941288117574135e-01, d2.cdf(50), 1e-10)
        assertEquals(9.999999741352392e-01, d2.cdf(100), 1e-10)
    }

    @Test
    fun testLogPmfKnownValues() {
        val d = NegativeBinomialDistribution(5, 0.4)
        assertEquals(-2.55858246917933, d.logPmf(3), 1e-10)
        assertEquals(-2.29929987124925, d.logPmf(5), 1e-10)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(-1))
    }

    @Test
    fun testSfKnownValues() {
        val d = NegativeBinomialDistribution(5, 0.4)
        assertEquals(0.6331032576, d.sf(5), 1e-10)
        assertEquals(0.217277705650176, d.sf(10), 1e-10)
    }

    @Test
    fun testSfLargeK() {
        // Large k values — sf uses complementary regularizedBeta for precision
        val d = NegativeBinomialDistribution(5, 0.4)
        assertEquals(4.766685066418214e-08, d.sf(50), 1e-15)
        assertEquals(5.091810499366119e-18, d.sf(100), 1e-25)

        val d2 = NegativeBinomialDistribution(10, 0.3)
        assertEquals(5.871188242586509e-03, d2.sf(50), 1e-10)
        assertEquals(2.586476076790538e-08, d2.sf(100), 1e-15)
    }

    // --- Quantile ---

    @Test
    fun testQuantileIntKnownValues() {
        val d = NegativeBinomialDistribution(5, 0.4)
        assertEquals(4, d.quantileInt(0.25))
        assertEquals(7, d.quantileInt(0.5))
        assertEquals(10, d.quantileInt(0.75))
    }

    // --- Moments ---

    @Test
    fun testMoments() {
        val d = NegativeBinomialDistribution(5, 0.4)
        assertEquals(7.5, d.mean, 1e-10)
        assertEquals(18.75, d.variance, 1e-10)
        // scipy: stats.nbinom(5, 0.4).stats(moments='sk')
        assertEquals(0.923760430703401, d.skewness, 1e-10)
        assertEquals(1.25333333333333, d.kurtosis, 1e-10)
    }

    @Test
    fun testMomentsR1P05() {
        val d = NegativeBinomialDistribution(1, 0.5)
        // Same as Geometric(0.5)
        assertEquals(2.12132034355964, d.skewness, 1e-10)
        assertEquals(6.5, d.kurtosis, 1e-10)
    }

    // --- Entropy ---

    @Test
    fun testEntropy() {
        // scipy: stats.nbinom(5, 0.4).entropy()
        assertEquals(2.80603593100731, NegativeBinomialDistribution(5, 0.4).entropy, 1e-6)
        // scipy: stats.nbinom(1, 0.5).entropy()
        assertEquals(1.38629436111989, NegativeBinomialDistribution(1, 0.5).entropy, 1e-8)
    }

    // --- Edge cases ---

    @Test
    fun testPmfAtZero() {
        val d = NegativeBinomialDistribution(3, 0.5)
        // pmf(0) = p^r = 0.5^3 = 0.125
        assertEquals(0.125, d.pmf(0), 1e-15)
    }

    @Test
    fun testP1Degenerate() {
        val d = NegativeBinomialDistribution(5, 1.0)
        assertEquals(1.0, d.pmf(0), 1e-15)
        assertEquals(0.0, d.pmf(1), 1e-15)
        assertEquals(0.0, d.mean, 1e-15)
        assertEquals(1.0, d.cdf(0), 1e-15)
        assertEquals(1.0, d.cdf(5), 1e-15)
        assertEquals(0.0, d.sf(0), 1e-15)
        assertEquals(0.0, d.sf(5), 1e-15)
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { NegativeBinomialDistribution(0, 0.5) }
        assertFailsWith<InvalidParameterException> { NegativeBinomialDistribution(-1, 0.5) }
        assertFailsWith<InvalidParameterException> { NegativeBinomialDistribution(5, 0.0) }
        assertFailsWith<InvalidParameterException> { NegativeBinomialDistribution(5, -0.1) }
        assertFailsWith<InvalidParameterException> { NegativeBinomialDistribution(5, 1.1) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = NegativeBinomialDistribution(5, 0.4)
        assertFailsWith<InvalidParameterException> { d.quantileInt(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantileInt(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testExpLogPmfConsistency() {
        val d = NegativeBinomialDistribution(5, 0.4)
        for (k in 0..15) {
            assertEquals(d.pmf(k), exp(d.logPmf(k)), 1e-12, "exp(logPmf($k)) ≈ pmf($k)")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val d = NegativeBinomialDistribution(5, 0.4)
        for (k in -1..20) {
            assertEquals(1.0, d.sf(k) + d.cdf(k), 1e-12, "sf($k) + cdf($k) ≈ 1")
        }
    }

    @Test
    fun testSampleStats() {
        val d = NegativeBinomialDistribution(5, 0.4)
        val rng = kotlin.random.Random(42)
        val samples = d.sample(100_000, rng)
        val doubles = samples.map { it.toDouble() }
        val sampleMean = doubles.average()
        assertEquals(7.5, sampleMean, 0.4, "sample mean ≈ 7.5")
        val sampleVar = doubles.sumOf { (it - sampleMean) * (it - sampleMean) } / (doubles.size - 1)
        assertEquals(d.variance, sampleVar, maxOf(d.variance * 0.1, 0.05), "sample variance ≈ ${d.variance}")
    }

    @Test
    fun testCdfMonotonicity() {
        val d = NegativeBinomialDistribution(5, 0.4)
        var prev = 0.0
        for (k in 0..30) {
            val cdfVal = d.cdf(k)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testCdfQuantileRoundTrip() {
        val d = NegativeBinomialDistribution(5, 0.4)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            val k = d.quantileInt(p)
            assertTrue(d.cdf(k) >= p, "cdf(quantileInt($p)) >= $p")
            if (k > 0) assertTrue(d.cdf(k - 1) < p, "cdf(quantileInt($p)-1) < $p")
        }
    }

    @Test
    fun testPmfSumsToOne() {
        val d = NegativeBinomialDistribution(5, 0.4)
        val upper = d.quantileInt(1.0 - 1e-10)
        val total = (0..upper).sumOf { d.pmf(it) }
        assertEquals(1.0, total, 1e-10)
    }
}
