package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class NegativeBinomialDistributionTest : DiscreteDistributionPropertyTests() {

    override fun createDistribution() = NegativeBinomialDistribution(5, 0.4)
    override val testKRange = -1..20

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
        assertEquals(0, d.quantileInt(0.0))
        assertEquals(4, d.quantileInt(0.25))
        assertEquals(7, d.quantileInt(0.5))
        assertEquals(10, d.quantileInt(0.75))
        assertEquals(Int.MAX_VALUE, d.quantileInt(1.0))
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
        assertEquals(0.0, d.variance, 1e-15)
        assertEquals(0.0, d.entropy, 1e-15)
        assertEquals(1.0, d.cdf(0), 1e-15)
        assertEquals(1.0, d.cdf(5), 1e-15)
        assertEquals(0.0, d.sf(0), 1e-15)
        assertEquals(0.0, d.sf(5), 1e-15)
        assertEquals(0, d.quantileInt(0.5))
        assertEquals(Int.MAX_VALUE, d.quantileInt(1.0))
    }

    @Test
    fun testCdfSfAtIntMaxValue() {
        val d = NegativeBinomialDistribution(5, 0.4)
        assertEquals(1.0, d.cdf(Int.MAX_VALUE), 1e-15)
        assertEquals(0.0, d.sf(Int.MAX_VALUE), 1e-15)
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

    @Test
    fun testSmallProbability() {
        val d = NegativeBinomialDistribution(3, 0.05)
        // scipy: stats.nbinom(3, 0.05).mean() = 57, var = 1140
        assertEquals(57.0, d.mean, 1e-10)
        assertEquals(1140.0, d.variance, 1e-10)
        assertEquals(0.0, d.cdf(-1), 1e-15)
        // CDF should be very small at k=0 for small p: p^r = 0.05^3 = 0.000125
        assertEquals(1.25e-4, d.pmf(0), 1e-10)
        // Quantile round-trip
        val k50 = d.quantileInt(0.5)
        assertTrue(d.cdf(k50) >= 0.5)
        if (k50 > 0) assertTrue(d.cdf(k50 - 1) < 0.5)
    }

    @Test
    fun testLargeSuccesses() {
        val d = NegativeBinomialDistribution(100, 0.5)
        // mean = r*q/p = 100*0.5/0.5 = 100
        assertEquals(100.0, d.mean, 1e-10)
        assertEquals(200.0, d.variance, 1e-10)
        // Quantile round-trip
        val k50 = d.quantileInt(0.5)
        assertTrue(d.cdf(k50) >= 0.5)
        if (k50 > 0) assertTrue(d.cdf(k50 - 1) < 0.5)
    }

}
