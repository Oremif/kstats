package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PoissonDistributionTest : DiscreteDistributionPropertyTests() {

    override fun createDistribution() = PoissonDistribution(3.0)
    override val testKRange = -1..15

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPmfKnownValues() {
        val d = PoissonDistribution(3.0)
        // scipy: stats.poisson(3).pmf(k)
        assertEquals(0.0497870683678639, d.pmf(0), 1e-12)
        assertEquals(0.224041807655388, d.pmf(2), 1e-12)
        assertEquals(0.224041807655388, d.pmf(3), 1e-12)
        assertEquals(0.100818813444925, d.pmf(5), 1e-12)
        assertEquals(0.0, d.pmf(-1), 1e-15)
    }

    @Test
    fun testCdfKnownValues() {
        val d = PoissonDistribution(3.0)
        assertEquals(0.049787068367864, d.cdf(0), 1e-10)
        assertEquals(0.423190081126843, d.cdf(2), 1e-10)
        assertEquals(0.916082057968696, d.cdf(5), 1e-10)
        assertEquals(0.0, d.cdf(-1), 1e-15)
    }

    @Test
    fun testLogPmfKnownValues() {
        val d = PoissonDistribution(3.0)
        assertEquals(-1.49592260322373, d.logPmf(2), 1e-10)
        assertEquals(-2.2944302994415, d.logPmf(5), 1e-10)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(-1))
    }

    @Test
    fun testSfKnownValues() {
        val d = PoissonDistribution(3.0)
        // scipy: stats.poisson(3).sf(k)
        assertEquals(0.576809918873156, d.sf(2), 1e-10)
        assertEquals(0.0839179420313035, d.sf(5), 1e-10)
        assertEquals(1.0, d.sf(-1), 1e-15)
    }

    @Test
    fun testSfLambda10() {
        val d = PoissonDistribution(10.0)
        assertEquals(0.416960249807015, d.sf(10), 1e-10)
    }

    // --- Quantile ---

    @Test
    fun testQuantileIntKnownValues() {
        val d = PoissonDistribution(3.0)
        assertEquals(2, d.quantileInt(0.25))
        assertEquals(3, d.quantileInt(0.5))
        assertEquals(4, d.quantileInt(0.75))
        assertEquals(8, d.quantileInt(0.99))
        assertEquals(0, d.quantileInt(0.0))
    }

    // --- Moments ---

    @Test
    fun testMoments() {
        val d = PoissonDistribution(3.0)
        assertEquals(3.0, d.mean, 1e-15)
        assertEquals(3.0, d.variance, 1e-15)
        // scipy: stats.poisson(3).stats(moments='sk')
        assertEquals(0.577350269189626, d.skewness, 1e-12)
        assertEquals(0.333333333333333, d.kurtosis, 1e-12)
    }

    @Test
    fun testMomentsLambda10() {
        val d = PoissonDistribution(10.0)
        assertEquals(0.316227766016838, d.skewness, 1e-12)
        assertEquals(0.1, d.kurtosis, 1e-12)
    }

    // --- Entropy ---

    @Test
    fun testEntropy() {
        // scipy: stats.poisson(3).entropy()
        assertEquals(1.93147019814857, PoissonDistribution(3.0).entropy, 1e-8)
        // scipy: stats.poisson(10).entropy()
        assertEquals(2.56140993527491, PoissonDistribution(10.0).entropy, 1e-8)
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { PoissonDistribution(0.0) }
        assertFailsWith<InvalidParameterException> { PoissonDistribution(-1.0) }
        assertFailsWith<InvalidParameterException> { PoissonDistribution(Double.NaN) }
        assertFailsWith<InvalidParameterException> { PoissonDistribution(Double.POSITIVE_INFINITY) }
        assertFailsWith<InvalidParameterException> { PoissonDistribution(Double.NEGATIVE_INFINITY) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = PoissonDistribution(3.0)
        assertFailsWith<InvalidParameterException> { d.quantileInt(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantileInt(1.1) }
    }

    @Test
    fun testExtremeParameters() {
        // λ=500: previously triggered ConvergenceException, now handled by dynamic iteration limit
        val d500 = PoissonDistribution(500.0)
        val cdf500 = d500.cdf(500)
        assertTrue(cdf500 > 0.4 && cdf500 < 0.6, "Poisson(500).cdf(500) should be near 0.5, got $cdf500")

        // λ=1000: large parameter
        val d1000 = PoissonDistribution(1000.0)
        val cdf1000 = d1000.cdf(1000)
        assertTrue(cdf1000 > 0.4 && cdf1000 < 0.6, "Poisson(1000).cdf(1000) should be near 0.5, got $cdf1000")

        // λ=100: moderately large parameter
        val d1 = PoissonDistribution(100.0)
        assertTrue(d1.cdf(100) > 0.4 && d1.cdf(100) < 0.6)
        assertTrue(d1.sf(100) > 0.4 && d1.sf(100) < 0.6)

        // λ=1e-10: tiny
        val d2 = PoissonDistribution(1e-10)
        // scipy: pmf(0) ≈ 1.0
        assertEquals(1.0, d2.pmf(0), 1e-9)
        // scipy: sf(0) ≈ 1e-10
        assertEquals(1e-10, d2.sf(0), 1e-15)
    }

}
