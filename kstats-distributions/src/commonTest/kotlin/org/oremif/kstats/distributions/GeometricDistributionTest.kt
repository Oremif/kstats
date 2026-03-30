package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class GeometricDistributionTest : DiscreteDistributionPropertyTests() {

    override fun createDistribution() = GeometricDistribution(0.3)
    override val testKRange = -1..15

    // --- Basic correctness (scipy 15-digit refs, using nbinom(1, p)) ---

    @Test
    fun testPmfKnownValues() {
        val d = GeometricDistribution(0.3)
        // scipy: stats.nbinom(1, 0.3).pmf(k)
        assertEquals(0.3, d.pmf(0), 1e-15)
        assertEquals(0.21, d.pmf(1), 1e-15)
        assertEquals(0.147, d.pmf(2), 1e-15)
        assertEquals(0.050421, d.pmf(5), 1e-10)
        assertEquals(0.0, d.pmf(-1), 1e-15)
    }

    @Test
    fun testCdfKnownValues() {
        val d = GeometricDistribution(0.3)
        assertEquals(0.3, d.cdf(0), 1e-15)
        assertEquals(0.657, d.cdf(2), 1e-12)
        assertEquals(0.882351, d.cdf(5), 1e-10)
        assertEquals(0.0, d.cdf(-1), 1e-15)
    }

    @Test
    fun testLogPmfKnownValues() {
        val d = GeometricDistribution(0.3)
        assertEquals(-1.20397280432594, d.logPmf(0), 1e-12)
        assertEquals(-1.9173226922034, d.logPmf(2), 1e-10)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(-1))
    }

    @Test
    fun testSfKnownValues() {
        val d = GeometricDistribution(0.3)
        // scipy: stats.nbinom(1, 0.3).sf(k)
        assertEquals(0.7, d.sf(0), 1e-15)
        assertEquals(0.343, d.sf(2), 1e-12)
        assertEquals(0.117649, d.sf(5), 1e-10)
        assertEquals(1.0, d.sf(-1), 1e-15)
    }

    // --- Quantile ---

    @Test
    fun testQuantileIntKnownValues() {
        val d = GeometricDistribution(0.3)
        assertEquals(0, d.quantileInt(0.25))
        assertEquals(1, d.quantileInt(0.5))
        assertEquals(3, d.quantileInt(0.75))
        assertEquals(0, d.quantileInt(0.0))
        assertEquals(Int.MAX_VALUE, d.quantileInt(1.0))
    }

    @Test
    fun testCdfSfAtIntMaxValue() {
        val d = GeometricDistribution(0.3)
        assertEquals(1.0, d.cdf(Int.MAX_VALUE), 1e-15)
        assertEquals(0.0, d.sf(Int.MAX_VALUE), 1e-15)
    }

    // --- Moments ---

    @Test
    fun testMoments() {
        val d = GeometricDistribution(0.3)
        assertEquals(2.33333333333333, d.mean, 1e-10)
        assertEquals(7.77777777777778, d.variance, 1e-10)
        // scipy: stats.nbinom(1, 0.3).stats(moments='sk')
        assertEquals(2.03188863586847, d.skewness, 1e-10)
        assertEquals(6.12857142857143, d.kurtosis, 1e-10)
    }

    @Test
    fun testMomentsP05() {
        val d = GeometricDistribution(0.5)
        assertEquals(2.12132034355964, d.skewness, 1e-10)
        assertEquals(6.5, d.kurtosis, 1e-10)
    }

    // --- Entropy ---

    @Test
    fun testEntropy() {
        // scipy: stats.nbinom(1, 0.3).entropy()
        assertEquals(2.03621434018294, GeometricDistribution(0.3).entropy, 1e-10)
        assertEquals(1.38629436111989, GeometricDistribution(0.5).entropy, 1e-10)
    }

    @Test
    fun testEntropyDegenerate() {
        assertEquals(0.0, GeometricDistribution(1.0).entropy, 1e-15)
    }

    // --- Edge cases ---

    @Test
    fun testP1Degenerate() {
        val d = GeometricDistribution(1.0)
        assertEquals(1.0, d.pmf(0), 1e-15)
        assertEquals(0.0, d.pmf(1), 1e-15)
        assertEquals(0.0, d.mean, 1e-15)
        assertEquals(0.0, d.variance, 1e-15)
        assertEquals(1.0, d.cdf(0), 1e-15)
        assertEquals(1.0, d.cdf(5), 1e-15)
        assertEquals(0.0, d.sf(0), 1e-15)
        assertEquals(0.0, d.sf(5), 1e-15)
        assertEquals(0.0, d.logPmf(0), 1e-15)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(1))
        assertEquals(0, d.quantileInt(0.5))
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { GeometricDistribution(0.0) }
        assertFailsWith<InvalidParameterException> { GeometricDistribution(-0.1) }
        assertFailsWith<InvalidParameterException> { GeometricDistribution(1.1) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = GeometricDistribution(0.5)
        assertFailsWith<InvalidParameterException> { d.quantileInt(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantileInt(1.1) }
    }

    @Test
    fun testExtremeParameters() {
        // p=1e-8: very long tail
        val d1 = GeometricDistribution(1e-8)
        assertEquals(1e-8, d1.pmf(0), 1e-20)
        assertEquals((1.0 - 1e-8) / 1e-8, d1.mean, 1.0)
        assertTrue(d1.mean.isFinite())
        assertTrue(d1.variance.isFinite())

        // p=0.999: very short tail
        val d2 = GeometricDistribution(0.999)
        assertEquals(0.999, d2.pmf(0), 1e-15)
        // pmf(10) = 0.999 * 0.001^10 ≈ 9.99e-31
        assertTrue(d2.pmf(10) > 0.0 && d2.pmf(10) < 1e-29)
    }

}
