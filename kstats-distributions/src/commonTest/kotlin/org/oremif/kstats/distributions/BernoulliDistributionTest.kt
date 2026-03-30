package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BernoulliDistributionTest : DiscreteDistributionPropertyTests() {

    override fun createDistribution() = BernoulliDistribution(0.7)
    override val testKRange = -1..2
    override val consistencyTol = 1e-15
    override val sfCdfTol = 1e-15

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPmfKnownValues() {
        val d = BernoulliDistribution(0.7)
        // scipy: stats.bernoulli(0.7).pmf(k)
        assertEquals(0.3, d.pmf(0), 1e-15)
        assertEquals(0.7, d.pmf(1), 1e-15)
        assertEquals(0.0, d.pmf(2), 1e-15)
        assertEquals(0.0, d.pmf(-1), 1e-15)
    }

    @Test
    fun testCdfKnownValues() {
        val d = BernoulliDistribution(0.7)
        assertEquals(0.0, d.cdf(-1), 1e-15)
        assertEquals(0.3, d.cdf(0), 1e-15)
        assertEquals(1.0, d.cdf(1), 1e-15)
        assertEquals(1.0, d.cdf(5), 1e-15)
    }

    @Test
    fun testLogPmfKnownValues() {
        val d = BernoulliDistribution(0.7)
        // scipy: stats.bernoulli(0.7).logpmf(k)
        assertEquals(-1.20397280432594, d.logPmf(0), 1e-12)
        assertEquals(-0.356674943938732, d.logPmf(1), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(2))
    }

    @Test
    fun testSfKnownValues() {
        val d = BernoulliDistribution(0.7)
        assertEquals(0.7, d.sf(0), 1e-15)
        assertEquals(0.0, d.sf(1), 1e-15)
    }

    // --- Quantile ---

    @Test
    fun testQuantileIntKnownValues() {
        val d = BernoulliDistribution(0.7)
        assertEquals(0, d.quantileInt(0.25))
        assertEquals(1, d.quantileInt(0.5))
        assertEquals(1, d.quantileInt(0.75))
        assertEquals(0, d.quantileInt(0.0))
        assertEquals(1, d.quantileInt(1.0))
    }

    // --- Moments ---

    @Test
    fun testMoments() {
        val d = BernoulliDistribution(0.7)
        assertEquals(0.7, d.mean, 1e-15)
        assertEquals(0.21, d.variance, 1e-15)
        // scipy: stats.bernoulli(0.7).stats(moments='sk')
        assertEquals(-0.87287156094397, d.skewness, 1e-12)
        assertEquals(-1.23809523809524, d.kurtosis, 1e-12)
    }

    @Test
    fun testMomentsSymmetric() {
        val d = BernoulliDistribution(0.5)
        assertEquals(0.0, d.skewness, 1e-15)
        assertEquals(-2.0, d.kurtosis, 1e-12)
    }

    @Test
    fun testMomentsP03() {
        val d = BernoulliDistribution(0.3)
        assertEquals(0.872871560943969, d.skewness, 1e-12)
        assertEquals(-1.23809523809524, d.kurtosis, 1e-12)
    }

    // --- Entropy ---

    @Test
    fun testEntropy() {
        // scipy: stats.bernoulli(0.7).entropy()
        assertEquals(0.610864302054894, BernoulliDistribution(0.7).entropy, 1e-12)
        assertEquals(0.693147180559945, BernoulliDistribution(0.5).entropy, 1e-12)
        assertEquals(0.610864302054894, BernoulliDistribution(0.3).entropy, 1e-12)
    }

    @Test
    fun testEntropyDegenerate() {
        assertEquals(0.0, BernoulliDistribution(0.0).entropy, 1e-15)
        assertEquals(0.0, BernoulliDistribution(1.0).entropy, 1e-15)
    }

    // --- Edge cases ---

    @Test
    fun testDegenerateP0() {
        val d = BernoulliDistribution(0.0)
        assertEquals(1.0, d.pmf(0), 1e-15)
        assertEquals(0.0, d.pmf(1), 1e-15)
        assertEquals(0.0, d.mean, 1e-15)
        assertEquals(0.0, d.variance, 1e-15)
    }

    @Test
    fun testDegenerateP1() {
        val d = BernoulliDistribution(1.0)
        assertEquals(0.0, d.pmf(0), 1e-15)
        assertEquals(1.0, d.pmf(1), 1e-15)
        assertEquals(1.0, d.mean, 1e-15)
        assertEquals(0.0, d.variance, 1e-15)
    }

    @Test
    fun testSkewnessKurtosisDegenerate() {
        assertTrue(BernoulliDistribution(0.0).skewness.isNaN())
        assertTrue(BernoulliDistribution(1.0).skewness.isNaN())
        assertTrue(BernoulliDistribution(0.0).kurtosis.isNaN())
        assertTrue(BernoulliDistribution(1.0).kurtosis.isNaN())
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { BernoulliDistribution(-0.1) }
        assertFailsWith<InvalidParameterException> { BernoulliDistribution(1.1) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = BernoulliDistribution(0.5)
        assertFailsWith<InvalidParameterException> { d.quantileInt(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantileInt(1.1) }
    }

    @Test
    fun testExtremeParameters() {
        // p ≈ 0: near-zero probability
        val d1 = BernoulliDistribution(1e-15)
        assertEquals(1.0, d1.pmf(0), 1e-14)
        assertEquals(1e-15, d1.pmf(1), 1e-17) // FP precision: impl computes 1-(1-p) instead of p
        assertEquals(1e-15, d1.mean, 1e-17)
        assertEquals(1e-15, d1.sf(0), 1e-17)
        assertTrue(d1.logPmf(1).isFinite())
        // scipy: logpmf(1) = -34.5388
        assertEquals(-34.5387763949107, d1.logPmf(1), 1e-6)

        // p ≈ 1: near-one probability
        val d2 = BernoulliDistribution(1.0 - 1e-15)
        assertEquals(1.0 - 1e-15, d2.pmf(1), 1e-14)
        assertTrue(d2.logPmf(0).isFinite())
    }

}
