package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class UniformDiscreteDistributionTest : DiscreteDistributionPropertyTests() {

    override fun createDistribution() = UniformDiscreteDistribution(1, 6)
    override val testKRange = 0..7

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPmfKnownValues() {
        val d = UniformDiscreteDistribution(1, 6) // dice
        // scipy: stats.randint(1, 7).pmf(k)
        assertEquals(0.166666666666667, d.pmf(1), 1e-12)
        assertEquals(0.166666666666667, d.pmf(3), 1e-12)
        assertEquals(0.166666666666667, d.pmf(6), 1e-12)
        assertEquals(0.0, d.pmf(0), 1e-15)
        assertEquals(0.0, d.pmf(7), 1e-15)
    }

    @Test
    fun testCdfKnownValues() {
        val d = UniformDiscreteDistribution(1, 6)
        assertEquals(0.166666666666667, d.cdf(1), 1e-12)
        assertEquals(0.5, d.cdf(3), 1e-12)
        assertEquals(1.0, d.cdf(6), 1e-15)
        assertEquals(0.0, d.cdf(0), 1e-15)
    }

    @Test
    fun testLogPmfKnownValues() {
        val d = UniformDiscreteDistribution(1, 6)
        assertEquals(-1.79175946922805, d.logPmf(3), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(0))
    }

    @Test
    fun testSfKnownValues() {
        val d = UniformDiscreteDistribution(1, 6)
        assertEquals(0.5, d.sf(3), 1e-12)
        assertEquals(0.0, d.sf(6), 1e-15)
    }

    // --- Quantile ---

    @Test
    fun testQuantileIntKnownValues() {
        val d = UniformDiscreteDistribution(1, 6)
        assertEquals(1, d.quantileInt(0.0))
        assertEquals(2, d.quantileInt(0.25))
        assertEquals(3, d.quantileInt(0.5))
        assertEquals(6, d.quantileInt(0.99))
        assertEquals(6, d.quantileInt(1.0))
    }

    @Test
    fun testQuantileDoubleInterface() {
        val d: Distribution = UniformDiscreteDistribution(1, 6)
        assertEquals(3.0, d.quantile(0.5), 1e-15)
        assertEquals(1.0, d.quantile(0.0), 1e-15)
        assertEquals(6.0, d.quantile(1.0), 1e-15)
    }

    // --- Moments ---

    @Test
    fun testMoments() {
        val d = UniformDiscreteDistribution(1, 6)
        assertEquals(3.5, d.mean, 1e-12)
        assertEquals(2.91666666666667, d.variance, 1e-10)
        assertEquals(0.0, d.skewness, 1e-15)
        // scipy: stats.randint(1, 7).stats(moments='k')
        assertEquals(-1.26857142857143, d.kurtosis, 1e-10)
    }

    @Test
    fun testMoments0To9() {
        val d = UniformDiscreteDistribution(0, 9)
        assertEquals(4.5, d.mean, 1e-12)
        assertEquals(8.25, d.variance, 1e-10)
        assertEquals(0.0, d.skewness, 1e-15)
        assertEquals(-1.22424242424242, d.kurtosis, 1e-10)
    }

    // --- Entropy ---

    @Test
    fun testEntropy() {
        // scipy: stats.randint(1, 7).entropy() = ln(6)
        assertEquals(1.79175946922805, UniformDiscreteDistribution(1, 6).entropy, 1e-12)
        // scipy: stats.randint(0, 10).entropy() = ln(10)
        assertEquals(2.30258509299405, UniformDiscreteDistribution(0, 9).entropy, 1e-12)
    }

    @Test
    fun testEntropyDegenerate() {
        assertEquals(0.0, UniformDiscreteDistribution(5, 5).entropy, 1e-15)
    }

    // --- Edge cases ---

    @Test
    fun testDegenerateSinglePoint() {
        val d = UniformDiscreteDistribution(5, 5)
        assertEquals(1.0, d.pmf(5), 1e-15)
        assertEquals(0.0, d.pmf(4), 1e-15)
        assertEquals(5.0, d.mean, 1e-15)
        assertEquals(0.0, d.variance, 1e-15)
    }

    @Test
    fun testKurtosisDegenerate() {
        assertTrue(UniformDiscreteDistribution(5, 5).kurtosis.isNaN())
    }

    @Test
    fun testExtremeParameters() {
        // Wide range: -1M to 1M (2000001 values)
        val d1 = UniformDiscreteDistribution(-1_000_000, 1_000_000)
        assertEquals(1.0 / 2_000_001.0, d1.pmf(0), 1e-15)
        assertEquals(0.0, d1.mean, 1e-10)
        // cdf(0) = 1000001/2000001
        assertEquals(1_000_001.0 / 2_000_001.0, d1.cdf(0), 1e-10)

        // Near Int.MAX_VALUE boundary — no integer overflow after Long-based n
        val lo = Int.MAX_VALUE - 10
        val hi = Int.MAX_VALUE
        val d2 = UniformDiscreteDistribution(lo, hi)
        // 11 values: pmf = 1/11
        assertEquals(1.0 / 11.0, d2.pmf(lo), 1e-15)
        assertEquals((lo / 2.0 + hi / 2.0), d2.mean, 1e-6)
        // sample should not throw
        d2.sample(kotlin.random.Random(0))
    }

    @Test
    fun testFullIntRange() {
        // Range spanning Int.MIN_VALUE to 0 — n > Int.MAX_VALUE (requires Long)
        val d = UniformDiscreteDistribution(Int.MIN_VALUE, 0)
        val expectedN = Int.MAX_VALUE.toLong() + 2 // 2147483649
        assertEquals(1.0 / expectedN.toDouble(), d.pmf(0), 1e-25)
        assertEquals(Int.MIN_VALUE / 2.0, d.mean, 1e-6)
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { UniformDiscreteDistribution(5, 4) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = UniformDiscreteDistribution(1, 6)
        assertFailsWith<InvalidParameterException> { d.quantileInt(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantileInt(1.1) }
    }

}
