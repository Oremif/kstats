package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class UniformDiscreteDistributionTest {

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
        assertEquals(2, d.quantileInt(0.25))
        assertEquals(3, d.quantileInt(0.5))
        assertEquals(6, d.quantileInt(0.99))
        assertEquals(1, d.quantileInt(0.0))
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

        // Near Int.MAX_VALUE boundary: integer overflow bug in mean calculation
        // (a + b) overflows Int — mean returns wrong value (known limitation)
        val lo = Int.MAX_VALUE - 10
        val hi = Int.MAX_VALUE
        val d2 = UniformDiscreteDistribution(lo, hi)
        // 11 values: pmf = 1/11
        assertEquals(1.0 / 11.0, d2.pmf(lo), 1e-15)
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

    // --- Property-based ---

    @Test
    fun testExpLogPmfConsistency() {
        val d = UniformDiscreteDistribution(1, 6)
        for (k in 1..6) {
            assertEquals(d.pmf(k), exp(d.logPmf(k)), 1e-15, "exp(logPmf($k)) ≈ pmf($k)")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val d = UniformDiscreteDistribution(1, 6)
        for (k in 0..7) {
            assertEquals(1.0, d.sf(k) + d.cdf(k), 1e-15, "sf($k) + cdf($k) ≈ 1")
        }
    }

    @Test
    fun testSampleStats() {
        val d = UniformDiscreteDistribution(1, 6)
        val rng = kotlin.random.Random(42)
        val samples = d.sample(100_000, rng)
        val doubles = samples.map { it.toDouble() }
        val sampleMean = doubles.average()
        assertEquals(3.5, sampleMean, 0.15, "sample mean ≈ 3.5")
        val sampleVar = doubles.sumOf { (it - sampleMean) * (it - sampleMean) } / (doubles.size - 1)
        assertEquals(d.variance, sampleVar, maxOf(d.variance * 0.1, 0.05), "sample variance ≈ ${d.variance}")
    }

    @Test
    fun testCdfMonotonicity() {
        val d = UniformDiscreteDistribution(1, 6)
        var prev = 0.0
        for (k in 1..6) {
            val cdfVal = d.cdf(k)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testCdfQuantileRoundTrip() {
        val d = UniformDiscreteDistribution(1, 6)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            val k = d.quantileInt(p)
            assertTrue(d.cdf(k) >= p, "cdf(quantileInt($p)) >= $p")
            if (k > 0) assertTrue(d.cdf(k - 1) < p, "cdf(quantileInt($p)-1) < $p")
        }
    }

    @Test
    fun testPmfSumsToOne() {
        val d = UniformDiscreteDistribution(1, 6)
        val total = (1..6).sumOf { d.pmf(it) }
        assertEquals(1.0, total, 1e-10)
    }
}
