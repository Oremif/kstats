package org.oremif.kstats.distributions

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DistributionTest {

    private val tol = 1e-15

    private class StubContinuousDistribution(
        override val mean: Double = 0.0,
        override val variance: Double = 1.0,
        override val skewness: Double = 0.0,
        override val kurtosis: Double = 0.0,
        private val cdfValue: Double = 0.5,
    ) : ContinuousDistribution {
        override fun pdf(x: Double): Double = 0.0
        override fun logPdf(x: Double): Double = Double.NEGATIVE_INFINITY
        override fun cdf(x: Double): Double = cdfValue
        override fun quantile(p: Double): Double = p
        override fun sample(random: Random): Double = random.nextDouble()
    }

    @Test
    fun standardDeviationDefaultSqrtVariance() {
        val d = StubContinuousDistribution(variance = 4.0)
        assertEquals(2.0, d.standardDeviation, tol)
    }

    @Test
    fun standardDeviationZeroVariance() {
        val d = StubContinuousDistribution(variance = 0.0)
        assertEquals(0.0, d.standardDeviation, tol)
    }

    @Test
    fun standardDeviationLargeVariance() {
        val d = StubContinuousDistribution(variance = 1e12)
        assertEquals(1e6, d.standardDeviation, tol)
    }

    @Test
    fun sfDefaultOneMinusCdf() {
        val d = StubContinuousDistribution(cdfValue = 0.3)
        assertEquals(0.7, d.sf(0.0), tol)
    }

    @Test
    fun sfBoundaryCdfZero() {
        val d = StubContinuousDistribution(cdfValue = 0.0)
        assertEquals(1.0, d.sf(0.0), tol)
    }

    @Test
    fun sfBoundaryCdfOne() {
        val d = StubContinuousDistribution(cdfValue = 1.0)
        assertEquals(0.0, d.sf(0.0), tol)
    }

    @Test
    fun entropyDefaultIsNaN() {
        val d = StubContinuousDistribution()
        assertEquals(Double.NaN, d.entropy)
    }

    // --- DiscreteDistribution defaults ---

    private class StubDiscreteDistribution(
        override val mean: Double = 0.0,
        override val variance: Double = 1.0,
        private val cdfValue: Double = 0.5,
    ) : DiscreteDistribution {
        override fun pmf(k: Int): Double = 0.0
        override fun logPmf(k: Int): Double = Double.NEGATIVE_INFINITY
        override fun cdf(k: Int): Double = cdfValue
        override fun quantileInt(p: Double): Int = 0
        override fun sample(random: Random): Int = 0
    }

    @Test
    fun discreteStandardDeviationDefault() {
        val d = StubDiscreteDistribution(variance = 9.0)
        assertEquals(3.0, d.standardDeviation, tol)
    }

    @Test
    fun discreteSfIntDefault() {
        val d = StubDiscreteDistribution(cdfValue = 0.3)
        assertEquals(0.7, d.sf(0), tol)
    }

    @Test
    fun discreteSfDoubleDefault() {
        val d = StubDiscreteDistribution(cdfValue = 0.3)
        assertEquals(0.7, d.sf(0.0), tol)
    }

    @Test
    fun discreteCdfDoubleBridgesToInt() {
        val d = StubDiscreteDistribution(cdfValue = 0.8)
        assertEquals(0.8, d.cdf(2.7), tol)
    }

    @Test
    fun discreteQuantileDoubleDelegatesToQuantileInt() {
        val d = StubDiscreteDistribution()
        assertEquals(0.0, d.quantile(0.5), tol)
    }

    @Test
    fun discreteSkewnessDefaultIsNaN() {
        val d = StubDiscreteDistribution()
        assertEquals(Double.NaN, d.skewness)
    }

    @Test
    fun discreteKurtosisDefaultIsNaN() {
        val d = StubDiscreteDistribution()
        assertEquals(Double.NaN, d.kurtosis)
    }

    @Test
    fun discreteEntropyDefaultIsNaN() {
        val d = StubDiscreteDistribution()
        assertEquals(Double.NaN, d.entropy)
    }

    // --- cdf(Double) floors correctly for negative fractional values ---

    @Test
    fun discreteCdfDoubleFloorsNegativeFraction() {
        // cdf(-0.5) should delegate to cdf(-1), not cdf(0)
        val poisson = PoissonDistribution(rate = 5.0)
        assertEquals(0.0, poisson.cdf(-0.5), tol, "cdf(-0.5) should be 0 for Poisson")
        assertEquals(0.0, poisson.cdf(-0.1), tol, "cdf(-0.1) should be 0 for Poisson")
    }

    @Test
    fun discreteCdfDoublePositiveFractionFloors() {
        val binom = BinomialDistribution(10, 0.3)
        // cdf(2.7) should equal cdf(2)
        assertEquals(binom.cdf(2), binom.cdf(2.7), tol)
        // cdf(3.0) should equal cdf(3)
        assertEquals(binom.cdf(3), binom.cdf(3.0), tol)
    }

    // --- logPmf edge cases for degenerate parameters ---

    @Test
    fun binomialLogPmfDegenerateP0() {
        val d = BinomialDistribution(10, 0.0)
        assertEquals(0.0, d.logPmf(0), tol, "logPmf(0) should be 0 when p=0")
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(1), "logPmf(1) should be -Inf when p=0")
    }

    @Test
    fun binomialLogPmfDegenerateP1() {
        val d = BinomialDistribution(10, 1.0)
        assertEquals(0.0, d.logPmf(10), tol, "logPmf(n) should be 0 when p=1")
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(0), "logPmf(0) should be -Inf when p=1")
    }

    @Test
    fun geometricLogPmfDegenerateP1() {
        val d = GeometricDistribution(1.0)
        assertEquals(0.0, d.logPmf(0), tol, "logPmf(0) should be 0 when p=1")
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(1), "logPmf(1) should be -Inf when p=1")
    }

    // --- sample() degenerate cases ---

    @Test
    fun binomialSampleDegenerateP0LargeN() {
        val d = BinomialDistribution(100, 0.0)
        assertEquals(0, d.sample(Random(42)), "sample should return 0 when p=0")
    }

    @Test
    fun binomialSampleDegenerateP1LargeN() {
        val d = BinomialDistribution(100, 1.0)
        assertEquals(100, d.sample(Random(42)), "sample should return n when p=1")
    }

    @Test
    fun geometricSampleDegenerateP1() {
        val d = GeometricDistribution(1.0)
        assertEquals(0, d.sample(Random(42)), "sample should return 0 when p=1")
    }

    // --- sample(n) validation ---

    @Test
    fun continuousSampleNegativeNThrows() {
        val d = NormalDistribution.STANDARD
        assertFailsWith<IllegalArgumentException> { d.sample(-1, Random(42)) }
    }

    @Test
    fun discreteSampleNegativeNThrows() {
        val d = PoissonDistribution(5.0)
        assertFailsWith<IllegalArgumentException> { d.sample(-1, Random(42)) }
    }

    @Test
    fun sampleZeroNReturnsEmpty() {
        val continuous = NormalDistribution.STANDARD
        assertTrue(continuous.sample(0, Random(42)).isEmpty())
        val discrete = PoissonDistribution(5.0)
        assertTrue(discrete.sample(0, Random(42)).isEmpty())
    }

    // --- Box-Muller / sampling robustness ---

    @Test
    fun normalSampleProducesFiniteValues() {
        val d = NormalDistribution.STANDARD
        val samples = d.sample(100_000, Random(42))
        assertTrue(samples.all { it.isFinite() }, "all samples should be finite")
    }

    @Test
    fun exponentialSampleProducesFiniteValues() {
        val d = ExponentialDistribution.STANDARD
        val samples = d.sample(100_000, Random(42))
        assertTrue(samples.all { it.isFinite() }, "all samples should be finite")
    }
}
