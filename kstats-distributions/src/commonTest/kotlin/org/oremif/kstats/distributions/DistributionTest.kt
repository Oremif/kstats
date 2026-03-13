package org.oremif.kstats.distributions

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

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
}
