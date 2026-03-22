package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BetaDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val b = BetaDistribution(2.0, 5.0)
        // scipy: stats.beta(2, 5).pdf(x)
        assertEquals(0.0, b.pdf(0.0), 1e-12)
        assertEquals(1.9683, b.pdf(0.1), 1e-10)
        assertEquals(2.4576, b.pdf(0.2), 1e-10)
        assertEquals(2.1609, b.pdf(0.3), 1e-10)
        assertEquals(0.9375, b.pdf(0.5), 1e-10)
        assertEquals(0.0384, b.pdf(0.8), 1e-10)
        assertEquals(0.0, b.pdf(1.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val b = BetaDistribution(2.0, 5.0)
        // scipy: stats.beta(2, 5).cdf(x)
        assertEquals(0.0, b.cdf(0.0), 1e-12)
        assertEquals(0.114265, b.cdf(0.1), 1e-5)
        assertEquals(0.34464, b.cdf(0.2), 1e-5)
        assertEquals(0.579825, b.cdf(0.3), 1e-5)
        assertEquals(0.890625, b.cdf(0.5), 1e-5)
        assertEquals(0.9984, b.cdf(0.8), 1e-5)
        assertEquals(1.0, b.cdf(1.0), 1e-12)
    }

    @Test
    fun testLogPdfKnownValues() {
        val b = BetaDistribution(2.0, 5.0)
        assertEquals(0.677170226036805, b.logPdf(0.1), 1e-10)
        assertEquals(0.899185263971216, b.logPdf(0.2), 1e-10)
        assertEquals(0.77052480158129, b.logPdf(0.3), 1e-10)
        assertEquals(-0.0645385211375711, b.logPdf(0.5), 1e-10)
        assertEquals(-3.25969781938846, b.logPdf(0.8), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val b = BetaDistribution(2.0, 5.0)
        // scipy: stats.beta(2, 5).ppf(p)
        assertEquals(0.0267631911427551, b.quantile(0.01), 1e-6)
        assertEquals(0.0925952589131287, b.quantile(0.1), 1e-6)
        assertEquals(0.161162916790327, b.quantile(0.25), 1e-6)
        assertEquals(0.26444998329566, b.quantile(0.5), 1e-6)
        assertEquals(0.389479485200725, b.quantile(0.75), 1e-6)
        assertEquals(0.510316306551492, b.quantile(0.9), 1e-6)
        assertEquals(0.705686328319707, b.quantile(0.99), 1e-6)
    }

    @Test
    fun testMoments() {
        val b = BetaDistribution(2.0, 5.0)
        assertEquals(0.285714285714286, b.mean, 1e-12)
        assertEquals(0.0255102040816327, b.variance, 1e-12)
        assertEquals(0.596284793999944, b.skewness, 1e-10)
        assertEquals(-0.12, b.kurtosis, 1e-10)
    }

    // --- Edge cases ---

    @Test
    fun testOutsideSupport() {
        val b = BetaDistribution(2.0, 5.0)
        assertEquals(0.0, b.pdf(-0.1), 1e-12)
        assertEquals(0.0, b.pdf(1.1), 1e-12)
        assertEquals(0.0, b.cdf(-0.1), 1e-12)
        assertEquals(1.0, b.cdf(1.1), 1e-12)
        assertEquals(1.0, b.sf(-0.1), 1e-12)
        assertEquals(0.0, b.sf(1.1), 1e-12)
    }

    @Test
    fun testQuantileBoundaries() {
        val b = BetaDistribution(2.0, 5.0)
        assertEquals(0.0, b.quantile(0.0), 1e-12)
        assertEquals(1.0, b.quantile(1.0), 1e-12)
    }

    // --- Degenerate / special shapes ---

    @Test
    fun testUniformEquivalence() {
        // Beta(1,1) = Uniform(0,1)
        val b = BetaDistribution(1.0, 1.0)
        assertEquals(1.0, b.pdf(0.5), 1e-12)
        assertEquals(0.5, b.cdf(0.5), 1e-10)
        assertEquals(0.5, b.mean, 1e-12)
        assertEquals(1.0 / 12.0, b.variance, 1e-12)
    }

    @Test
    fun testUShapedBeta() {
        // Beta(0.5, 0.5) - arcsine distribution
        val b = BetaDistribution(0.5, 0.5)
        assertEquals(0.5, b.mean, 1e-12)
        assertEquals(0.125, b.variance, 1e-12)
        // scipy: stats.beta(0.5, 0.5).cdf(0.5)
        assertEquals(0.5, b.cdf(0.5), 1e-8)
        assertEquals(0.636619772367581, b.pdf(0.5), 1e-8)
    }

    @Test
    fun testJShapedBeta() {
        // Beta(5, 1) - power distribution
        val b = BetaDistribution(5.0, 1.0)
        assertEquals(0.833333333333333, b.mean, 1e-12)
        // scipy: stats.beta(5, 1).sf(0.9)
        assertEquals(0.40951, b.sf(0.9), 1e-5)
    }

    // --- NaN/inf handling ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { BetaDistribution(0.0, 1.0) }
        assertFailsWith<InvalidParameterException> { BetaDistribution(-1.0, 1.0) }
        assertFailsWith<InvalidParameterException> { BetaDistribution(1.0, 0.0) }
        assertFailsWith<InvalidParameterException> { BetaDistribution(1.0, -1.0) }
    }

    @Test
    fun testNaNInputs() {
        val b = BetaDistribution(2.0, 5.0)
        assertTrue(b.pdf(Double.NaN).isNaN(), "pdf(NaN) should be NaN")
        assertTrue(b.logPdf(Double.NaN).isNaN(), "logPdf(NaN) should be NaN")
        assertTrue(b.cdf(Double.NaN).isNaN(), "cdf(NaN) should be NaN")
        assertTrue(b.sf(Double.NaN).isNaN(), "sf(NaN) should be NaN")
    }

    @Test
    fun testInfinityInputs() {
        val b = BetaDistribution(2.0, 5.0)
        assertEquals(0.0, b.pdf(Double.POSITIVE_INFINITY), 1e-12)
        assertEquals(0.0, b.pdf(Double.NEGATIVE_INFINITY), 1e-12)
        assertEquals(1.0, b.cdf(Double.POSITIVE_INFINITY), 1e-12)
        assertEquals(0.0, b.cdf(Double.NEGATIVE_INFINITY), 1e-12)
        assertEquals(0.0, b.sf(Double.POSITIVE_INFINITY), 1e-12)
        assertEquals(1.0, b.sf(Double.NEGATIVE_INFINITY), 1e-12)
    }

    @Test
    fun testQuantileInvalidP() {
        val b = BetaDistribution(2.0, 5.0)
        assertFailsWith<InvalidParameterException> { b.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { b.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val b = BetaDistribution(2.0, 5.0)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            assertEquals(p, b.cdf(b.quantile(p)), 1e-8, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val b = BetaDistribution(2.0, 5.0)
        for (x in listOf(0.0, 0.1, 0.3, 0.5, 0.8, 1.0)) {
            assertEquals(1.0, b.sf(x) + b.cdf(x), 1e-10, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testSfKnownValues() {
        val b = BetaDistribution(2.0, 5.0)
        // scipy: stats.beta(2, 5).sf(x)
        assertEquals(0.885735, b.sf(0.1), 1e-5)
        assertEquals(0.420175, b.sf(0.3), 1e-5)
        assertEquals(0.109375, b.sf(0.5), 1e-5)
        assertEquals(0.0016, b.sf(0.8), 1e-5)
    }

    @Test
    fun testSfSymmetryRelation() {
        // sf(x) for Beta(a,b) should equal cdf(1-x) for Beta(b,a) (symmetry)
        val b1 = BetaDistribution(2.0, 5.0)
        val b2 = BetaDistribution(5.0, 2.0)
        for (x in listOf(0.1, 0.3, 0.5, 0.7, 0.9)) {
            assertEquals(b1.sf(x), b2.cdf(1.0 - x), 1e-10, "sf($x) = cdf_mirror(1-$x)")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val b = BetaDistribution(2.0, 5.0)
        for (x in listOf(0.1, 0.2, 0.3, 0.5, 0.8)) {
            assertEquals(b.pdf(x), exp(b.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleStats() {
        val b = BetaDistribution(2.0, 5.0) // mean=0.2857
        val rng = kotlin.random.Random(42)
        val samples = b.sample(100_000, rng)
        val sampleMean = samples.average()
        assertEquals(0.2857, sampleMean, 0.02, "sample mean ≈ 0.2857")
        val sampleVar = samples.sumOf { (it - sampleMean) * (it - sampleMean) } / (samples.size - 1)
        assertEquals(b.variance, sampleVar, maxOf(b.variance * 0.1, 0.05), "sample variance ≈ ${b.variance}")
    }

    @Test
    fun testCdfMonotonicity() {
        val b = BetaDistribution(2.0, 5.0)
        var prev = 0.0
        for (x in listOf(0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)) {
            val cdfVal = b.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testEntropy() {
        assertEquals(-0.484530714995488, BetaDistribution(2.0, 5.0).entropy, 1e-10)
        assertEquals(-0.241564475270491, BetaDistribution(0.5, 0.5).entropy, 1e-10)
        assertEquals(0.0, BetaDistribution(1.0, 1.0).entropy, 1e-10) // Uniform(0,1)
        assertEquals(-6.261125597836372, BetaDistribution(0.1, 0.1).entropy, 1e-10)
        assertEquals(-0.817636660417293, BetaDistribution(10.0, 3.0).entropy, 1e-10)
        assertEquals(-0.431945622001443, BetaDistribution(1.0, 3.0).entropy, 1e-10)
    }

    @Test
    fun testExtremeParameters() {
        // α=β=1000: peaked at 0.5
        val d1 = BetaDistribution(1000.0, 1000.0)
        // scipy: cdf(0.5) = 0.5
        assertEquals(0.5, d1.cdf(0.5), 1e-6)
        // scipy: cdf(0.52) = 0.963221
        assertEquals(0.963220516721358, d1.cdf(0.52), 1e-4)

        // α=β=0.01: bimodal U-shape
        val d2 = BetaDistribution(0.01, 0.01)
        // scipy: cdf(0.5) = 0.5
        assertEquals(0.5, d2.cdf(0.5), 1e-6)
        // scipy: pdf(0.5) ≈ 0.01973
        assertEquals(0.019727852239474, d2.pdf(0.5), 1e-6)

        // α=0.1, β=100: asymmetric, concentrated near 0
        val d3 = BetaDistribution(0.1, 100.0)
        // scipy: mean = 0.000999001
        assertEquals(0.000999001, d3.mean, 1e-6)
        // scipy: cdf(0.01) = 0.975893
        assertEquals(0.975892711294802, d3.cdf(0.01), 1e-4)
    }

    @Test
    fun testPdfIntegration() {
        val d = BetaDistribution(2.0, 5.0)
        val eps = 1e-6
        val lower = d.quantile(eps)
        val upper = d.quantile(1.0 - eps)
        val integral = trapezoidalIntegral({ d.pdf(it) }, lower, upper)
        assertEquals(d.cdf(upper) - d.cdf(lower), integral, 1e-4)
    }
}
