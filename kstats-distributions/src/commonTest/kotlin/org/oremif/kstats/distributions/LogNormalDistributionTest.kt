package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LogNormalDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val d = LogNormalDistribution(0.0, 1.0)
        // scipy: stats.lognorm(s=1, scale=exp(0)).pdf(x)
        assertEquals(0.627496077115924, d.pdf(0.5), 1e-12)
        assertEquals(0.398942280401433, d.pdf(1.0), 1e-12)
        assertEquals(0.156874019278981, d.pdf(2.0), 1e-12)
        assertEquals(0.0218507148303272, d.pdf(5.0), 1e-12)
        assertEquals(0.00281590189015268, d.pdf(10.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val d = LogNormalDistribution(0.0, 1.0)
        // scipy: stats.lognorm(s=1, scale=exp(0)).cdf(x)
        assertEquals(0.244108595785583, d.cdf(0.5), 1e-10)
        assertEquals(0.5, d.cdf(1.0), 1e-10)
        assertEquals(0.755891404214417, d.cdf(2.0), 1e-10)
        assertEquals(0.946239689548337, d.cdf(5.0), 1e-10)
        assertEquals(0.9893489006583, d.cdf(10.0), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val d = LogNormalDistribution(0.0, 1.0)
        // scipy: stats.lognorm(s=1, scale=exp(0)).sf(x)
        assertEquals(0.755891404214417, d.sf(0.5), 1e-10)
        assertEquals(0.5, d.sf(1.0), 1e-10)
        assertEquals(0.244108595785583, d.sf(2.0), 1e-10)
        assertEquals(0.0537603104516631, d.sf(5.0), 1e-10)
        assertEquals(0.0106510993417001, d.sf(10.0), 1e-10)
    }

    @Test
    fun testLogPdfKnownValues() {
        val d = LogNormalDistribution(0.0, 1.0)
        assertEquals(-0.466017859603828, d.logPdf(0.5), 1e-10)
        assertEquals(-0.918938533204673, d.logPdf(1.0), 1e-10)
        assertEquals(-1.85231222072372, d.logPdf(2.0), 1e-10)
        assertEquals(-3.82352164262889, d.logPdf(5.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val d = LogNormalDistribution(0.0, 1.0)
        // scipy: stats.lognorm(s=1, scale=exp(0)).ppf(p)
        assertEquals(0.097651733070336, d.quantile(0.01), 1e-8)
        assertEquals(0.27760624185201, d.quantile(0.1), 1e-8)
        assertEquals(0.509416283863278, d.quantile(0.25), 1e-8)
        assertEquals(1.0, d.quantile(0.5), 1e-10)
        assertEquals(1.96303108415826, d.quantile(0.75), 1e-8)
        assertEquals(3.60222447927916, d.quantile(0.9), 1e-8)
        assertEquals(10.2404736563121, d.quantile(0.99), 1e-6)
    }

    @Test
    fun testMoments() {
        val d = LogNormalDistribution(0.0, 1.0)
        assertEquals(1.64872127070013, d.mean, 1e-10)
        assertEquals(4.6707742704716, d.variance, 1e-8)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.lognorm(s=1, scale=exp(0)).entropy()
        assertEquals(1.41893853320467, LogNormalDistribution(0.0, 1.0).entropy, 1e-10)
        // scipy: stats.lognorm(s=0.5, scale=exp(2)).entropy()
        assertEquals(2.72579135264473, LogNormalDistribution(2.0, 0.5).entropy, 1e-10)
    }

    // --- Edge cases ---

    @Test
    fun testNonPositiveX() {
        val d = LogNormalDistribution(0.0, 1.0)
        assertEquals(0.0, d.pdf(0.0), 1e-12)
        assertEquals(0.0, d.pdf(-1.0), 1e-12)
        assertEquals(0.0, d.cdf(0.0), 1e-12)
        assertEquals(0.0, d.cdf(-1.0), 1e-12)
        assertEquals(1.0, d.sf(0.0), 1e-12)
        assertEquals(1.0, d.sf(-1.0), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPdf(0.0))
        assertEquals(Double.NEGATIVE_INFINITY, d.logPdf(-1.0))
    }

    @Test
    fun testQuantileBoundaries() {
        val d = LogNormalDistribution(0.0, 1.0)
        assertEquals(0.0, d.quantile(0.0), 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, d.quantile(1.0))
    }

    // --- Different param combos ---

    @Test
    fun testMu2Sigma05() {
        val d = LogNormalDistribution(2.0, 0.5)
        // scipy: stats.lognorm(s=0.5, scale=exp(2))
        assertEquals(8.37289748812726, d.mean, 1e-8)
        assertEquals(19.9117189538339, d.variance, 1e-6)
        // cdf
        assertEquals(0.217364732271511, d.cdf(5.0), 1e-8)
        assertEquals(0.727467038315737, d.cdf(10.0), 1e-8)
        // sf
        assertEquals(0.782635267728489, d.sf(5.0), 1e-8)
        assertEquals(0.272532961684263, d.sf(10.0), 1e-8)
        // ppf
        assertEquals(2.30902662894362, d.quantile(0.01), 1e-6)
        assertEquals(7.38905609893065, d.quantile(0.5), 1e-6)
        assertEquals(23.6455263654204, d.quantile(0.99), 1e-5)
    }

    @Test
    fun testSfUpperTail() {
        val d = LogNormalDistribution(0.0, 1.0)
        // scipy: stats.lognorm(s=1, scale=exp(0)).sf(x)
        assertEquals(0.00136893348785809, d.sf(20.0), 1e-8)
        assertEquals(4.57630952498886e-05, d.sf(50.0), 1e-9)
        assertEquals(2.06064339597171e-06, d.sf(100.0), 1e-10)
    }

    // --- NaN/inf handling ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { LogNormalDistribution(0.0, 0.0) }
        assertFailsWith<InvalidParameterException> { LogNormalDistribution(0.0, -1.0) }
        assertFailsWith<InvalidParameterException> { LogNormalDistribution(0.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { LogNormalDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { LogNormalDistribution(Double.POSITIVE_INFINITY, 1.0) }
        assertFailsWith<InvalidParameterException> { LogNormalDistribution(Double.NEGATIVE_INFINITY, 1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = LogNormalDistribution(0.0, 1.0)
        assertFailsWith<InvalidParameterException> { d.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantile(1.1) }
    }

    @Test
    fun testNaNAndInfinityInputs() {
        val d = LogNormalDistribution(0.0, 1.0)
        assertTrue(d.pdf(Double.NaN).isNaN(), "pdf(NaN) should be NaN")
        assertTrue(d.cdf(Double.NaN).isNaN(), "cdf(NaN) should be NaN")
        assertTrue(d.sf(Double.NaN).isNaN(), "sf(NaN) should be NaN")
        assertTrue(d.logPdf(Double.NaN).isNaN(), "logPdf(NaN) should be NaN")

        assertEquals(0.0, d.pdf(Double.POSITIVE_INFINITY), 1e-12)
        assertEquals(1.0, d.cdf(Double.POSITIVE_INFINITY), 1e-12)
        assertEquals(0.0, d.sf(Double.POSITIVE_INFINITY), 1e-12)
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val d = LogNormalDistribution(0.0, 1.0)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            assertEquals(p, d.cdf(d.quantile(p)), 1e-10, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val d = LogNormalDistribution(0.0, 1.0)
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 5.0, 10.0)) {
            assertEquals(1.0, d.sf(x) + d.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val d = LogNormalDistribution(0.0, 1.0)
        for (x in listOf(0.5, 1.0, 2.0, 5.0, 10.0)) {
            assertEquals(d.pdf(x), exp(d.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleStats() {
        val d = LogNormalDistribution(0.0, 1.0) // mean ≈ 1.649
        val rng = kotlin.random.Random(42)
        val samples = d.sample(100_000, rng)
        val sampleMean = samples.average()
        assertEquals(1.649, sampleMean, 0.15, "sample mean ≈ 1.649")
        val sampleVar = samples.sumOf { (it - sampleMean) * (it - sampleMean) } / (samples.size - 1)
        assertEquals(d.variance, sampleVar, maxOf(d.variance * 0.1, 0.05), "sample variance ≈ ${d.variance}")
    }

    @Test
    fun testCdfMonotonicity() {
        val d = LogNormalDistribution(0.0, 1.0)
        var prev = 0.0
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)) {
            val cdfVal = d.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testCdfSymmetryInLogSpace() {
        // LogNormal(mu, sigma): cdf(exp(mu)) = 0.5
        val d = LogNormalDistribution(2.0, 0.5)
        assertEquals(0.5, d.cdf(exp(2.0)), 1e-10)
    }

    @Test
    fun testExtremeParameters() {
        // σ=10: heavy tail
        val d1 = LogNormalDistribution(0.0, 10.0)
        // scipy: cdf(1) = 0.5 (median = exp(μ) = 1)
        assertEquals(0.5, d1.cdf(1.0), 1e-10)

        // σ=0.01: concentrated around exp(μ)=1
        val d2 = LogNormalDistribution(0.0, 0.01)
        // scipy: pdf(1) ≈ 39.894228
        assertEquals(39.894228040143268, d2.pdf(1.0), 1e-6)

        // μ=50: huge median at exp(50)
        val d3 = LogNormalDistribution(50.0, 1.0)
        // scipy: cdf(exp(50)) = 0.5
        assertEquals(0.5, d3.cdf(exp(50.0)), 1e-10)
    }

    @Test
    fun testPdfIntegration() {
        val d = LogNormalDistribution(0.0, 1.0)
        val eps = 1e-6
        val lower = d.quantile(eps)
        val upper = d.quantile(1.0 - eps)
        val integral = trapezoidalIntegral({ d.pdf(it) }, lower, upper)
        assertEquals(d.cdf(upper) - d.cdf(lower), integral, 1e-4)
    }
}
