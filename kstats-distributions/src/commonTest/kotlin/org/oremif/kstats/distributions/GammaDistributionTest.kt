package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class GammaDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val g = GammaDistribution(2.0, 1.0)
        // scipy: stats.gamma(2, scale=1).pdf(x)
        assertEquals(0.303265329856317, g.pdf(0.5), 1e-12)
        assertEquals(0.367879441171442, g.pdf(1.0), 1e-12)
        assertEquals(0.270670566473225, g.pdf(2.0), 1e-12)
        assertEquals(0.149361205103592, g.pdf(3.0), 1e-12)
        assertEquals(0.0336897349954273, g.pdf(5.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val g = GammaDistribution(2.0, 1.0)
        // scipy: stats.gamma(2, scale=1).cdf(x)
        assertEquals(0.0902040104310499, g.cdf(0.5), 1e-10)
        assertEquals(0.264241117657115, g.cdf(1.0), 1e-10)
        assertEquals(0.593994150290162, g.cdf(2.0), 1e-10)
        assertEquals(0.800851726528544, g.cdf(3.0), 1e-10)
        assertEquals(0.959572318005487, g.cdf(5.0), 1e-10)
    }

    @Test
    fun testLogPdfKnownValues() {
        val g = GammaDistribution(2.0, 1.0)
        assertEquals(-1.19314718055995, g.logPdf(0.5), 1e-10)
        assertEquals(-1.0, g.logPdf(1.0), 1e-10)
        assertEquals(-1.30685281944005, g.logPdf(2.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val g = GammaDistribution(2.0, 1.0)
        // scipy: stats.gamma(2, scale=1).ppf(p)
        // Note: extreme p values (0.01, 0.99) may not converge well — see DIST-010
        assertEquals(0.531811608389612, g.quantile(0.1), 1e-6)
        assertEquals(0.961278763114777, g.quantile(0.25), 1e-6)
        assertEquals(1.67834699001666, g.quantile(0.5), 1e-6)
        assertEquals(2.6926345288897, g.quantile(0.75), 1e-6)
        assertEquals(3.88972016986743, g.quantile(0.9), 1e-6)
    }

    @Test
    fun testMoments() {
        val g = GammaDistribution(2.0, 1.0)
        assertEquals(2.0, g.mean, 1e-12)
        assertEquals(2.0, g.variance, 1e-12)
        assertEquals(2.0 / sqrt(2.0), g.skewness, 1e-12)
        assertEquals(3.0, g.kurtosis, 1e-12)
    }

    @Test
    fun testMomentsWithRate() {
        val g = GammaDistribution(3.0, 2.0)
        assertEquals(1.5, g.mean, 1e-12)
        assertEquals(0.75, g.variance, 1e-12)
    }

    // --- Edge cases ---

    @Test
    fun testPdfAtZero() {
        // shape=2: pdf(0) = 0
        assertEquals(0.0, GammaDistribution(2.0, 1.0).pdf(0.0), 1e-12)
        // shape=1: pdf(0) = rate
        assertEquals(2.0, GammaDistribution(1.0, 2.0).pdf(0.0), 1e-12)
        // shape<1: pdf(0) = +Inf
        assertEquals(Double.POSITIVE_INFINITY, GammaDistribution(0.5, 1.0).pdf(0.0))
    }

    @Test
    fun testNegativeX() {
        val g = GammaDistribution(2.0, 1.0)
        assertEquals(0.0, g.pdf(-1.0), 1e-12)
        assertEquals(0.0, g.cdf(-1.0), 1e-12)
        assertEquals(1.0, g.sf(-1.0), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, g.logPdf(-1.0))
    }

    @Test
    fun testQuantileBoundaries() {
        val g = GammaDistribution(2.0, 1.0)
        assertEquals(0.0, g.quantile(0.0), 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, g.quantile(1.0))
    }

    // --- Degenerate / extreme params ---

    @Test
    fun testSmallShape() {
        val g = GammaDistribution(0.5, 2.0) // scale=0.5
        // scipy: stats.gamma(0.5, scale=0.5)
        assertEquals(0.25, g.mean, 1e-12)
        assertEquals(0.125, g.variance, 1e-12)
    }

    @Test
    fun testLargeShape() {
        val g = GammaDistribution(5.0, 0.5) // scale=2.0
        assertEquals(10.0, g.mean, 1e-12)
        assertEquals(20.0, g.variance, 1e-12)
        // scipy: stats.gamma(5, scale=2).cdf(10)
        assertEquals(0.559506714934788, g.cdf(10.0), 1e-8)
    }

    @Test
    fun testExponentialEquivalence() {
        // Gamma(1, rate) = Exponential(rate)
        val g = GammaDistribution(1.0, 1.0)
        assertEquals(1.0, g.mean, 1e-12)
        assertEquals(1.0, g.variance, 1e-12)
        assertEquals(0.367879441171442, g.sf(1.0), 1e-10)
    }

    // --- NaN/inf handling ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { GammaDistribution(0.0, 1.0) }
        assertFailsWith<InvalidParameterException> { GammaDistribution(-1.0, 1.0) }
        assertFailsWith<InvalidParameterException> { GammaDistribution(1.0, 0.0) }
        assertFailsWith<InvalidParameterException> { GammaDistribution(1.0, -1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val g = GammaDistribution(2.0, 1.0)
        assertFailsWith<InvalidParameterException> { g.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { g.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val g = GammaDistribution(2.0, 1.0)
        // Note: extreme p values (0.01, 0.99) may not converge well — see DIST-010
        for (p in listOf(0.1, 0.25, 0.5, 0.75, 0.9)) {
            assertEquals(p, g.cdf(g.quantile(p)), 1e-8, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val g = GammaDistribution(2.0, 1.0)
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 5.0, 10.0)) {
            assertEquals(1.0, g.sf(x) + g.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testSfUpperTail() {
        val g = GammaDistribution(2.0, 1.0)
        // scipy: stats.gamma(2, scale=1).sf(x)
        assertEquals(0.000499399227387334, g.sf(10.0), 1e-10)
        assertEquals(4.89443712802922e-06, g.sf(15.0), 1e-11)
        assertEquals(4.32842260712097e-08, g.sf(20.0), 1e-13)
    }

    @Test
    fun testLogPdfConsistency() {
        val g = GammaDistribution(2.0, 1.0)
        for (x in listOf(0.5, 1.0, 2.0, 5.0)) {
            assertEquals(g.pdf(x), exp(g.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleStats() {
        val g = GammaDistribution(5.0, 0.5) // mean=10, var=20
        val rng = kotlin.random.Random(42)
        val samples = g.sample(100_000, rng)
        val sampleMean = samples.average()
        assertEquals(10.0, sampleMean, 0.5, "sample mean ≈ 10")
        val sampleVar = samples.sumOf { (it - sampleMean) * (it - sampleMean) } / (samples.size - 1)
        assertEquals(g.variance, sampleVar, maxOf(g.variance * 0.1, 0.05), "sample variance ≈ ${g.variance}")
    }

    @Test
    fun testCdfMonotonicity() {
        val g = GammaDistribution(2.0, 1.0)
        var prev = 0.0
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)) {
            val cdfVal = g.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testEntropyNaN() {
        // entropy requires digamma (deferred to MATH-001)
        assertTrue(GammaDistribution(2.0, 1.0).entropy.isNaN())
    }

    @Test
    fun testPdfIntegration() {
        val d = GammaDistribution(2.0, 1.0)
        val eps = 1e-6
        val lower = d.quantile(eps)
        val upper = d.quantile(1.0 - eps)
        val integral = trapezoidalIntegral({ d.pdf(it) }, lower, upper)
        assertEquals(d.cdf(upper) - d.cdf(lower), integral, 1e-4)
    }
}
