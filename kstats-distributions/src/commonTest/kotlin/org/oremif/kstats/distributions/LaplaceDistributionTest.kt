package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LaplaceDistributionTest : ContinuousDistributionPropertyTests() {
    override fun createDistribution() = LaplaceDistribution(3.0, 2.0)
    override val testPoints = listOf(-5.0, -1.0, 0.0, 0.5, 1.0, 3.0, 5.0, 10.0)
    private val std = LaplaceDistribution.STANDARD
    private val tol = 1e-10
    private val pdfTol = 1e-12
    private val statTol = 0.05

    // --- Basic correctness (scipy reference values) ---

    @Test
    fun testPdfKnownValues() {
        // scipy: stats.laplace(loc=0, scale=1).pdf(x)
        assertEquals(0.024893534183932, std.pdf(-3.0), pdfTol)
        assertEquals(0.183939720585721, std.pdf(-1.0), pdfTol)
        assertEquals(0.5, std.pdf(0.0), pdfTol)
        assertEquals(0.303265329856317, std.pdf(0.5), pdfTol)
        assertEquals(0.183939720585721, std.pdf(1.0), pdfTol)
        assertEquals(0.024893534183932, std.pdf(3.0), pdfTol)
    }

    @Test
    fun testLogPdfKnownValues() {
        // scipy: stats.laplace(loc=0, scale=1).logpdf(x)
        assertEquals(-3.69314718055995, std.logPdf(-3.0), pdfTol)
        assertEquals(-1.69314718055995, std.logPdf(-1.0), pdfTol)
        assertEquals(-0.693147180559945, std.logPdf(0.0), pdfTol)
        assertEquals(-1.19314718055995, std.logPdf(0.5), pdfTol)
        assertEquals(-1.69314718055995, std.logPdf(1.0), pdfTol)
        assertEquals(-3.69314718055995, std.logPdf(3.0), pdfTol)
    }

    @Test
    fun testCdfKnownValues() {
        // scipy: stats.laplace(loc=0, scale=1).cdf(x)
        assertEquals(0.024893534183932, std.cdf(-3.0), tol)
        assertEquals(0.183939720585721, std.cdf(-1.0), tol)
        assertEquals(0.5, std.cdf(0.0), tol)
        assertEquals(0.696734670143683, std.cdf(0.5), tol)
        assertEquals(0.816060279414279, std.cdf(1.0), tol)
        assertEquals(0.975106465816068, std.cdf(3.0), tol)
    }

    @Test
    fun testQuantileKnownValues() {
        // scipy: stats.laplace(loc=0, scale=1).ppf(p)
        assertEquals(-3.91202300542815, std.quantile(0.01), tol)
        assertEquals(-1.6094379124341, std.quantile(0.1), tol)
        assertEquals(-0.693147180559945, std.quantile(0.25), tol)
        assertEquals(0.0, std.quantile(0.5), tol)
        assertEquals(0.693147180559945, std.quantile(0.75), tol)
        assertEquals(1.6094379124341, std.quantile(0.9), tol)
        assertEquals(3.91202300542815, std.quantile(0.99), tol)
    }

    @Test
    fun testMean() {
        assertEquals(0.0, std.mean, tol)
    }

    @Test
    fun testVariance() {
        assertEquals(2.0, std.variance, tol)
    }

    @Test
    fun testSkewness() {
        assertEquals(0.0, std.skewness, tol)
    }

    @Test
    fun testKurtosis() {
        // Excess kurtosis = 3
        assertEquals(3.0, std.kurtosis, tol)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.laplace(loc=0, scale=1).entropy() = 1.69314718055995
        assertEquals(1.69314718055995, std.entropy, pdfTol)
        // scipy: stats.laplace(loc=3, scale=2).entropy() = 2.38629436111989
        val d = LaplaceDistribution(3.0, 2.0)
        assertEquals(2.38629436111989, d.entropy, pdfTol)
    }

    // --- Non-standard parameters ---

    @Test
    fun testNonStandardPdf() {
        val d = LaplaceDistribution(3.0, 2.0)
        // scipy: stats.laplace(loc=3, scale=2).pdf(x)
        assertEquals(0.0338338208091532, d.pdf(-1.0), pdfTol)
        assertEquals(0.0557825400371075, d.pdf(0.0), pdfTol)
        assertEquals(0.25, d.pdf(3.0), pdfTol)
        assertEquals(0.0919698602928606, d.pdf(5.0), pdfTol)
        assertEquals(0.00754934585557963, d.pdf(10.0), pdfTol)
    }

    @Test
    fun testNonStandardLogPdf() {
        val d = LaplaceDistribution(3.0, 2.0)
        // scipy: stats.laplace(loc=3, scale=2).logpdf(x)
        assertEquals(-3.38629436111989, d.logPdf(-1.0), pdfTol)
        assertEquals(-2.88629436111989, d.logPdf(0.0), pdfTol)
        assertEquals(-1.38629436111989, d.logPdf(3.0), pdfTol)
        assertEquals(-2.38629436111989, d.logPdf(5.0), pdfTol)
        assertEquals(-4.88629436111989, d.logPdf(10.0), pdfTol)
    }

    @Test
    fun testNonStandardCdf() {
        val d = LaplaceDistribution(3.0, 2.0)
        // scipy: stats.laplace(loc=3, scale=2).cdf(x)
        assertEquals(0.0676676416183064, d.cdf(-1.0), tol)
        assertEquals(0.111565080074215, d.cdf(0.0), tol)
        assertEquals(0.5, d.cdf(3.0), tol)
        assertEquals(0.816060279414279, d.cdf(5.0), tol)
        assertEquals(0.984901308288841, d.cdf(10.0), tol)
    }

    @Test
    fun testNonStandardQuantile() {
        val d = LaplaceDistribution(3.0, 2.0)
        // scipy: stats.laplace(loc=3, scale=2).ppf(p)
        assertEquals(-4.82404601085629, d.quantile(0.01), tol)
        assertEquals(3.0, d.quantile(0.5), tol)
        assertEquals(10.8240460108563, d.quantile(0.99), tol)
    }

    @Test
    fun testNonStandardMoments() {
        val d = LaplaceDistribution(3.0, 2.0)
        assertEquals(3.0, d.mean, tol)
        assertEquals(8.0, d.variance, tol)
    }

    // --- Edge cases ---

    @Test
    fun testCdfAtExtremes() {
        assertEquals(0.0, std.cdf(Double.NEGATIVE_INFINITY), tol)
        assertEquals(1.0, std.cdf(Double.POSITIVE_INFINITY), tol)
    }

    @Test
    fun testQuantileAtBoundaries() {
        assertEquals(Double.NEGATIVE_INFINITY, std.quantile(0.0))
        assertEquals(Double.POSITIVE_INFINITY, std.quantile(1.0))
    }

    @Test
    fun testPdfSymmetry() {
        for (x in listOf(0.5, 1.0, 2.0, 5.0)) {
            assertEquals(std.pdf(x), std.pdf(-x), pdfTol, "pdf should be symmetric around mu")
        }
        val d = LaplaceDistribution(3.0, 2.0)
        for (offset in listOf(0.5, 1.0, 2.0, 5.0)) {
            assertEquals(d.pdf(3.0 + offset), d.pdf(3.0 - offset), pdfTol)
        }
    }

    // --- Extreme parameters ---

    @Test
    fun testExtremeScaleSmall() {
        val d = LaplaceDistribution(0.0, 1e-6)
        // scipy: stats.laplace(loc=0, scale=1e-6).pdf(0) = 500000
        assertEquals(500000.0, d.pdf(0.0), 1e-4)
        // scipy: cdf(1e-7) = 0.54758129098202
        assertEquals(0.54758129098202, d.cdf(1e-7), 1e-6)
    }

    @Test
    fun testExtremeScaleLarge() {
        val d = LaplaceDistribution(0.0, 1e6)
        // scipy: pdf(0) = 5e-07
        assertEquals(5e-7, d.pdf(0.0), 1e-15)
        // scipy: cdf(1e6) = 0.816060279414279
        assertEquals(0.816060279414279, d.cdf(1e6), 1e-6)
    }

    @Test
    fun testPdfFarFromMean() {
        // Far tail: pdf should be very small but not zero
        val pdfVal = std.pdf(30.0)
        assertTrue(pdfVal > 0.0, "pdf in tail should be positive")
        assertTrue(pdfVal < 1e-12, "pdf in tail should be tiny")
    }

    // --- Non-finite ---

    @Test
    fun testPdfNaN() {
        assertEquals(Double.NaN, std.pdf(Double.NaN), 0.0)
    }

    @Test
    fun testCdfNaN() {
        assertEquals(Double.NaN, std.cdf(Double.NaN), 0.0)
    }

    @Test
    fun testCdfInfinity() {
        assertEquals(1.0, std.cdf(Double.POSITIVE_INFINITY), 0.0)
        assertEquals(0.0, std.cdf(Double.NEGATIVE_INFINITY), 0.0)
    }

    // --- Validation ---

    @Test
    fun testInvalidScaleZero() {
        assertFailsWith<InvalidParameterException> { LaplaceDistribution(0.0, 0.0) }
    }

    @Test
    fun testInvalidScaleNegative() {
        assertFailsWith<InvalidParameterException> { LaplaceDistribution(0.0, -1.0) }
    }

    @Test
    fun testInvalidQuantileProbability() {
        assertFailsWith<InvalidParameterException> { std.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { std.quantile(1.1) }
    }
}
