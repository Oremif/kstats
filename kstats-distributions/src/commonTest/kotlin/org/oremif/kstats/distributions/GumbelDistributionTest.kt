package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class GumbelDistributionTest : ContinuousDistributionPropertyTests() {
    override fun createDistribution() = GumbelDistribution(2.0, 3.0)
    override val testPoints = listOf(-5.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
    private val std = GumbelDistribution.STANDARD
    private val tol = 1e-10
    private val pdfTol = 1e-12
    private val statTol = 0.05

    // --- Basic correctness (scipy reference values) ---

    @Test
    fun testPdfKnownValues() {
        // scipy: stats.gumbel_r(loc=0, scale=1).pdf(x)
        assertEquals(3.800542504044357e-08, std.pdf(-3.0), pdfTol)
        assertEquals(0.179374078734017, std.pdf(-1.0), pdfTol)
        assertEquals(0.367879441171442, std.pdf(0.0), pdfTol)
        assertEquals(0.330704298890418, std.pdf(0.5), pdfTol)
        assertEquals(0.254646380043583, std.pdf(1.0), pdfTol)
        assertEquals(0.118204951593143, std.pdf(2.0), pdfTol)
        assertEquals(0.047369009677908, std.pdf(3.0), pdfTol)
        assertEquals(0.006692699677536, std.pdf(5.0), pdfTol)
    }

    @Test
    fun testLogPdfKnownValues() {
        // scipy: stats.gumbel_r(loc=0, scale=1).logpdf(x)
        assertEquals(-17.08553692318767, std.logPdf(-3.0), pdfTol)
        assertEquals(-1.718281828459045, std.logPdf(-1.0), pdfTol)
        assertEquals(-1.0, std.logPdf(0.0), pdfTol)
        assertEquals(-1.106530659712633, std.logPdf(0.5), pdfTol)
        assertEquals(-1.367879441171442, std.logPdf(1.0), pdfTol)
        assertEquals(-2.135335283236613, std.logPdf(2.0), pdfTol)
        assertEquals(-3.049787068367864, std.logPdf(3.0), pdfTol)
        assertEquals(-5.006737946999086, std.logPdf(5.0), pdfTol)
    }

    @Test
    fun testCdfKnownValues() {
        // scipy: stats.gumbel_r(loc=0, scale=1).cdf(x)
        assertEquals(1.892178694838292e-09, std.cdf(-3.0), tol)
        assertEquals(0.065988035845313, std.cdf(-1.0), tol)
        assertEquals(0.367879441171442, std.cdf(0.0), tol)
        assertEquals(0.545239211892605, std.cdf(0.5), tol)
        assertEquals(0.692200627555346, std.cdf(1.0), tol)
        assertEquals(0.873423018493117, std.cdf(2.0), tol)
        assertEquals(0.951431992900453, std.cdf(3.0), tol)
        assertEquals(0.993284702067842, std.cdf(5.0), tol)
    }

    @Test
    fun testQuantileKnownValues() {
        // scipy: stats.gumbel_r(loc=0, scale=1).ppf(p)
        assertEquals(-1.527179625807901, std.quantile(0.01), tol)
        assertEquals(-0.834032445247956, std.quantile(0.1), tol)
        assertEquals(-0.326634259978281, std.quantile(0.25), tol)
        assertEquals(0.366512920581664, std.quantile(0.5), tol)
        assertEquals(1.245899323707238, std.quantile(0.75), tol)
        assertEquals(2.250367327312445, std.quantile(0.9), tol)
        assertEquals(4.600149226776579, std.quantile(0.99), tol)
    }

    @Test
    fun testMean() {
        // scipy: 0.5772156649015329
        assertEquals(0.5772156649015329, std.mean, tol)
    }

    @Test
    fun testVariance() {
        // pi^2 / 6 ≈ 1.6449340668482264
        assertEquals(1.644934066848226, std.variance, tol)
    }

    @Test
    fun testSkewness() {
        assertEquals(1.1395470994717452, std.skewness, tol)
    }

    @Test
    fun testKurtosis() {
        assertEquals(2.4, std.kurtosis, tol)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.gumbel_r(loc=0, scale=1).entropy() = 1.5772156649015329
        assertEquals(1.5772156649015329, std.entropy, pdfTol)
        // scipy: stats.gumbel_r(loc=2, scale=3).entropy() = 2.675827953569643
        val d = GumbelDistribution(2.0, 3.0)
        assertEquals(2.675827953569643, d.entropy, pdfTol)
    }

    // --- Non-standard parameters ---

    @Test
    fun testNonStandardPdf() {
        val d = GumbelDistribution(2.0, 3.0)
        // scipy: stats.gumbel_r(loc=2, scale=3).pdf(x)
        assertEquals(0.059791359578006, d.pdf(-1.0), pdfTol)
        assertEquals(0.092580229628578, d.pdf(0.0), pdfTol)
        assertEquals(0.122626480390481, d.pdf(2.0), pdfTol)
        assertEquals(0.084882126681194, d.pdf(5.0), pdfTol)
        assertEquals(0.021606471416297, d.pdf(10.0), pdfTol)
    }

    @Test
    fun testNonStandardLogPdf() {
        val d = GumbelDistribution(2.0, 3.0)
        // scipy: stats.gumbel_r(loc=2, scale=3).logpdf(x)
        assertEquals(-2.816894117127155, d.logPdf(-1.0), pdfTol)
        assertEquals(-2.379679663056119, d.logPdf(0.0), pdfTol)
        assertEquals(-2.098612288668110, d.logPdf(2.0), pdfTol)
        assertEquals(-2.466491729839552, d.logPdf(5.0), pdfTol)
        assertEquals(-3.834762406557578, d.logPdf(10.0), pdfTol)
    }

    @Test
    fun testNonStandardCdf() {
        val d = GumbelDistribution(2.0, 3.0)
        // scipy: stats.gumbel_r(loc=2, scale=3).cdf(x)
        assertEquals(0.065988035845313, d.cdf(-1.0), tol)
        assertEquals(0.142596824325841, d.cdf(0.0), tol)
        assertEquals(0.367879441171442, d.cdf(2.0), tol)
        assertEquals(0.692200627555346, d.cdf(5.0), tol)
        assertEquals(0.932875571206799, d.cdf(10.0), tol)
    }

    @Test
    fun testNonStandardQuantile() {
        val d = GumbelDistribution(2.0, 3.0)
        // scipy: stats.gumbel_r(loc=2, scale=3).ppf(p)
        assertEquals(-2.581538877423704, d.quantile(0.01), tol)
        assertEquals(3.099538761744993, d.quantile(0.5), tol)
        assertEquals(15.800447680329737, d.quantile(0.99), tol)
    }

    @Test
    fun testNonStandardMoments() {
        val d = GumbelDistribution(2.0, 3.0)
        // scipy: mean = 3.731646994704599
        assertEquals(3.731646994704599, d.mean, tol)
        // scipy: var = 14.80440660163404
        assertEquals(14.80440660163404, d.variance, tol)
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
    fun testPdfFarFromMean() {
        // Far left tail: pdf underflows to 0 due to double-exponential decay
        val pdfLeft = std.pdf(-10.0)
        assertTrue(pdfLeft >= 0.0, "pdf in left tail should be non-negative")
        assertTrue(pdfLeft < 1e-100, "pdf in far left tail should be essentially zero")
        // Far right tail: pdf should be small but non-zero
        val pdfRight = std.pdf(10.0)
        assertTrue(pdfRight > 0.0, "pdf in right tail should be positive")
        assertTrue(pdfRight < 1e-3, "pdf in right tail should be small")
    }

    @Test
    fun testExtremeScaleSmall() {
        val d = GumbelDistribution(0.0, 1e-6)
        // pdf at mode (x=mu) = 1/(beta*e) ≈ 367879.441
        assertEquals(1.0 / (1e-6 * kotlin.math.E), d.pdf(0.0), 1e-2)
        assertEquals(0.367879441171442, d.cdf(0.0), 1e-6)
    }

    @Test
    fun testExtremeScaleLarge() {
        val d = GumbelDistribution(0.0, 1e6)
        // pdf at mode = 1/(beta*e) ≈ 3.679e-7
        assertEquals(1.0 / (1e6 * kotlin.math.E), d.pdf(0.0), 1e-15)
        assertEquals(0.367879441171442, d.cdf(0.0), 1e-6)
    }

    // --- Non-finite ---

    @Test
    fun testPdfNaN() {
        assertEquals(Double.NaN, std.pdf(Double.NaN), 0.0)
    }

    @Test
    fun testPdfInfinity() {
        assertEquals(0.0, std.pdf(Double.POSITIVE_INFINITY), 0.0)
        assertEquals(0.0, std.pdf(Double.NEGATIVE_INFINITY), 0.0)
    }

    @Test
    fun testLogPdfInfinity() {
        assertEquals(Double.NEGATIVE_INFINITY, std.logPdf(Double.POSITIVE_INFINITY))
        assertEquals(Double.NEGATIVE_INFINITY, std.logPdf(Double.NEGATIVE_INFINITY))
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
    fun testInvalidBetaZero() {
        assertFailsWith<InvalidParameterException> { GumbelDistribution(0.0, 0.0) }
    }

    @Test
    fun testInvalidBetaNegative() {
        assertFailsWith<InvalidParameterException> { GumbelDistribution(0.0, -1.0) }
    }

    @Test
    fun testInvalidQuantileProbability() {
        assertFailsWith<InvalidParameterException> { std.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { std.quantile(1.1) }
    }
}
