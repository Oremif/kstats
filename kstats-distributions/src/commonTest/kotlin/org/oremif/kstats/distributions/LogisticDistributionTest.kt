package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LogisticDistributionTest : ContinuousDistributionPropertyTests() {

    override fun createDistribution(): ContinuousDistribution = LogisticDistribution(2.0, 3.0)
    override val testPoints = listOf(-10.0, -5.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0)

    private val std = LogisticDistribution.STANDARD
    private val tol = 1e-10
    private val pdfTol = 1e-12

    // --- Basic correctness (scipy reference values) ---

    @Test
    fun testPdfKnownValues() {
        // scipy: stats.logistic(loc=0, scale=1).pdf(x)
        assertEquals(0.0451766597309121, std.pdf(-3.0), pdfTol)
        assertEquals(0.196611933241482, std.pdf(-1.0), pdfTol)
        assertEquals(0.25, std.pdf(0.0), pdfTol)
        assertEquals(0.235003712201594, std.pdf(0.5), pdfTol)
        assertEquals(0.196611933241482, std.pdf(1.0), pdfTol)
        assertEquals(0.0451766597309121, std.pdf(3.0), pdfTol)
    }

    @Test
    fun testLogPdfKnownValues() {
        // scipy: stats.logistic(loc=0, scale=1).logpdf(x)
        assertEquals(-3.09717470314748, std.logPdf(-3.0), pdfTol)
        assertEquals(-1.62652337503645, std.logPdf(-1.0), pdfTol)
        assertEquals(-1.38629436111989, std.logPdf(0.0), pdfTol)
        assertEquals(-1.44815396836021, std.logPdf(0.5), pdfTol)
        assertEquals(-1.62652337503645, std.logPdf(1.0), pdfTol)
        assertEquals(-3.09717470314748, std.logPdf(3.0), pdfTol)
    }

    @Test
    fun testCdfKnownValues() {
        // scipy: stats.logistic(loc=0, scale=1).cdf(x)
        assertEquals(0.0474258731775668, std.cdf(-3.0), tol)
        assertEquals(0.268941421369995, std.cdf(-1.0), tol)
        assertEquals(0.5, std.cdf(0.0), tol)
        assertEquals(0.622459331201855, std.cdf(0.5), tol)
        assertEquals(0.731058578630005, std.cdf(1.0), tol)
        assertEquals(0.952574126822433, std.cdf(3.0), tol)
    }

    @Test
    fun testQuantileKnownValues() {
        // scipy: stats.logistic(loc=0, scale=1).ppf(p)
        assertEquals(-4.59511985013459, std.quantile(0.01), tol)
        assertEquals(-2.19722457733622, std.quantile(0.1), tol)
        assertEquals(-1.09861228866811, std.quantile(0.25), tol)
        assertEquals(0.0, std.quantile(0.5), tol)
        assertEquals(1.09861228866811, std.quantile(0.75), tol)
        assertEquals(2.19722457733622, std.quantile(0.9), tol)
        assertEquals(4.59511985013459, std.quantile(0.99), tol)
    }

    @Test
    fun testMean() {
        assertEquals(0.0, std.mean, tol)
    }

    @Test
    fun testVariance() {
        // pi^2 / 3 ≈ 3.28986813369645
        assertEquals(kotlin.math.PI * kotlin.math.PI / 3.0, std.variance, tol)
    }

    @Test
    fun testSkewness() {
        assertEquals(0.0, std.skewness, tol)
    }

    @Test
    fun testKurtosis() {
        // Excess kurtosis = 6/5 = 1.2
        assertEquals(1.2, std.kurtosis, tol)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.logistic(loc=0, scale=1).entropy() = 2.0
        assertEquals(2.0, std.entropy, pdfTol)
        // scipy: stats.logistic(loc=2, scale=3).entropy() = 3.09861228866811
        val d = LogisticDistribution(2.0, 3.0)
        assertEquals(3.09861228866811, d.entropy, pdfTol)
    }

    // --- Non-standard parameters ---

    @Test
    fun testNonStandardPdf() {
        val d = LogisticDistribution(2.0, 3.0)
        // scipy: stats.logistic(loc=2, scale=3).pdf(x)
        assertEquals(0.0655373110804939, d.pdf(-1.0), pdfTol)
        assertEquals(0.0747191299670762, d.pdf(0.0), pdfTol)
        assertEquals(0.0833333333333333, d.pdf(2.0), pdfTol)
        assertEquals(0.0655373110804939, d.pdf(5.0), pdfTol)
        assertEquals(0.0202493920637984, d.pdf(10.0), pdfTol)
    }

    @Test
    fun testNonStandardLogPdf() {
        val d = LogisticDistribution(2.0, 3.0)
        // scipy: stats.logistic(loc=2, scale=3).logpdf(x)
        assertEquals(-2.72513566370456, d.logPdf(-1.0), pdfTol)
        assertEquals(-2.59401912903892, d.logPdf(0.0), pdfTol)
        assertEquals(-2.484906649788, d.logPdf(2.0), pdfTol)
        assertEquals(-2.72513566370456, d.logPdf(5.0), pdfTol)
        assertEquals(-3.89963050742107, d.logPdf(10.0), pdfTol)
    }

    @Test
    fun testNonStandardCdf() {
        val d = LogisticDistribution(2.0, 3.0)
        // scipy: stats.logistic(loc=2, scale=3).cdf(x)
        assertEquals(0.268941421369995, d.cdf(-1.0), tol)
        assertEquals(0.339243631234183, d.cdf(0.0), tol)
        assertEquals(0.5, d.cdf(2.0), tol)
        assertEquals(0.731058578630005, d.cdf(5.0), tol)
        assertEquals(0.935030830871336, d.cdf(10.0), tol)
    }

    @Test
    fun testNonStandardQuantile() {
        val d = LogisticDistribution(2.0, 3.0)
        // scipy: stats.logistic(loc=2, scale=3).ppf(p)
        assertEquals(-11.7853595504038, d.quantile(0.01), tol)
        assertEquals(2.0, d.quantile(0.5), tol)
        assertEquals(15.7853595504038, d.quantile(0.99), tol)
    }

    @Test
    fun testNonStandardMoments() {
        val d = LogisticDistribution(2.0, 3.0)
        assertEquals(2.0, d.mean, tol)
        assertEquals(kotlin.math.PI * kotlin.math.PI * 9.0 / 3.0, d.variance, tol)
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
        val d = LogisticDistribution(2.0, 3.0)
        for (offset in listOf(0.5, 1.0, 2.0, 5.0)) {
            assertEquals(d.pdf(2.0 + offset), d.pdf(2.0 - offset), pdfTol)
        }
    }

    // --- Extreme parameters ---

    @Test
    fun testExtremeScaleSmall() {
        val d = LogisticDistribution(0.0, 1e-6)
        // scipy: stats.logistic(loc=0, scale=1e-6).pdf(0) = 250000
        assertEquals(250000.0, d.pdf(0.0), 1e-4)
        // scipy: cdf(1e-7) = 0.52497918747894
        assertEquals(0.52497918747894, d.cdf(1e-7), 1e-6)
    }

    @Test
    fun testExtremeScaleLarge() {
        val d = LogisticDistribution(0.0, 1e6)
        // scipy: pdf(0) = 2.5e-07
        assertEquals(2.5e-7, d.pdf(0.0), 1e-15)
        // scipy: cdf(1e6) = 0.731058578630005
        assertEquals(0.731058578630005, d.cdf(1e6), 1e-6)
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
        assertFailsWith<InvalidParameterException> { LogisticDistribution(0.0, 0.0) }
    }

    @Test
    fun testInvalidScaleNegative() {
        assertFailsWith<InvalidParameterException> { LogisticDistribution(0.0, -1.0) }
    }

    @Test
    fun testInvalidScaleNaN() {
        assertFailsWith<InvalidParameterException> { LogisticDistribution(0.0, Double.NaN) }
    }

    @Test
    fun testInvalidMuNaN() {
        assertFailsWith<InvalidParameterException> { LogisticDistribution(Double.NaN, 1.0) }
    }

    @Test
    fun testInvalidMuInfinity() {
        assertFailsWith<InvalidParameterException> { LogisticDistribution(Double.POSITIVE_INFINITY, 1.0) }
        assertFailsWith<InvalidParameterException> { LogisticDistribution(Double.NEGATIVE_INFINITY, 1.0) }
    }

    @Test
    fun testInvalidQuantileProbability() {
        assertFailsWith<InvalidParameterException> { std.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { std.quantile(1.1) }
    }
}
