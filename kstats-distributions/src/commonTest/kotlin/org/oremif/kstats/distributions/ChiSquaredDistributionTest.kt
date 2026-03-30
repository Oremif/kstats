package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ChiSquaredDistributionTest : ContinuousDistributionPropertyTests() {

    override fun createDistribution(): ContinuousDistribution = ChiSquaredDistribution(10.0)
    override val testPoints = listOf(0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
    override val pValues = listOf(0.1, 0.25, 0.5, 0.75, 0.9)
    override val roundTripTol = 1e-8

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val d = ChiSquaredDistribution(5.0)
        // scipy: stats.chi2(5).pdf(x)
        assertEquals(0.00400012982810046, d.pdf(0.1), 1e-12)
        assertEquals(0.0806569081730478, d.pdf(1.0), 1e-12)
        assertEquals(0.138369165806865, d.pdf(2.0), 1e-12)
        assertEquals(0.122041521349387, d.pdf(5.0), 1e-12)
        assertEquals(0.0283345553417345, d.pdf(10.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val d = ChiSquaredDistribution(5.0)
        // scipy: stats.chi2(5).cdf(x)
        assertEquals(0.000162316611922615, d.cdf(0.1), 1e-10)
        assertEquals(0.0374342267527036, d.cdf(1.0), 1e-10)
        assertEquals(0.15085496391539, d.cdf(2.0), 1e-10)
        assertEquals(0.584119813004492, d.cdf(5.0), 1e-10)
        assertEquals(0.924764753853488, d.cdf(10.0), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val d = ChiSquaredDistribution(5.0)
        // scipy: stats.chi2(5).sf(x)
        assertEquals(0.999837683388077, d.sf(0.1), 1e-10)
        assertEquals(0.962565773247296, d.sf(1.0), 1e-10)
        assertEquals(0.84914503608461, d.sf(2.0), 1e-10)
        assertEquals(0.415880186995508, d.sf(5.0), 1e-10)
        assertEquals(0.0752352461465122, d.sf(10.0), 1e-10)
    }

    @Test
    fun testSfUpperTail() {
        val d = ChiSquaredDistribution(5.0)
        // scipy: stats.chi2(5).sf(x) — far tail
        assertEquals(0.00124973056303138, d.sf(20.0), 1e-10)
        assertEquals(1.38579733670096e-09, d.sf(50.0), 1e-14)
    }

    @Test
    fun testLogPdfKnownValues() {
        val d = ChiSquaredDistribution(10.0)
        // scipy: stats.chi2(10).pdf(x) → logpdf
        assertEquals(ln(0.000789753463167491), d.logPdf(1.0), 1e-8)
        assertEquals(ln(0.0668009428905426), d.logPdf(5.0), 1e-10)
        assertEquals(ln(0.0877336848839254), d.logPdf(10.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val d = ChiSquaredDistribution(10.0)
        // scipy: stats.chi2(10).ppf(p)
        assertEquals(2.55821216018721, d.quantile(0.01), 1e-6)
        assertEquals(3.94029913611906, d.quantile(0.05), 1e-6)
        assertEquals(4.86518205192533, d.quantile(0.1), 1e-6)
        assertEquals(9.34181776559197, d.quantile(0.5), 1e-6)
        assertEquals(15.9871791721053, d.quantile(0.9), 1e-6)
        assertEquals(18.3070380532751, d.quantile(0.95), 1e-5)
        assertEquals(23.2092511589544, d.quantile(0.99), 1e-4)
    }

    @Test
    fun testMoments() {
        val d5 = ChiSquaredDistribution(5.0)
        assertEquals(5.0, d5.mean, 1e-12)
        assertEquals(10.0, d5.variance, 1e-12)
        assertEquals(sqrt(8.0 / 5.0), d5.skewness, 1e-12)
        assertEquals(12.0 / 5.0, d5.kurtosis, 1e-12)

        val d10 = ChiSquaredDistribution(10.0)
        assertEquals(10.0, d10.mean, 1e-12)
        assertEquals(20.0, d10.variance, 1e-12)
    }

    // --- Edge cases ---

    @Test
    fun testPdfAtZero() {
        // df=2: pdf(0)=0.5
        assertEquals(0.5, ChiSquaredDistribution(2.0).pdf(0.0), 1e-12)
        // df<2: pdf(0)=+Inf
        assertEquals(Double.POSITIVE_INFINITY, ChiSquaredDistribution(1.0).pdf(0.0))
        // df>2: pdf(0)=0
        assertEquals(0.0, ChiSquaredDistribution(5.0).pdf(0.0), 1e-12)
    }

    @Test
    fun testNegativeX() {
        val d = ChiSquaredDistribution(5.0)
        assertEquals(0.0, d.pdf(-1.0), 1e-12)
        assertEquals(0.0, d.cdf(-1.0), 1e-12)
        assertEquals(1.0, d.sf(-1.0), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPdf(-1.0))
    }

    @Test
    fun testQuantileBoundaries() {
        val d = ChiSquaredDistribution(5.0)
        assertEquals(0.0, d.quantile(0.0), 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, d.quantile(1.0))
    }

    // --- Different df ---

    @Test
    fun testDf1() {
        val d = ChiSquaredDistribution(1.0)
        // scipy: stats.chi2(1).cdf(x)
        assertEquals(0.682689492137086, d.cdf(1.0), 1e-10)
        assertEquals(0.974652681322532, d.cdf(5.0), 1e-10)
        // sf
        assertEquals(0.317310507862911, d.sf(1.0), 1e-10)
        assertEquals(0.0253473186774683, d.sf(5.0), 1e-10)
    }

    @Test
    fun testDf30() {
        val d = ChiSquaredDistribution(30.0)
        // scipy: stats.chi2(30)
        assertEquals(30.0, d.mean, 1e-12)
        assertEquals(60.0, d.variance, 1e-12)
        assertEquals(0.0834584729346629, d.cdf(20.0), 1e-8)
        assertEquals(0.916541527065337, d.sf(20.0), 1e-8)
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { ChiSquaredDistribution(0.0) }
        assertFailsWith<InvalidParameterException> { ChiSquaredDistribution(-1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = ChiSquaredDistribution(5.0)
        assertFailsWith<InvalidParameterException> { d.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantile(1.1) }
    }

    @Test
    fun testNaNInput() {
        val d = ChiSquaredDistribution(5.0)
        assertTrue(d.pdf(Double.NaN).isNaN())
        assertTrue(d.logPdf(Double.NaN).isNaN())
        assertTrue(d.cdf(Double.NaN).isNaN())
        assertTrue(d.sf(Double.NaN).isNaN())
        assertFailsWith<InvalidParameterException> { d.quantile(Double.NaN) }
    }

    @Test
    fun testInfinityInput() {
        val d = ChiSquaredDistribution(5.0)
        assertEquals(0.0, d.pdf(Double.POSITIVE_INFINITY))
        assertEquals(Double.NEGATIVE_INFINITY, d.logPdf(Double.POSITIVE_INFINITY))
        assertEquals(1.0, d.cdf(Double.POSITIVE_INFINITY), 1e-12)
        assertEquals(0.0, d.sf(Double.POSITIVE_INFINITY), 1e-12)
        assertEquals(0.0, d.pdf(Double.NEGATIVE_INFINITY))
        assertEquals(0.0, d.cdf(Double.NEGATIVE_INFINITY), 1e-12)
        assertEquals(1.0, d.sf(Double.NEGATIVE_INFINITY), 1e-12)
    }

    // --- Property-based ---

    @Test
    fun testEntropy() {
        assertEquals(0.783757110473934, ChiSquaredDistribution(1.0).entropy, 1e-10)
        assertEquals(1.693147180559945, ChiSquaredDistribution(2.0).entropy, 1e-10)
        assertEquals(2.423095090065000, ChiSquaredDistribution(5.0).entropy, 1e-10)
        assertEquals(2.846730337180690, ChiSquaredDistribution(10.0).entropy, 1e-10)
        assertEquals(3.708056270224887, ChiSquaredDistribution(50.0).entropy, 1e-10)
        assertEquals(-0.939420444774176, ChiSquaredDistribution(0.5).entropy, 1e-10)
    }

    @Test
    fun testExtremeParameters() {
        // df=1000: large shape parameter
        val d1 = ChiSquaredDistribution(1000.0)
        assertEquals(1000.0, d1.mean, 1e-10)
        // scipy: cdf(1000) ≈ 0.505947
        assertEquals(0.505947146170760, d1.cdf(1000.0), 1e-3)

        // df=0.1: spike near 0
        val d2 = ChiSquaredDistribution(0.1)
        // scipy: cdf(0.01) ≈ 0.787966
        assertEquals(0.787965781308072, d2.cdf(0.01), 1e-4)
    }

}
