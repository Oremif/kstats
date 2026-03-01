package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ln
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LevyDistributionTest {
    private val std = LevyDistribution.STANDARD
    private val tol = 1e-10
    private val pdfTol = 1e-12
    private val statTol = 0.10

    // ========================================
    // Config 1: Standard (mu=0, c=1)
    // scipy: stats.levy(loc=0, scale=1)
    // ========================================

    @Test
    fun testPdfKnownValues() {
        // scipy: stats.levy(loc=0, scale=1).pdf(x)
        assertEquals(0.415107497420595, std.pdf(0.5), pdfTol)
        assertEquals(0.241970724519143, std.pdf(1.0), pdfTol)
        assertEquals(0.109847822366931, std.pdf(2.0), pdfTol)
        assertEquals(0.0322868451743072, std.pdf(5.0), pdfTol)
        assertEquals(0.0120003894843014, std.pdf(10.0), pdfTol)
        assertEquals(0.00111715160678894, std.pdf(50.0), pdfTol)
    }

    @Test
    fun testLogPdfKnownValues() {
        // scipy: stats.levy(loc=0, scale=1).logpdf(x)
        assertEquals(-0.879217762364755, std.logPdf(0.5), pdfTol)
        assertEquals(-1.41893853320467, std.logPdf(1.0), pdfTol)
        assertEquals(-2.20865930404459, std.logPdf(2.0), pdfTol)
        assertEquals(-3.43309540185582, std.logPdf(5.0), pdfTol)
        assertEquals(-4.42281617269574, std.logPdf(10.0), pdfTol)
        assertEquals(-6.79697304134689, std.logPdf(50.0), pdfTol)
    }

    @Test
    fun testCdfKnownValues() {
        // scipy: stats.levy(loc=0, scale=1).cdf(x)
        assertEquals(0.157299207050285, std.cdf(0.5), tol)
        assertEquals(0.317310507862914, std.cdf(1.0), tol)
        assertEquals(0.479500122186953, std.cdf(2.0), tol)
        assertEquals(0.654720846018577, std.cdf(5.0), tol)
        assertEquals(0.751829634045849, std.cdf(10.0), tol)
        assertEquals(0.887537083981715, std.cdf(50.0), tol)
    }

    @Test
    fun testSfKnownValues() {
        // scipy: stats.levy(loc=0, scale=1).sf(x)
        assertEquals(0.842700792949715, std.sf(0.5), tol)
        assertEquals(0.682689492137086, std.sf(1.0), tol)
        assertEquals(0.520499877813047, std.sf(2.0), tol)
        assertEquals(0.345279153981423, std.sf(5.0), tol)
        assertEquals(0.248170365954151, std.sf(10.0), tol)
        assertEquals(0.112462916018285, std.sf(50.0), tol)
    }

    @Test
    fun testQuantileKnownValues() {
        // scipy: stats.levy(loc=0, scale=1).ppf(p)
        assertEquals(0.15071824930114, std.quantile(0.01), tol)
        assertEquals(0.369611509468195, std.quantile(0.1), tol)
        assertEquals(0.755684430050973, std.quantile(0.25), tol)
        assertEquals(2.19810933831773, std.quantile(0.5), tol)
        assertEquals(9.84920432182438, std.quantile(0.75), tol)
        assertEquals(63.3281176770168, std.quantile(0.9), 1e-7)
        assertEquals(6365.86438510622, std.quantile(0.99), 1e-3)
    }

    @Test
    fun testMean() {
        assertEquals(Double.POSITIVE_INFINITY, std.mean)
    }

    @Test
    fun testVariance() {
        assertEquals(Double.POSITIVE_INFINITY, std.variance)
    }

    @Test
    fun testSkewness() {
        assertTrue(std.skewness.isNaN())
    }

    @Test
    fun testKurtosis() {
        assertTrue(std.kurtosis.isNaN())
    }

    @Test
    fun testEntropy() {
        // Analytical: (1 + 3γ + ln(16πc²))/2; scipy agrees to ~1e-10 (it integrates numerically)
        assertEquals(3.32448280151174, std.entropy, 2e-10)
    }

    // ========================================
    // Config 2: Non-standard (mu=2, c=3)
    // scipy: stats.levy(loc=2, scale=3)
    // ========================================

    @Test
    fun testNonStandardPdf() {
        val d = LevyDistribution(2.0, 3.0)
        // scipy: stats.levy(loc=2, scale=3).pdf(x)
        assertEquals(0.0973043466592829, d.pdf(2.5), pdfTol)
        assertEquals(0.154180329803769, d.pdf(3.0), pdfTol)
        assertEquals(0.115399742104091, d.pdf(4.0), pdfTol)
        assertEquals(0.0457854347260912, d.pdf(7.0), pdfTol)
        assertEquals(0.0188073029768256, d.pdf(12.0), pdfTol)
        assertEquals(0.00189664850142708, d.pdf(52.0), pdfTol)
    }

    @Test
    fun testNonStandardCdf() {
        val d = LevyDistribution(2.0, 3.0)
        // scipy: stats.levy(loc=2, scale=3).cdf(x)
        assertEquals(0.0143058784354296, d.cdf(2.5), tol)
        assertEquals(0.0832645166635504, d.cdf(3.0), tol)
        assertEquals(0.220671361919847, d.cdf(4.0), tol)
        assertEquals(0.438578026081, d.cdf(7.0), tol)
        assertEquals(0.583882420770365, d.cdf(12.0), tol)
        assertEquals(0.80649594050734, d.cdf(52.0), tol)
    }

    @Test
    fun testNonStandardQuantile() {
        val d = LevyDistribution(2.0, 3.0)
        // scipy: stats.levy(loc=2, scale=3).ppf(p)
        assertEquals(2.45215474790342, d.quantile(0.01), tol)
        assertEquals(3.10883452840458, d.quantile(0.1), tol)
        assertEquals(4.26705329015292, d.quantile(0.25), tol)
        assertEquals(8.5943280149532, d.quantile(0.5), tol)
        assertEquals(31.5476129654731, d.quantile(0.75), tol)
        assertEquals(191.98435303105, d.quantile(0.9), 1e-6)
        assertEquals(19099.5931553187, d.quantile(0.99), 1e-2)
    }

    @Test
    fun testNonStandardEntropy() {
        val d = LevyDistribution(2.0, 3.0)
        // Analytical: (1 + 3γ + ln(16πc²))/2; scipy agrees to ~1e-10
        assertEquals(4.42309509017985, d.entropy, 2e-10)
    }

    // ========================================
    // Edge cases
    // ========================================

    @Test
    fun testPdfAtBoundary() {
        assertEquals(0.0, std.pdf(0.0), 0.0)
        assertEquals(0.0, std.pdf(-1.0), 0.0)
    }

    @Test
    fun testLogPdfAtBoundary() {
        assertEquals(Double.NEGATIVE_INFINITY, std.logPdf(0.0))
        assertEquals(Double.NEGATIVE_INFINITY, std.logPdf(-1.0))
    }

    @Test
    fun testCdfAtBoundary() {
        assertEquals(0.0, std.cdf(0.0), 0.0)
        assertEquals(0.0, std.cdf(-1.0), 0.0)
    }

    @Test
    fun testSfAtBoundary() {
        assertEquals(1.0, std.sf(0.0), 0.0)
        assertEquals(1.0, std.sf(-1.0), 0.0)
    }

    @Test
    fun testQuantileAtBoundaries() {
        assertEquals(0.0, std.quantile(0.0), 0.0)
        assertEquals(Double.POSITIVE_INFINITY, std.quantile(1.0))
    }

    // ========================================
    // Extreme parameters
    // ========================================

    @Test
    fun testSmallScale() {
        val d = LevyDistribution(0.0, 0.01)
        // scipy: stats.levy(loc=0, scale=0.01)
        assertEquals(41.5107497420595, d.pdf(0.005), 1e-8)
        assertEquals(24.1970724519143, d.pdf(0.01), 1e-8)
        assertEquals(0.157299207050285, d.cdf(0.005), tol)
        assertEquals(0.317310507862914, d.cdf(0.01), tol)
        assertEquals(0.0219810933831773, d.quantile(0.5), tol)
        // Analytical entropy; scipy agrees to ~1e-10
        assertEquals(-1.28068738447635, d.entropy, 2e-10)
    }

    @Test
    fun testLargeScale() {
        val d = LevyDistribution(0.0, 1000.0)
        // scipy: stats.levy(loc=0, scale=1000)
        assertEquals(0.000415107497420595, d.pdf(500.0), 1e-16)
        assertEquals(0.000241970724519143, d.pdf(1000.0), 1e-16)
        assertEquals(0.157299207050285, d.cdf(500.0), tol)
        assertEquals(0.317310507862914, d.cdf(1000.0), tol)
        assertEquals(2198.10933831773, d.quantile(0.5), 1e-6)
        // Analytical entropy; scipy agrees to ~1e-10
        assertEquals(10.2322380804939, d.entropy, 2e-10)
    }

    @Test
    fun testPdfFarFromMode() {
        val pdfVal = std.pdf(1e6)
        assertTrue(pdfVal > 0.0, "pdf in far tail should be positive")
        assertTrue(pdfVal < 1e-8, "pdf in far tail should be tiny")
    }

    // ========================================
    // Non-finite
    // ========================================

    @Test
    fun testPdfNaN() {
        assertTrue(std.pdf(Double.NaN).isNaN())
    }

    @Test
    fun testCdfInfinity() {
        assertEquals(1.0, std.cdf(Double.POSITIVE_INFINITY), 0.0)
        assertEquals(0.0, std.cdf(Double.NEGATIVE_INFINITY), 0.0)
    }

    // ========================================
    // Property-based
    // ========================================

    @Test
    fun testCdfQuantileRoundTrip() {
        val ps = doubleArrayOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9)
        for (p in ps) {
            assertEquals(p, std.cdf(std.quantile(p)), 1e-8, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testQuantileCdfRoundTrip() {
        val xs = doubleArrayOf(0.5, 1.0, 2.0, 5.0, 10.0, 50.0)
        for (x in xs) {
            assertEquals(x, std.quantile(std.cdf(x)), 1e-6, "quantile(cdf($x)) ≈ $x")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val xs = doubleArrayOf(0.5, 1.0, 2.0, 5.0, 10.0, 50.0)
        for (x in xs) {
            assertEquals(1.0, std.sf(x) + std.cdf(x), 1e-14, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val xs = doubleArrayOf(0.5, 1.0, 2.0, 5.0, 10.0, 50.0)
        for (x in xs) {
            assertEquals(ln(std.pdf(x)), std.logPdf(x), pdfTol, "logPdf($x) ≈ ln(pdf($x))")
        }
        val d = LevyDistribution(2.0, 3.0)
        for (x in doubleArrayOf(2.5, 3.0, 4.0, 7.0, 12.0, 52.0)) {
            assertEquals(ln(d.pdf(x)), d.logPdf(x), pdfTol)
        }
    }

    @Test
    fun testPdfNonNegative() {
        val xs = doubleArrayOf(-10.0, 0.0, 0.001, 0.5, 1.0, 10.0, 100.0, 1000.0)
        for (x in xs) {
            assertTrue(std.pdf(x) >= 0.0, "pdf($x) should be non-negative")
        }
    }

    @Test
    fun testCdfMonotonic() {
        val xs = listOf(0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 1000.0)
        for (i in 1 until xs.size) {
            assertTrue(std.cdf(xs[i]) >= std.cdf(xs[i - 1]), "cdf should be monotonically non-decreasing")
        }
    }

    @Test
    fun testPdfIntegration() {
        // Lévy has steep rise near mu and heavy tail — use moderate bounds
        val lower = std.quantile(0.01)
        val upper = std.quantile(0.99)
        val integral = trapezoidalIntegral({ std.pdf(it) }, lower, upper)
        assertEquals(std.cdf(upper) - std.cdf(lower), integral, 1e-2)
    }

    @Test
    fun testSampleMedian() {
        // Mean is infinite for Lévy, so test median instead
        val samples = std.sample(100_000, Random(42))
        val sortedSamples = samples.sorted()
        val sampleMedian = sortedSamples[sortedSamples.size / 2]
        val theoreticalMedian = std.quantile(0.5) // 2.198...
        assertEquals(theoreticalMedian, sampleMedian, statTol * theoreticalMedian)
    }

    @Test
    fun testSampleCdfQuantile() {
        // Verify 25th percentile from samples
        val d = LevyDistribution(2.0, 3.0)
        val samples = d.sample(100_000, Random(42)).sorted()
        val sample25 = samples[(samples.size * 0.25).toInt()]
        val theoretical25 = d.quantile(0.25) // 4.267...
        assertEquals(theoretical25, sample25, statTol * theoretical25)
    }

    // ========================================
    // Validation
    // ========================================

    @Test
    fun testInvalidScaleZero() {
        assertFailsWith<InvalidParameterException> { LevyDistribution(0.0, 0.0) }
    }

    @Test
    fun testInvalidScaleNegative() {
        assertFailsWith<InvalidParameterException> { LevyDistribution(0.0, -1.0) }
    }

    @Test
    fun testInvalidQuantileProbability() {
        assertFailsWith<InvalidParameterException> { std.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { std.quantile(1.1) }
    }
}
