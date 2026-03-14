package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.math.ln
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TriangularDistributionTest {
    // Config 1: Symmetric — a=0, b=1, c=0.5
    private val sym = TriangularDistribution(0.0, 1.0, 0.5)
    // Config 2: Asymmetric — a=1, b=5, c=3
    private val asym = TriangularDistribution(1.0, 5.0, 3.0)
    // Config 3: Mode at left — a=-2, b=4, c=-2
    private val modeLeft = TriangularDistribution(-2.0, 4.0, -2.0)
    // Config 4: Mode at right — a=0, b=10, c=10
    private val modeRight = TriangularDistribution(0.0, 10.0, 10.0)

    private val tol = 1e-10
    private val pdfTol = 1e-12
    private val statTol = 0.05

    // ========================================
    // Config 1: Symmetric (a=0, b=1, c=0.5)
    // scipy: stats.triang(c=0.5, loc=0, scale=1)
    // ========================================

    @Test
    fun testSymPdfKnownValues() {
        assertEquals(0.0, sym.pdf(-0.1), pdfTol)
        assertEquals(0.0, sym.pdf(0.0), pdfTol)
        assertEquals(0.4, sym.pdf(0.1), pdfTol)
        assertEquals(1.0, sym.pdf(0.25), pdfTol)
        assertEquals(2.0, sym.pdf(0.5), pdfTol)
        assertEquals(1.0, sym.pdf(0.75), pdfTol)
        assertEquals(0.4, sym.pdf(0.9), pdfTol)
        assertEquals(0.0, sym.pdf(1.0), pdfTol)
        assertEquals(0.0, sym.pdf(1.1), pdfTol)
    }

    @Test
    fun testSymLogPdfKnownValues() {
        assertEquals(Double.NEGATIVE_INFINITY, sym.logPdf(-0.1))
        assertEquals(ln(0.4), sym.logPdf(0.1), pdfTol)
        assertEquals(ln(2.0), sym.logPdf(0.5), pdfTol)
        assertEquals(ln(0.4), sym.logPdf(0.9), pdfTol)
        assertEquals(Double.NEGATIVE_INFINITY, sym.logPdf(1.1))
    }

    @Test
    fun testSymCdfKnownValues() {
        // scipy: stats.triang(c=0.5, loc=0, scale=1).cdf(x)
        assertEquals(0.0, sym.cdf(0.0), tol)
        assertEquals(0.02, sym.cdf(0.1), tol)
        assertEquals(0.125, sym.cdf(0.25), tol)
        assertEquals(0.5, sym.cdf(0.5), tol)
        assertEquals(0.875, sym.cdf(0.75), tol)
        assertEquals(0.98, sym.cdf(0.9), tol)
        assertEquals(1.0, sym.cdf(1.0), tol)
    }

    @Test
    fun testSymQuantileKnownValues() {
        // scipy: stats.triang(c=0.5, loc=0, scale=1).ppf(p)
        assertEquals(0.0, sym.quantile(0.0), tol)
        assertEquals(0.22360679774997896, sym.quantile(0.1), tol)
        assertEquals(0.35355339059327373, sym.quantile(0.25), tol)
        assertEquals(0.5, sym.quantile(0.5), tol)
        assertEquals(0.6464466094067263, sym.quantile(0.75), tol)
        assertEquals(0.7763932022500210, sym.quantile(0.9), tol)
        assertEquals(1.0, sym.quantile(1.0), tol)
    }

    @Test
    fun testSymStatistics() {
        // mean = (0+1+0.5)/3 = 0.5
        assertEquals(0.5, sym.mean, tol)
        // variance = (0+1+0.25-0-0-0.5)/18 = 0.75/18 = 1/24
        assertEquals(1.0 / 24.0, sym.variance, tol)
        // skewness = 0 (symmetric)
        assertEquals(0.0, sym.skewness, tol)
        // kurtosis = -0.6
        assertEquals(-0.6, sym.kurtosis, tol)
        // entropy = 0.5 + ln(0.5)
        assertEquals(0.5 + ln(0.5), sym.entropy, pdfTol)
    }

    // ========================================
    // Config 2: Asymmetric (a=1, b=5, c=3)
    // scipy: stats.triang(c=0.5, loc=1, scale=4)
    // ========================================

    @Test
    fun testAsymPdfKnownValues() {
        // scipy: stats.triang(c=0.5, loc=1, scale=4).pdf(x)
        assertEquals(0.0, asym.pdf(0.5), pdfTol)
        assertEquals(0.0, asym.pdf(1.0), pdfTol)
        assertEquals(0.125, asym.pdf(1.5), pdfTol)
        assertEquals(0.25, asym.pdf(2.0), pdfTol)
        assertEquals(0.5, asym.pdf(3.0), pdfTol)
        assertEquals(0.25, asym.pdf(4.0), pdfTol)
        assertEquals(0.0, asym.pdf(5.0), pdfTol)
        assertEquals(0.0, asym.pdf(5.5), pdfTol)
    }

    @Test
    fun testAsymCdfKnownValues() {
        // scipy: stats.triang(c=0.5, loc=1, scale=4).cdf(x)
        assertEquals(0.0, asym.cdf(1.0), tol)
        assertEquals(0.03125, asym.cdf(1.5), tol)
        assertEquals(0.125, asym.cdf(2.0), tol)
        assertEquals(0.5, asym.cdf(3.0), tol)
        assertEquals(0.875, asym.cdf(4.0), tol)
        assertEquals(1.0, asym.cdf(5.0), tol)
    }

    @Test
    fun testAsymQuantileKnownValues() {
        // scipy: stats.triang(c=0.5, loc=1, scale=4).ppf(p)
        assertEquals(1.0, asym.quantile(0.0), tol)
        assertEquals(1.8944271909999159, asym.quantile(0.1), tol)
        assertEquals(2.414213562373095, asym.quantile(0.25), tol)
        assertEquals(3.0, asym.quantile(0.5), tol)
        assertEquals(3.585786437626905, asym.quantile(0.75), tol)
        assertEquals(4.105572809000084, asym.quantile(0.9), tol)
        assertEquals(5.0, asym.quantile(1.0), tol)
    }

    @Test
    fun testAsymStatistics() {
        // mean = (1+5+3)/3 = 3.0
        assertEquals(3.0, asym.mean, tol)
        // variance = (1+25+9-5-3-15)/18 = 12/18 = 2/3
        assertEquals(2.0 / 3.0, asym.variance, tol)
        assertEquals(-0.6, asym.kurtosis, tol)
        // entropy = 0.5 + ln(4/2) = 0.5 + ln(2)
        assertEquals(0.5 + ln(2.0), asym.entropy, pdfTol)
    }

    // ========================================
    // Config 3: Mode at left (a=-2, b=4, c=-2)
    // scipy: stats.triang(c=0.0, loc=-2, scale=6)
    // ========================================

    @Test
    fun testModeLeftPdfKnownValues() {
        // Right triangle: pdf(a)=2/(b-a)=1/3, linearly decreasing to 0 at b
        assertEquals(1.0 / 3.0, modeLeft.pdf(-2.0), pdfTol)
        assertEquals(2.0 / 9.0, modeLeft.pdf(0.0), pdfTol)
        assertEquals(1.0 / 9.0, modeLeft.pdf(2.0), pdfTol)
        assertEquals(0.0, modeLeft.pdf(4.0), pdfTol)
    }

    @Test
    fun testModeLeftCdfKnownValues() {
        // scipy: stats.triang(c=0.0, loc=-2, scale=6).cdf(x)
        assertEquals(0.0, modeLeft.cdf(-2.0), tol)
        // cdf(0) = 1 - (4-0)^2/(6*6) = 1 - 16/36 = 1 - 4/9
        assertEquals(1.0 - 16.0 / 36.0, modeLeft.cdf(0.0), tol)
        // cdf(2) = 1 - (4-2)^2/(6*6) = 1 - 4/36 = 1 - 1/9
        assertEquals(1.0 - 4.0 / 36.0, modeLeft.cdf(2.0), tol)
        assertEquals(1.0, modeLeft.cdf(4.0), tol)
    }

    @Test
    fun testModeLeftQuantileKnownValues() {
        assertEquals(-2.0, modeLeft.quantile(0.0), tol)
        assertEquals(4.0, modeLeft.quantile(1.0), tol)
        // quantile(0.5): pc=0, so p>pc → b - sqrt((1-p)*(b-a)*(b-c))
        // = 4 - sqrt(0.5*6*6) = 4 - sqrt(18) = 4 - 3*sqrt(2)
        assertEquals(4.0 - 3.0 * kotlin.math.sqrt(2.0), modeLeft.quantile(0.5), tol)
    }

    @Test
    fun testModeLeftStatistics() {
        // mean = (-2+4+(-2))/3 = 0
        assertEquals(0.0, modeLeft.mean, tol)
        // variance = (4+16+4-(-8)-(-4+8))/18 = (4+16+4+8-(-4)-8))/18
        // a²=4, b²=16, c²=4, ab=-8, ac=4, bc=-8
        // (4+16+4-(-8)-4-(-8))/18 = (4+16+4+8-4+8)/18 = 36/18 = 2
        assertEquals(2.0, modeLeft.variance, tol)
        assertEquals(-0.6, modeLeft.kurtosis, tol)
    }

    // ========================================
    // Config 4: Mode at right (a=0, b=10, c=10)
    // scipy: stats.triang(c=1.0, loc=0, scale=10)
    // ========================================

    @Test
    fun testModeRightPdfKnownValues() {
        // Left triangle: linearly increasing from 0 at a to 2/(b-a)=0.2 at b
        assertEquals(0.0, modeRight.pdf(0.0), pdfTol)
        assertEquals(0.04, modeRight.pdf(2.0), pdfTol)
        assertEquals(0.1, modeRight.pdf(5.0), pdfTol)
        assertEquals(0.16, modeRight.pdf(8.0), pdfTol)
        assertEquals(0.2, modeRight.pdf(10.0), pdfTol)
    }

    @Test
    fun testModeRightCdfKnownValues() {
        // cdf(x) = (x-a)^2/((b-a)(c-a)) = x^2/100 for x<=10
        assertEquals(0.0, modeRight.cdf(0.0), tol)
        assertEquals(0.04, modeRight.cdf(2.0), tol)
        assertEquals(0.25, modeRight.cdf(5.0), tol)
        assertEquals(0.64, modeRight.cdf(8.0), tol)
        assertEquals(1.0, modeRight.cdf(10.0), tol)
    }

    @Test
    fun testModeRightQuantileKnownValues() {
        assertEquals(0.0, modeRight.quantile(0.0), tol)
        assertEquals(10.0, modeRight.quantile(1.0), tol)
        // quantile(0.5): pc=1.0, so p<=pc → a + sqrt(p*(b-a)*(c-a))
        // = 0 + sqrt(0.5*10*10) = sqrt(50) = 5*sqrt(2)
        assertEquals(5.0 * kotlin.math.sqrt(2.0), modeRight.quantile(0.5), tol)
    }

    @Test
    fun testModeRightStatistics() {
        // mean = (0+10+10)/3 = 20/3
        assertEquals(20.0 / 3.0, modeRight.mean, tol)
        assertEquals(-0.6, modeRight.kurtosis, tol)
    }

    // ========================================
    // Edge cases
    // ========================================

    @Test
    fun testPdfOutsideSupport() {
        assertEquals(0.0, sym.pdf(-1.0), 0.0)
        assertEquals(0.0, sym.pdf(2.0), 0.0)
        assertEquals(0.0, asym.pdf(0.0), 0.0)
        assertEquals(0.0, asym.pdf(6.0), 0.0)
    }

    @Test
    fun testLogPdfOutsideSupport() {
        assertEquals(Double.NEGATIVE_INFINITY, sym.logPdf(-1.0))
        assertEquals(Double.NEGATIVE_INFINITY, sym.logPdf(2.0))
    }

    @Test
    fun testCdfOutsideSupport() {
        assertEquals(0.0, sym.cdf(-1.0), 0.0)
        assertEquals(1.0, sym.cdf(2.0), 0.0)
    }

    @Test
    fun testSfOutsideSupport() {
        assertEquals(1.0, sym.sf(-1.0), 0.0)
        assertEquals(0.0, sym.sf(2.0), 0.0)
    }

    @Test
    fun testExtremeParameters() {
        val d = TriangularDistribution(0.0, 1e6, 500000.0)
        assertEquals(500000.0 / 3.0 + 1e6 / 3.0, d.mean, 1.0)
        assertTrue(d.pdf(500000.0) > 0.0)
        assertTrue(d.cdf(500000.0) in 0.0..1.0)
        assertEquals(0.0, d.cdf(-1.0), 0.0)
        assertEquals(1.0, d.cdf(2e6), 0.0)
    }

    // ========================================
    // Property-based tests
    // ========================================

    @Test
    fun testCdfQuantileRoundTrip() {
        val ps = doubleArrayOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        for (dist in listOf(sym, asym, modeLeft, modeRight)) {
            for (p in ps) {
                assertEquals(p, dist.cdf(dist.quantile(p)), tol, "cdf(quantile($p)) ≈ $p for $dist")
            }
        }
    }

    @Test
    fun testQuantileCdfRoundTrip() {
        for (dist in listOf(sym, asym)) {
            val xs = doubleArrayOf(
                dist.quantile(0.1), dist.quantile(0.3), dist.quantile(0.5),
                dist.quantile(0.7), dist.quantile(0.9)
            )
            for (x in xs) {
                assertEquals(x, dist.quantile(dist.cdf(x)), tol, "quantile(cdf($x)) ≈ $x for $dist")
            }
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        for (dist in listOf(sym, asym, modeLeft, modeRight)) {
            val xs = doubleArrayOf(
                dist.quantile(0.01), dist.quantile(0.1), dist.quantile(0.25),
                dist.quantile(0.5), dist.quantile(0.75), dist.quantile(0.9), dist.quantile(0.99)
            )
            for (x in xs) {
                assertEquals(1.0, dist.sf(x) + dist.cdf(x), 1e-14, "sf($x) + cdf($x) ≈ 1 for $dist")
            }
        }
    }

    @Test
    fun testLogPdfConsistency() {
        for (dist in listOf(sym, asym)) {
            val xs = doubleArrayOf(
                dist.quantile(0.1), dist.quantile(0.3), dist.quantile(0.5),
                dist.quantile(0.7), dist.quantile(0.9)
            )
            for (x in xs) {
                assertEquals(ln(dist.pdf(x)), dist.logPdf(x), pdfTol, "logPdf($x) ≈ ln(pdf($x))")
            }
        }
    }

    @Test
    fun testPdfNonNegative() {
        for (dist in listOf(sym, asym, modeLeft, modeRight)) {
            for (i in 0..100) {
                val x = dist.quantile(i / 100.0)
                assertTrue(dist.pdf(x) >= 0.0, "pdf($x) should be non-negative for $dist")
            }
        }
    }

    @Test
    fun testCdfMonotonic() {
        for (dist in listOf(sym, asym, modeLeft, modeRight)) {
            val xs = (0..100).map { dist.quantile(it / 100.0) }
            for (i in 1 until xs.size) {
                assertTrue(
                    dist.cdf(xs[i]) >= dist.cdf(xs[i - 1]),
                    "cdf should be monotonically non-decreasing for $dist"
                )
            }
        }
    }

    @Test
    fun testPdfIntegration() {
        for (dist in listOf(sym, asym, modeLeft, modeRight)) {
            val eps = 1e-6
            val lower = dist.quantile(eps)
            val upper = dist.quantile(1.0 - eps)
            val integral = trapezoidalIntegral({ dist.pdf(it) }, lower, upper)
            assertEquals(dist.cdf(upper) - dist.cdf(lower), integral, 1e-4, "pdf integration for $dist")
        }
    }

    @Test
    fun testSampleMean() {
        for (dist in listOf(sym, asym, modeLeft, modeRight)) {
            val samples = dist.sample(100_000, Random(42))
            val sampleMean = samples.average()
            assertEquals(dist.mean, sampleMean, statTol * abs(dist.mean).coerceAtLeast(1.0), "sample mean for $dist")
        }
    }

    @Test
    fun testSampleVariance() {
        for (dist in listOf(sym, asym)) {
            val samples = dist.sample(100_000, Random(42))
            val sampleMean = samples.average()
            val sampleVar = samples.sumOf { (it - sampleMean) * (it - sampleMean) } / (samples.size - 1)
            assertEquals(
                dist.variance, sampleVar,
                statTol * dist.variance.coerceAtLeast(1.0), "sample variance for $dist"
            )
        }
    }

    // ========================================
    // Validation
    // ========================================

    @Test
    fun testInvalidAGreaterThanB() {
        assertFailsWith<InvalidParameterException> { TriangularDistribution(5.0, 1.0, 3.0) }
    }

    @Test
    fun testInvalidAEqualsB() {
        assertFailsWith<InvalidParameterException> { TriangularDistribution(1.0, 1.0, 1.0) }
    }

    @Test
    fun testInvalidCBelowA() {
        assertFailsWith<InvalidParameterException> { TriangularDistribution(0.0, 1.0, -0.1) }
    }

    @Test
    fun testInvalidCAboveB() {
        assertFailsWith<InvalidParameterException> { TriangularDistribution(0.0, 1.0, 1.1) }
    }

    @Test
    fun testInvalidQuantileProbability() {
        assertFailsWith<InvalidParameterException> { sym.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { sym.quantile(1.1) }
    }
}
