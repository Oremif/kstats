package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.math.ln
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ParetoDistributionTest {
    private val std = ParetoDistribution.STANDARD
    private val tol = 1e-10
    private val pdfTol = 1e-12
    private val statTol = 0.05

    // --- Basic correctness (scipy reference values) ---

    @Test
    fun testPdfKnownValues() {
        // scipy: pareto(b=1, scale=1).pdf(x)
        assertEquals(1.0, std.pdf(1.0), pdfTol)
        assertEquals(0.444444444444444, std.pdf(1.5), pdfTol)
        assertEquals(0.25, std.pdf(2.0), pdfTol)
        assertEquals(0.111111111111111, std.pdf(3.0), pdfTol)
        assertEquals(0.04, std.pdf(5.0), pdfTol)
        assertEquals(0.01, std.pdf(10.0), pdfTol)
    }

    @Test
    fun testLogPdfKnownValues() {
        // scipy: pareto(b=1, scale=1).logpdf(x)
        assertEquals(0.0, std.logPdf(1.0), pdfTol)
        assertEquals(-0.810930216216329, std.logPdf(1.5), pdfTol)
        assertEquals(-1.38629436111989, std.logPdf(2.0), pdfTol)
        assertEquals(-2.19722457733622, std.logPdf(3.0), pdfTol)
        assertEquals(-3.2188758248682, std.logPdf(5.0), pdfTol)
        assertEquals(-4.60517018598809, std.logPdf(10.0), pdfTol)
    }

    @Test
    fun testCdfKnownValues() {
        // scipy: pareto(b=1, scale=1).cdf(x)
        assertEquals(0.0, std.cdf(1.0), tol)
        assertEquals(0.333333333333333, std.cdf(1.5), tol)
        assertEquals(0.5, std.cdf(2.0), tol)
        assertEquals(0.666666666666667, std.cdf(3.0), tol)
        assertEquals(0.8, std.cdf(5.0), tol)
        assertEquals(0.9, std.cdf(10.0), tol)
    }

    @Test
    fun testQuantileKnownValues() {
        // scipy: pareto(b=1, scale=1).ppf(p)
        assertEquals(1.01010101010101, std.quantile(0.01), tol)
        assertEquals(1.11111111111111, std.quantile(0.1), tol)
        assertEquals(1.33333333333333, std.quantile(0.25), tol)
        assertEquals(2.0, std.quantile(0.5), tol)
        assertEquals(4.0, std.quantile(0.75), tol)
        assertEquals(10.0, std.quantile(0.9), tol)
        assertEquals(99.9999999999999, std.quantile(0.99), 1e-6)
    }

    @Test
    fun testMean() {
        // shape=1: mean is infinite
        assertEquals(Double.POSITIVE_INFINITY, std.mean)
    }

    @Test
    fun testVariance() {
        // shape=1: variance is infinite
        assertEquals(Double.POSITIVE_INFINITY, std.variance)
    }

    @Test
    fun testSkewness() {
        // shape=1: skewness undefined (α ≤ 3)
        assertTrue(std.skewness.isNaN())
    }

    @Test
    fun testKurtosis() {
        // shape=1: kurtosis undefined (α ≤ 4)
        assertTrue(std.kurtosis.isNaN())
    }

    @Test
    fun testEntropy() {
        // scipy: pareto(b=1, scale=1).entropy() = 2.0
        assertEquals(2.0, std.entropy, pdfTol)
        // scipy: pareto(b=3, scale=2).entropy() = 0.927868225225169
        val d = ParetoDistribution(3.0, 2.0)
        assertEquals(0.927868225225169, d.entropy, pdfTol)
    }

    // --- Non-standard parameters (shape=3, scale=2) ---

    @Test
    fun testNonStandardPdf() {
        val d = ParetoDistribution(3.0, 2.0)
        // scipy: pareto(b=3, scale=2).pdf(x)
        assertEquals(1.5, d.pdf(2.0), pdfTol)
        assertEquals(0.6144, d.pdf(2.5), pdfTol)
        assertEquals(0.296296296296296, d.pdf(3.0), pdfTol)
        assertEquals(0.09375, d.pdf(4.0), pdfTol)
        assertEquals(0.0384, d.pdf(5.0), pdfTol)
        assertEquals(0.0024, d.pdf(10.0), pdfTol)
    }

    @Test
    fun testNonStandardLogPdf() {
        val d = ParetoDistribution(3.0, 2.0)
        // scipy: pareto(b=3, scale=2).logpdf(x)
        assertEquals(0.405465108108164, d.logPdf(2.0), pdfTol)
        assertEquals(-0.487109097148675, d.logPdf(2.5), pdfTol)
        assertEquals(-1.21639532432449, d.logPdf(3.0), pdfTol)
        assertEquals(-2.36712361413162, d.logPdf(4.0), pdfTol)
        assertEquals(-3.25969781938846, d.logPdf(5.0), pdfTol)
        assertEquals(-6.03228654162824, d.logPdf(10.0), pdfTol)
    }

    @Test
    fun testNonStandardCdf() {
        val d = ParetoDistribution(3.0, 2.0)
        // scipy: pareto(b=3, scale=2).cdf(x)
        assertEquals(0.0, d.cdf(2.0), tol)
        assertEquals(0.488, d.cdf(2.5), tol)
        assertEquals(0.703703703703704, d.cdf(3.0), tol)
        assertEquals(0.875, d.cdf(4.0), tol)
        assertEquals(0.936, d.cdf(5.0), tol)
        assertEquals(0.992, d.cdf(10.0), tol)
    }

    @Test
    fun testNonStandardQuantile() {
        val d = ParetoDistribution(3.0, 2.0)
        // scipy: pareto(b=3, scale=2).ppf(p)
        assertEquals(2.00671145969597, d.quantile(0.01), tol)
        assertEquals(2.07148833730257, d.quantile(0.1), tol)
        assertEquals(2.20128483259642, d.quantile(0.25), tol)
        assertEquals(2.51984209978975, d.quantile(0.5), tol)
        assertEquals(3.1748021039364, d.quantile(0.75), tol)
        assertEquals(4.30886938006377, d.quantile(0.9), tol)
        assertEquals(9.28317766722556, d.quantile(0.99), tol)
    }

    @Test
    fun testNonStandardMoments() {
        val d = ParetoDistribution(3.0, 2.0)
        assertEquals(3.0, d.mean, tol)
        assertEquals(3.0, d.variance, tol)
        // shape=3: skewness undefined (α ≤ 3)
        assertTrue(d.skewness.isNaN())
        // shape=3: kurtosis undefined (α ≤ 4)
        assertTrue(d.kurtosis.isNaN())
    }

    // --- All moments defined (shape=5, scale=1) ---

    @Test
    fun testAllMomentsDefined() {
        val d = ParetoDistribution(5.0, 1.0)
        // scipy: pareto(b=5, scale=1)
        assertEquals(5.0, d.pdf(1.0), pdfTol)
        assertEquals(0.438957475994513, d.pdf(1.5), pdfTol)
        assertEquals(0.078125, d.pdf(2.0), pdfTol)

        assertEquals(0.0, d.cdf(1.0), tol)
        assertEquals(0.868312757201646, d.cdf(1.5), tol)
        assertEquals(0.96875, d.cdf(2.0), tol)

        assertEquals(1.25, d.mean, tol)
        assertEquals(0.104166666666667, d.variance, tol)
        assertEquals(4.6475800154489, d.skewness, tol)
        assertEquals(70.8, d.kurtosis, tol)
        assertEquals(-0.4094379124341, d.entropy, pdfTol)
    }

    // --- Edge cases ---

    @Test
    fun testPdfBelowSupport() {
        assertEquals(0.0, std.pdf(0.5), 0.0)
        assertEquals(0.0, std.pdf(0.0), 0.0)
        assertEquals(0.0, std.pdf(-1.0), 0.0)
        val d = ParetoDistribution(3.0, 2.0)
        assertEquals(0.0, d.pdf(1.5), 0.0)
        assertEquals(0.0, d.pdf(1.9999), 0.0)
    }

    @Test
    fun testCdfBelowSupport() {
        assertEquals(0.0, std.cdf(0.5), 0.0)
        assertEquals(0.0, std.cdf(0.0), 0.0)
        assertEquals(0.0, std.cdf(-1.0), 0.0)
        val d = ParetoDistribution(3.0, 2.0)
        assertEquals(0.0, d.cdf(1.5), 0.0)
    }

    @Test
    fun testLogPdfBelowSupport() {
        assertEquals(Double.NEGATIVE_INFINITY, std.logPdf(0.5))
        assertEquals(Double.NEGATIVE_INFINITY, std.logPdf(-1.0))
    }

    @Test
    fun testSfBelowSupport() {
        assertEquals(1.0, std.sf(0.5), 0.0)
        assertEquals(1.0, std.sf(-1.0), 0.0)
    }

    @Test
    fun testPdfAtScale() {
        // At x = scale, pdf = shape / scale
        assertEquals(1.0, std.pdf(1.0), pdfTol)
        val d = ParetoDistribution(3.0, 2.0)
        assertEquals(1.5, d.pdf(2.0), pdfTol)
    }

    @Test
    fun testCdfAtScale() {
        assertEquals(0.0, std.cdf(1.0), tol)
        val d = ParetoDistribution(3.0, 2.0)
        assertEquals(0.0, d.cdf(2.0), tol)
    }

    @Test
    fun testQuantileAtBoundaries() {
        assertEquals(1.0, std.quantile(0.0), tol) // quantile(0) = scale
        assertEquals(Double.POSITIVE_INFINITY, std.quantile(1.0))

        val d = ParetoDistribution(3.0, 2.0)
        assertEquals(2.0, d.quantile(0.0), tol) // quantile(0) = scale
        assertEquals(Double.POSITIVE_INFINITY, d.quantile(1.0))
    }

    @Test
    fun testSfAtScale() {
        assertEquals(1.0, std.sf(1.0), tol)
        val d = ParetoDistribution(3.0, 2.0)
        assertEquals(1.0, d.sf(2.0), tol)
    }

    @Test
    fun testSfKnownValues() {
        // scipy: pareto(b=1, scale=1).sf(x)
        assertEquals(1.0, std.sf(1.0), tol)
        assertEquals(0.5, std.sf(2.0), tol)
        assertEquals(0.2, std.sf(5.0), tol)
        assertEquals(0.1, std.sf(10.0), tol)

        val d = ParetoDistribution(3.0, 2.0)
        assertEquals(1.0, d.sf(2.0), tol)
        assertEquals(0.296296296296296, d.sf(3.0), tol)
        assertEquals(0.064, d.sf(5.0), tol)
    }

    // --- Moment undefined thresholds ---

    @Test
    fun testMeanInfiniteWhenShapeLe1() {
        assertEquals(Double.POSITIVE_INFINITY, ParetoDistribution(1.0, 1.0).mean)
        assertEquals(Double.POSITIVE_INFINITY, ParetoDistribution(0.5, 1.0).mean)
    }

    @Test
    fun testMeanFiniteWhenShapeGt1() {
        val d = ParetoDistribution(2.5, 1.0)
        assertEquals(1.66666666666667, d.mean, tol)
    }

    @Test
    fun testVarianceInfiniteWhenShapeLe2() {
        assertEquals(Double.POSITIVE_INFINITY, ParetoDistribution(2.0, 1.0).variance)
        assertEquals(Double.POSITIVE_INFINITY, ParetoDistribution(1.5, 1.0).variance)
        assertEquals(Double.POSITIVE_INFINITY, ParetoDistribution(1.0, 1.0).variance)
    }

    @Test
    fun testSkewnessNaNWhenShapeLe3() {
        assertTrue(ParetoDistribution(3.0, 1.0).skewness.isNaN())
        assertTrue(ParetoDistribution(2.0, 1.0).skewness.isNaN())
    }

    @Test
    fun testKurtosisNaNWhenShapeLe4() {
        assertTrue(ParetoDistribution(4.0, 1.0).kurtosis.isNaN())
        assertTrue(ParetoDistribution(3.0, 1.0).kurtosis.isNaN())
    }

    // --- Extreme parameters ---

    @Test
    fun testExtremeShapeLarge() {
        val d = ParetoDistribution(100.0, 1.0)
        // scipy: pareto(b=100, scale=1)
        assertEquals(100.0, d.pdf(1.0), pdfTol)
        assertEquals(36.6050705276355, d.pdf(1.01), 1e-6)
        assertEquals(0.630288787670881, d.cdf(1.01), 1e-6)
        assertEquals(1.01010101010101, d.mean, tol)
        assertEquals(-3.59517018598809, d.entropy, pdfTol)
    }

    @Test
    fun testExtremeScaleSmall() {
        val d = ParetoDistribution(2.0, 0.001)
        // scipy: pareto(b=2, scale=0.001)
        assertEquals(2000.0, d.pdf(0.001), pdfTol)
        assertEquals(2.0, d.pdf(0.01), pdfTol)
        assertEquals(0.99, d.cdf(0.01), tol)
        assertEquals(0.002, d.mean, tol)
        assertEquals(-6.10090245954208, d.entropy, pdfTol)
    }

    @Test
    fun testPdfFarInTail() {
        val pdfVal = std.pdf(1e6)
        assertTrue(pdfVal > 0.0, "pdf in tail should be positive")
        assertTrue(pdfVal < 1e-10, "pdf in tail should be tiny")
    }

    // --- Non-finite ---

    @Test
    fun testPdfNaN() {
        assertTrue(std.pdf(Double.NaN).isNaN())
    }

    @Test
    fun testCdfNaN() {
        assertTrue(std.cdf(Double.NaN).isNaN())
    }

    @Test
    fun testCdfInfinity() {
        assertEquals(1.0, std.cdf(Double.POSITIVE_INFINITY), 0.0)
        assertEquals(0.0, std.cdf(Double.NEGATIVE_INFINITY), 0.0)
    }

    @Test
    fun testSfInfinity() {
        assertEquals(0.0, std.sf(Double.POSITIVE_INFINITY), 0.0)
        assertEquals(1.0, std.sf(Double.NEGATIVE_INFINITY), 0.0)
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val ps = doubleArrayOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
        for (p in ps) {
            assertEquals(p, std.cdf(std.quantile(p)), tol, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testQuantileCdfRoundTrip() {
        val xs = doubleArrayOf(1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0)
        for (x in xs) {
            assertEquals(x, std.quantile(std.cdf(x)), tol, "quantile(cdf($x)) ≈ $x")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val xs = doubleArrayOf(1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 100.0)
        for (x in xs) {
            assertEquals(1.0, std.sf(x) + std.cdf(x), 1e-14, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val xs = doubleArrayOf(1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
        for (x in xs) {
            assertEquals(ln(std.pdf(x)), std.logPdf(x), pdfTol, "logPdf($x) ≈ ln(pdf($x))")
        }
        val d = ParetoDistribution(3.0, 2.0)
        for (x in doubleArrayOf(2.0, 2.5, 3.0, 5.0, 10.0)) {
            assertEquals(ln(d.pdf(x)), d.logPdf(x), pdfTol)
        }
    }

    @Test
    fun testSampleMean() {
        val d = ParetoDistribution(3.0, 2.0)
        val samples = d.sample(100_000, Random(42))
        val sampleMean = samples.average()
        assertEquals(d.mean, sampleMean, statTol * abs(d.mean).coerceAtLeast(1.0))
    }

    @Test
    fun testSampleVariance() {
        // Use shape=5 for faster convergence (heavier tails converge slowly)
        val d = ParetoDistribution(5.0, 1.0)
        val samples = d.sample(100_000, Random(42))
        val sampleMean = samples.average()
        val sampleVar = samples.sumOf { (it - sampleMean) * (it - sampleMean) } / (samples.size - 1)
        assertEquals(d.variance, sampleVar, statTol * d.variance.coerceAtLeast(1.0))
    }

    @Test
    fun testSampleAboveScale() {
        val d = ParetoDistribution(3.0, 2.0)
        val samples = d.sample(10_000, Random(42))
        for (s in samples) {
            assertTrue(s >= 2.0, "all samples should be ≥ scale")
        }
    }

    @Test
    fun testPdfNonNegative() {
        val xs = doubleArrayOf(0.0, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 100.0)
        for (x in xs) {
            assertTrue(std.pdf(x) >= 0.0, "pdf($x) should be non-negative")
        }
    }

    @Test
    fun testCdfMonotonic() {
        val xs = (0..20).map { 1.0 + it * 0.5 }
        for (i in 1 until xs.size) {
            assertTrue(std.cdf(xs[i]) >= std.cdf(xs[i - 1]), "cdf should be monotonically non-decreasing")
        }
    }

    @Test
    fun testPdfIntegration() {
        // Use shape=5 for compact support range (standard Pareto has very long tail)
        val d = ParetoDistribution(5.0, 1.0)
        val eps = 1e-6
        val lower = d.quantile(eps)
        val upper = d.quantile(1.0 - eps)
        val integral = trapezoidalIntegral({ d.pdf(it) }, lower, upper)
        assertEquals(d.cdf(upper) - d.cdf(lower), integral, 1e-4)
    }

    // --- Validation ---

    @Test
    fun testInvalidShapeZero() {
        assertFailsWith<InvalidParameterException> { ParetoDistribution(0.0, 1.0) }
    }

    @Test
    fun testInvalidShapeNegative() {
        assertFailsWith<InvalidParameterException> { ParetoDistribution(-1.0, 1.0) }
    }

    @Test
    fun testInvalidScaleZero() {
        assertFailsWith<InvalidParameterException> { ParetoDistribution(1.0, 0.0) }
    }

    @Test
    fun testInvalidScaleNegative() {
        assertFailsWith<InvalidParameterException> { ParetoDistribution(1.0, -1.0) }
    }

    @Test
    fun testInvalidQuantileProbability() {
        assertFailsWith<InvalidParameterException> { std.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { std.quantile(1.1) }
    }
}
