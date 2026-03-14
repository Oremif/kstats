package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class StudentTDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val t = StudentTDistribution(5.0)
        // scipy: stats.t(5).pdf(x)
        assertEquals(0.379606689822494, t.pdf(0.0), 1e-12)
        assertEquals(0.219679797350981, t.pdf(1.0), 1e-12)
        assertEquals(0.219679797350981, t.pdf(-1.0), 1e-12)
        assertEquals(0.0650903103262165, t.pdf(2.0), 1e-12)
        assertEquals(0.0172925788002229, t.pdf(3.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val t = StudentTDistribution(5.0)
        // scipy: stats.t(5).cdf(x)
        assertEquals(0.5, t.cdf(0.0), 1e-12)
        assertEquals(0.818391266175439, t.cdf(1.0), 1e-10)
        assertEquals(0.181608733824561, t.cdf(-1.0), 1e-10)
        assertEquals(0.949030260585071, t.cdf(2.0), 1e-10)
        assertEquals(0.0509697394149292, t.cdf(-2.0), 1e-10)
        assertEquals(0.984950376051269, t.cdf(3.0), 1e-10)
        assertEquals(0.0150496239487313, t.cdf(-3.0), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val t = StudentTDistribution(5.0)
        // scipy: stats.t(5).sf(x)
        assertEquals(0.5, t.sf(0.0), 1e-12)
        assertEquals(0.181608733824561, t.sf(1.0), 1e-10)
        assertEquals(0.0509697394149292, t.sf(2.0), 1e-10)
        assertEquals(0.0150496239487313, t.sf(3.0), 1e-10)
        assertEquals(0.818391266175439, t.sf(-1.0), 1e-10)
    }

    @Test
    fun testSfUpperTail() {
        val t = StudentTDistribution(5.0)
        // scipy: stats.t(5).sf(x) — far tail
        assertEquals(8.54737878714818e-05, t.sf(10.0), 1e-9)
        assertEquals(9.48000711231183e-10, t.sf(100.0), 1e-14)
    }

    @Test
    fun testLogPdfKnownValues() {
        val t = StudentTDistribution(10.0)
        // scipy: stats.t(10).pdf(x) → logpdf
        assertEquals(ln(0.389108383966031), t.logPdf(0.0), 1e-10)
        assertEquals(ln(0.230361989229139), t.logPdf(1.0), 1e-10)
        assertEquals(ln(0.0611457663212182), t.logPdf(2.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val t = StudentTDistribution(10.0)
        // scipy: stats.t(10).ppf(p)
        assertEquals(-2.76376946298153, t.quantile(0.01), 1e-4)
        assertEquals(-2.22813885198627, t.quantile(0.025), 1e-4)
        assertEquals(-1.81246112281168, t.quantile(0.05), 1e-4)
        assertEquals(-1.37218364282088, t.quantile(0.1), 1e-4)
        assertEquals(0.0, t.quantile(0.5), 1e-12)
        assertEquals(1.37218364282088, t.quantile(0.9), 1e-4)
        assertEquals(2.22813885198627, t.quantile(0.975), 1e-4)
    }

    @Test
    fun testMoments() {
        val t5 = StudentTDistribution(5.0)
        assertEquals(0.0, t5.mean, 1e-12)
        assertEquals(5.0 / 3.0, t5.variance, 1e-12)
        assertEquals(0.0, t5.skewness, 1e-12)

        val t10 = StudentTDistribution(10.0)
        assertEquals(0.0, t10.mean, 1e-12)
        assertEquals(10.0 / 8.0, t10.variance, 1e-12)
        assertEquals(6.0 / 6.0, t10.kurtosis, 1e-12) // 6/(df-4)
    }

    // --- Edge cases ---

    @Test
    fun testDf1IsCauchy() {
        // StudentT(df=1) = Cauchy(0,1)
        val t = StudentTDistribution(1.0)
        val c = CauchyDistribution.STANDARD
        for (x in listOf(-3.0, -1.0, 0.0, 1.0, 3.0)) {
            assertEquals(c.pdf(x), t.pdf(x), 1e-10, "T(1).pdf($x) = Cauchy.pdf($x)")
            assertEquals(c.cdf(x), t.cdf(x), 1e-10, "T(1).cdf($x) = Cauchy.cdf($x)")
        }
    }

    @Test
    fun testMomentsEdgeDf() {
        // df=1: mean=NaN, var=NaN
        assertTrue(StudentTDistribution(1.0).mean.isNaN())
        assertTrue(StudentTDistribution(1.0).variance.isNaN())
        // df=2: mean=0, var=Inf
        assertEquals(0.0, StudentTDistribution(2.0).mean, 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, StudentTDistribution(2.0).variance)
        // df=3: skewness=NaN (requires df>3), kurtosis=+Inf (2<df<=4)
        assertTrue(StudentTDistribution(3.0).skewness.isNaN())
        assertEquals(Double.POSITIVE_INFINITY, StudentTDistribution(3.0).kurtosis)
        // df=5: kurtosis = 6/(5-4) = 6
        assertEquals(6.0, StudentTDistribution(5.0).kurtosis, 1e-12)
    }

    @Test
    fun testQuantileBoundaries() {
        val t = StudentTDistribution(5.0)
        assertEquals(Double.NEGATIVE_INFINITY, t.quantile(0.0))
        assertEquals(Double.POSITIVE_INFINITY, t.quantile(1.0))
    }

    @Test
    fun testCdfSymmetry() {
        val t = StudentTDistribution(10.0)
        for (x in listOf(0.5, 1.0, 2.0, 3.0)) {
            assertEquals(1.0, t.cdf(x) + t.cdf(-x), 1e-12, "cdf($x) + cdf(-$x) ≈ 1")
        }
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { StudentTDistribution(0.0) }
        assertFailsWith<InvalidParameterException> { StudentTDistribution(-1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val t = StudentTDistribution(5.0)
        assertFailsWith<InvalidParameterException> { t.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { t.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val t = StudentTDistribution(10.0)
        for (p in listOf(0.1, 0.25, 0.5, 0.75, 0.9)) {
            assertEquals(p, t.cdf(t.quantile(p)), 1e-8, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val t = StudentTDistribution(5.0)
        for (x in listOf(-3.0, -1.0, 0.0, 1.0, 3.0, 10.0)) {
            assertEquals(1.0, t.sf(x) + t.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val t = StudentTDistribution(5.0)
        for (x in listOf(-3.0, -1.0, 0.0, 1.0, 3.0)) {
            assertEquals(t.pdf(x), exp(t.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleStats() {
        val t = StudentTDistribution(30.0) // mean=0, var=30/28≈1.071
        val rng = kotlin.random.Random(42)
        val samples = t.sample(100_000, rng)
        val sampleMean = samples.average()
        assertEquals(0.0, sampleMean, 0.1, "sample mean ≈ 0")
        val sampleVar = samples.sumOf { (it - sampleMean) * (it - sampleMean) } / (samples.size - 1)
        assertEquals(t.variance, sampleVar, maxOf(t.variance * 0.1, 0.05), "sample variance ≈ ${t.variance}")
    }

    @Test
    fun testCdfMonotonicity() {
        val t = StudentTDistribution(10.0)
        var prev = 0.0
        for (x in listOf(-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0)) {
            val cdfVal = t.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testEntropy() {
        assertEquals(2.531024246969291, StudentTDistribution(1.0).entropy, 1e-10) // ln(4*pi), equals Cauchy
        assertEquals(1.960279229160082, StudentTDistribution(2.0).entropy, 1e-10)
        assertEquals(1.627502672414396, StudentTDistribution(5.0).entropy, 1e-10)
        assertEquals(1.521262492975681, StudentTDistribution(10.0).entropy, 1e-10)
        assertEquals(1.452543329787207, StudentTDistribution(30.0).entropy, 1e-10)
        assertEquals(3.666727416103854, StudentTDistribution(0.5).entropy, 1e-10)
    }

    @Test
    fun testExtremeParameters() {
        // df=1000: approaches Normal(0,1)
        val d1 = StudentTDistribution(1000.0)
        // scipy: pdf(0) ≈ 0.398843 (close to Normal's 0.398942)
        assertEquals(0.398842557313707, d1.pdf(0.0), 1e-4)
        // scipy: cdf(3) ≈ 0.998617
        assertEquals(0.998616645477881, d1.cdf(3.0), 1e-4)

        // df=0.5: heavy tails
        val d2 = StudentTDistribution(0.5)
        // scipy: pdf(0) = 0.269676
        assertEquals(0.269676300594190, d2.pdf(0.0), 1e-6)
        // scipy: cdf(100) = 0.967930
        assertEquals(0.967930142978463, d2.cdf(100.0), 1e-4)
        // mean is NaN for df <= 1
        assertTrue(d2.mean.isNaN())
    }

    @Test
    fun testPdfIntegration() {
        val d = StudentTDistribution(5.0)
        val eps = 1e-6
        val lower = d.quantile(eps)
        val upper = d.quantile(1.0 - eps)
        val integral = trapezoidalIntegral({ d.pdf(it) }, lower, upper)
        assertEquals(d.cdf(upper) - d.cdf(lower), integral, 1e-4)
    }
}
