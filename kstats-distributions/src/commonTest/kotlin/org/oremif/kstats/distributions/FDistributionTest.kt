package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val f = FDistribution(5.0, 10.0)
        // scipy: stats.f(5, 10).pdf(x)
        assertEquals(0.227400093561907, f.pdf(0.1), 1e-10)
        assertEquals(0.687607002770625, f.pdf(0.5), 1e-10)
        assertEquals(0.49547978348664, f.pdf(1.0), 1e-10)
        assertEquals(0.162005742180115, f.pdf(2.0), 1e-10)
        assertEquals(0.0558267311455723, f.pdf(3.0), 1e-10)
        assertEquals(0.000478163040278585, f.pdf(10.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val f = FDistribution(5.0, 10.0)
        // scipy: stats.f(5, 10).cdf(x)
        assertEquals(0.0101150894697428, f.cdf(0.1), 1e-10)
        assertEquals(0.229975119349898, f.cdf(0.5), 1e-10)
        assertEquals(0.5348805734622, f.cdf(1.0), 1e-10)
        assertEquals(0.835805049100261, f.cdf(2.0), 1e-10)
        assertEquals(0.934442437906156, f.cdf(3.0), 1e-10)
        assertEquals(0.9987942193513, f.cdf(10.0), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val f = FDistribution(5.0, 10.0)
        // scipy: stats.f(5, 10).sf(x)
        assertEquals(0.989884910530257, f.sf(0.1), 1e-10)
        assertEquals(0.770024880650102, f.sf(0.5), 1e-10)
        assertEquals(0.4651194265378, f.sf(1.0), 1e-10)
        assertEquals(0.164194950899739, f.sf(2.0), 1e-10)
        assertEquals(0.0655575620938441, f.sf(3.0), 1e-10)
        assertEquals(0.00120578064869954, f.sf(10.0), 1e-10)
    }

    @Test
    fun testSfUpperTail() {
        val f = FDistribution(10.0, 20.0)
        // scipy: stats.f(10, 20).sf(x)
        assertEquals(0.0175095414784, f.sf(3.0), 1e-8)
        assertEquals(0.00109658938743534, f.sf(5.0), 1e-10)
        assertEquals(8.59411961792848e-06, f.sf(10.0), 1e-10)
    }

    @Test
    fun testLogPdfKnownValues() {
        val f = FDistribution(5.0, 10.0)
        assertEquals(ln(0.49547978348664), f.logPdf(1.0), 1e-10)
        assertEquals(ln(0.162005742180115), f.logPdf(2.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val f = FDistribution(5.0, 10.0)
        // scipy: stats.f(5, 10).ppf(p)
        // Note: small p values may not converge well — see DIST-010
        assertEquals(0.529141685567822, f.quantile(0.25), 1e-6)
        assertEquals(0.931933160851048, f.quantile(0.5), 1e-6)
        assertEquals(1.58532325938462, f.quantile(0.75), 1e-5)
        assertEquals(2.52164068620962, f.quantile(0.9), 1e-5)
        assertEquals(3.32583453041301, f.quantile(0.95), 1e-4)
    }

    @Test
    fun testMoments() {
        val f = FDistribution(5.0, 10.0)
        assertEquals(1.25, f.mean, 1e-12)
        assertEquals(1.35416666666667, f.variance, 1e-10)

        val f2 = FDistribution(10.0, 20.0)
        assertEquals(1.11111111111111, f2.mean, 1e-10)
        assertEquals(0.432098765432099, f2.variance, 1e-10)
    }

    // --- Edge cases ---

    @Test
    fun testPdfAtZero() {
        // d1=2: pdf(0) = 1.0
        assertEquals(1.0, FDistribution(2.0, 5.0).pdf(0.0), 1e-12)
        // d1<2: pdf(0) = +Inf
        assertEquals(Double.POSITIVE_INFINITY, FDistribution(1.0, 5.0).pdf(0.0))
        // d1>2: pdf(0) = 0
        assertEquals(0.0, FDistribution(5.0, 10.0).pdf(0.0), 1e-12)
    }

    @Test
    fun testNegativeX() {
        val f = FDistribution(5.0, 10.0)
        assertEquals(0.0, f.pdf(-1.0), 1e-12)
        assertEquals(0.0, f.cdf(-1.0), 1e-12)
        assertEquals(1.0, f.sf(-1.0), 1e-12)
    }

    @Test
    fun testQuantileBoundaries() {
        val f = FDistribution(5.0, 10.0)
        assertEquals(0.0, f.quantile(0.0), 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, f.quantile(1.0))
    }

    @Test
    fun testMomentsEdgeDf() {
        // d2<=2: mean=NaN
        assertTrue(FDistribution(5.0, 2.0).mean.isNaN())
        // d2<=4: variance=NaN
        assertTrue(FDistribution(5.0, 4.0).variance.isNaN())
        // d2<=6: skewness=NaN
        assertTrue(FDistribution(5.0, 6.0).skewness.isNaN())
        // d2<=8: kurtosis=NaN
        assertTrue(FDistribution(5.0, 8.0).kurtosis.isNaN())
    }

    // --- Different params ---

    @Test
    fun testF_2_5() {
        val f = FDistribution(2.0, 5.0)
        // scipy: stats.f(2, 5)
        assertEquals(0.366061854739391, f.cdf(0.5), 1e-10)
        assertEquals(0.633938145260609, f.sf(0.5), 1e-10)
        assertEquals(1.66666666666667, f.mean, 1e-10)
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { FDistribution(0.0, 10.0) }
        assertFailsWith<InvalidParameterException> { FDistribution(-1.0, 10.0) }
        assertFailsWith<InvalidParameterException> { FDistribution(5.0, 0.0) }
        assertFailsWith<InvalidParameterException> { FDistribution(5.0, -1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val f = FDistribution(5.0, 10.0)
        assertFailsWith<InvalidParameterException> { f.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { f.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val f = FDistribution(5.0, 10.0)
        // Note: small p values may not converge well — see DIST-010
        for (p in listOf(0.25, 0.5, 0.75, 0.9)) {
            assertEquals(p, f.cdf(f.quantile(p)), 1e-6, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val f = FDistribution(5.0, 10.0)
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 5.0, 10.0)) {
            assertEquals(1.0, f.sf(x) + f.cdf(x), 1e-10, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val f = FDistribution(5.0, 10.0)
        for (x in listOf(0.5, 1.0, 2.0, 5.0)) {
            assertEquals(f.pdf(x), exp(f.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleStats() {
        val f = FDistribution(5.0, 10.0) // mean=1.25
        val rng = kotlin.random.Random(42)
        val samples = f.sample(100_000, rng)
        val sampleMean = samples.average()
        assertEquals(1.25, sampleMean, 0.15, "sample mean ≈ 1.25")
        val sampleVar = samples.sumOf { (it - sampleMean) * (it - sampleMean) } / (samples.size - 1)
        assertEquals(f.variance, sampleVar, maxOf(f.variance * 0.1, 0.05), "sample variance ≈ ${f.variance}")
    }

    @Test
    fun testCdfMonotonicity() {
        val f = FDistribution(5.0, 10.0)
        var prev = 0.0
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)) {
            val cdfVal = f.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testEntropyNaN() {
        // entropy requires digamma (deferred to MATH-001)
        assertTrue(FDistribution(5.0, 10.0).entropy.isNaN())
    }

    @Test
    fun testExtremeParameters() {
        // df1=df2=1000: concentrated around 1
        val d1 = FDistribution(1000.0, 1000.0)
        // scipy: mean = 1000/998 ≈ 1.002004
        assertEquals(1000.0 / 998.0, d1.mean, 1e-6)
        // scipy: cdf(1.0) ≈ 0.5
        assertEquals(0.5, d1.cdf(1.0), 1e-3)

        // df1=df2=0.5: heavy tails
        val d2 = FDistribution(0.5, 0.5)
        // scipy: cdf(1.0) = 0.5 by symmetry
        assertEquals(0.5, d2.cdf(1.0), 1e-6)
    }

    @Test
    fun testPdfIntegration() {
        val d = FDistribution(5.0, 10.0)
        val eps = 1e-6
        val lower = d.quantile(eps)
        val upper = d.quantile(1.0 - eps)
        val integral = trapezoidalIntegral({ d.pdf(it) }, lower, upper)
        assertEquals(d.cdf(upper) - d.cdf(lower), integral, 1e-4)
    }
}
