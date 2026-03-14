package org.oremif.kstats.distributions

import kotlin.math.exp
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ExponentialDistributionTest {
    private val tol = 1e-10

    @Test
    fun testCdf() {
        val e = ExponentialDistribution(2.0)
        // 1 - e^(-2*1) = 1 - e^(-2)
        assertEquals(1.0 - exp(-2.0), e.cdf(1.0), tol)
    }

    @Test
    fun testQuantile() {
        val e = ExponentialDistribution(1.0)
        assertEquals(ln(2.0), e.quantile(0.5), tol) // median
    }

    @Test
    fun testMean() {
        val e = ExponentialDistribution(0.5)
        assertEquals(2.0, e.mean, tol)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.expon(scale=1).entropy() = 1.0
        val e1 = ExponentialDistribution(1.0)
        assertEquals(1.0, e1.entropy, 1e-12)
        // scipy: stats.expon(scale=2).entropy() = 1.693147180559945
        val e2 = ExponentialDistribution(0.5)
        assertEquals(1.693147180559945, e2.entropy, 1e-12)
    }

    @Test
    fun testQuantilePrecisionNearOne() {
        val e = ExponentialDistribution(1.0)
        // scipy: stats.expon.ppf(1 - 1e-15) = 34.539575992340879
        assertEquals(34.539575992340879, e.quantile(1.0 - 1e-15), 1e-6)
        // scipy: stats.expon.ppf(1 - 1e-10) = 23.025850847200090
        assertEquals(23.025850847200090, e.quantile(1.0 - 1e-10), 1e-6)
    }

    @Test
    fun testLogPdfConsistency() {
        val e = ExponentialDistribution(2.0)
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 5.0)) {
            assertEquals(e.pdf(x), exp(e.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSfConsistency() {
        val e = ExponentialDistribution(2.0)
        for (x in listOf(-1.0, 0.0, 0.5, 1.0, 5.0)) {
            assertEquals(1.0, e.sf(x) + e.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testExtremeParameters() {
        // rate=1e10: fast decay
        val d1 = ExponentialDistribution(1e10)
        assertEquals(1e-10, d1.mean, 1e-20)
        // scipy: cdf(1e-9) ≈ 0.999955
        assertEquals(0.999954600070238, d1.cdf(1e-9), 1e-6)

        // rate=1e-10: slow decay
        val d2 = ExponentialDistribution(1e-10)
        // scipy: sf(1e10) = exp(-1) ≈ 0.367879
        assertEquals(0.367879441171442, d2.sf(1e10), 1e-6)

        // Deep tail: sf(40) for Exp(1) = exp(-40) ≈ 4.248e-18
        val d3 = ExponentialDistribution(1.0)
        assertEquals(exp(-40.0), d3.sf(40.0), 1e-28)
    }

    @Test
    fun testCdfQuantileRoundTrip() {
        val d = ExponentialDistribution(2.0)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            assertEquals(p, d.cdf(d.quantile(p)), 1e-10, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testCdfMonotonicity() {
        val d = ExponentialDistribution(2.0)
        var prev = 0.0
        for (x in (0..10).map { it * 0.5 }) {
            val cdfVal = d.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testSampleStats() {
        val d = ExponentialDistribution(2.0)
        val rng = kotlin.random.Random(42)
        val samples = d.sample(100_000, rng)
        val sampleMean = samples.average()
        assertEquals(d.mean, sampleMean, 0.05, "sample mean ≈ ${d.mean}")
        val sampleVar = samples.sumOf { (it - sampleMean) * (it - sampleMean) } / (samples.size - 1)
        assertEquals(d.variance, sampleVar, maxOf(d.variance * 0.1, 0.05), "sample variance ≈ ${d.variance}")
    }

    @Test
    fun testPdfIntegration() {
        val d = ExponentialDistribution(2.0)
        val eps = 1e-6
        val lower = d.quantile(eps)
        val upper = d.quantile(1.0 - eps)
        val integral = trapezoidalIntegral({ d.pdf(it) }, lower, upper)
        assertEquals(d.cdf(upper) - d.cdf(lower), integral, 1e-4)
    }
}
