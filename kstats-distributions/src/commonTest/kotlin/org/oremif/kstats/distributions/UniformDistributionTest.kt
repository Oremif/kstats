package org.oremif.kstats.distributions

import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class UniformDistributionTest {
    private val tol = 1e-10

    @Test
    fun testPdf() {
        val u = UniformDistribution(0.0, 10.0)
        assertEquals(0.1, u.pdf(5.0), tol)
        assertEquals(0.0, u.pdf(-1.0), tol)
    }

    @Test
    fun testCdf() {
        val u = UniformDistribution(0.0, 10.0)
        assertEquals(0.5, u.cdf(5.0), tol)
    }

    @Test
    fun testQuantile() {
        val u = UniformDistribution(2.0, 8.0)
        assertEquals(5.0, u.quantile(0.5), tol)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.uniform(0, 10).entropy() = 2.302585092994046
        val u1 = UniformDistribution(0.0, 10.0)
        assertEquals(2.302585092994046, u1.entropy, 1e-12)
        // scipy: stats.uniform(2, 6).entropy() = 1.791759469228055
        val u2 = UniformDistribution(2.0, 8.0)
        assertEquals(1.791759469228055, u2.entropy, 1e-12)
    }

    @Test
    fun testLogPdfConsistency() {
        val u = UniformDistribution(0.0, 10.0)
        for (x in listOf(0.0, 2.5, 5.0, 7.5, 10.0)) {
            assertEquals(u.pdf(x), exp(u.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSfConsistency() {
        val u = UniformDistribution(0.0, 10.0)
        for (x in listOf(-1.0, 0.0, 5.0, 10.0, 11.0)) {
            assertEquals(1.0, u.sf(x) + u.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testExtremeParameters() {
        // Wide range: [-1e15, 1e15]
        val d1 = UniformDistribution(-1e15, 1e15)
        assertEquals(5e-16, d1.pdf(0.0), 1e-25)
        assertEquals(0.5, d1.cdf(0.0), 1e-10)

        // Narrow range: [0, 1e-15]
        val d2 = UniformDistribution(0.0, 1e-15)
        assertEquals(1e15, d2.pdf(5e-16), 1e5)

        // Large offset: [1e15, 1e15+1]
        val d3 = UniformDistribution(1e15, 1e15 + 1.0)
        assertEquals(1e15 + 0.5, d3.mean, 1.0)
        assertEquals(0.5, d3.cdf(1e15 + 0.5), 1e-10)
    }

    @Test
    fun testCdfQuantileRoundTrip() {
        val d = UniformDistribution(0.0, 10.0)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            assertEquals(p, d.cdf(d.quantile(p)), 1e-10, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testCdfMonotonicity() {
        val d = UniformDistribution(0.0, 10.0)
        var prev = 0.0
        for (x in (-1..11).map { it.toDouble() }) {
            val cdfVal = d.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testSampleStats() {
        val d = UniformDistribution(0.0, 10.0)
        val rng = kotlin.random.Random(42)
        val samples = d.sample(100_000, rng)
        val sampleMean = samples.average()
        assertEquals(d.mean, sampleMean, 0.05, "sample mean ≈ ${d.mean}")
        val sampleVar = samples.sumOf { (it - sampleMean) * (it - sampleMean) } / (samples.size - 1)
        assertEquals(d.variance, sampleVar, maxOf(d.variance * 0.1, 0.05), "sample variance ≈ ${d.variance}")
    }

    @Test
    fun testPdfIntegration() {
        val d = UniformDistribution(0.0, 10.0)
        val eps = 1e-6
        val lower = d.quantile(eps)
        val upper = d.quantile(1.0 - eps)
        val integral = trapezoidalIntegral({ d.pdf(it) }, lower, upper)
        assertEquals(d.cdf(upper) - d.cdf(lower), integral, 1e-4)
    }
}
