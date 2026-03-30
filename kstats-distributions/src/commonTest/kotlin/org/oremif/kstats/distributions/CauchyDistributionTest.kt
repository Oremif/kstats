package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.PI
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class CauchyDistributionTest : ContinuousDistributionPropertyTests() {
    override fun createDistribution() = CauchyDistribution.STANDARD
    override val testPoints = listOf(-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0)
    override val integrationEpsilon = 0.01

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val d = CauchyDistribution.STANDARD
        // scipy: stats.cauchy(0, 1).pdf(x)
        assertEquals(0.318309886183791, d.pdf(0.0), 1e-12)
        assertEquals(0.159154943091895, d.pdf(1.0), 1e-12)
        assertEquals(0.159154943091895, d.pdf(-1.0), 1e-12)
        assertEquals(0.0636619772367581, d.pdf(2.0), 1e-12)
        assertEquals(0.254647908947033, d.pdf(0.5), 1e-12)
        assertEquals(0.00315158303152268, d.pdf(10.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val d = CauchyDistribution.STANDARD
        // scipy: stats.cauchy(0, 1).cdf(x)
        assertEquals(0.5, d.cdf(0.0), 1e-12)
        assertEquals(0.75, d.cdf(1.0), 1e-12)
        assertEquals(0.25, d.cdf(-1.0), 1e-12)
        assertEquals(0.852416382349567, d.cdf(2.0), 1e-10)
        assertEquals(0.647583617650433, d.cdf(0.5), 1e-10)
        assertEquals(0.968274482569446, d.cdf(10.0), 1e-10)
        assertEquals(0.0317255174305536, d.cdf(-10.0), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val d = CauchyDistribution.STANDARD
        // scipy: stats.cauchy(0, 1).sf(x)
        assertEquals(0.5, d.sf(0.0), 1e-12)
        assertEquals(0.25, d.sf(1.0), 1e-12)
        assertEquals(0.75, d.sf(-1.0), 1e-12)
        assertEquals(0.147583617650433, d.sf(2.0), 1e-10)
        assertEquals(0.352416382349567, d.sf(0.5), 1e-10)
        assertEquals(0.0317255174305536, d.sf(10.0), 1e-10)
    }

    @Test
    fun testSfUpperTail() {
        val d = CauchyDistribution.STANDARD
        // scipy: stats.cauchy(0, 1).sf(x) — far tail
        assertEquals(0.00318299276490819, d.sf(100.0), 1e-10)
        assertEquals(0.000318309780080517, d.sf(1000.0), 1e-12)
        assertEquals(3.18309886171431e-07, d.sf(1e6), 1e-13)
    }

    @Test
    fun testLogPdfKnownValues() {
        val d = CauchyDistribution.STANDARD
        // scipy: stats.cauchy(0, 1).logpdf(x)
        assertEquals(-1.1447298858494, d.logPdf(0.0), 1e-10)
        assertEquals(-1.83787706640935, d.logPdf(1.0), 1e-10)
        assertEquals(-5.75985040269066, d.logPdf(10.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val d = CauchyDistribution.STANDARD
        // scipy: stats.cauchy(0, 1).ppf(p)
        assertEquals(-31.8205159537739, d.quantile(0.01), 1e-6)
        assertEquals(-6.31375151467504, d.quantile(0.05), 1e-6)
        assertEquals(-3.07768353717525, d.quantile(0.1), 1e-6)
        assertEquals(-1.0, d.quantile(0.25), 1e-10)
        assertEquals(0.0, d.quantile(0.5), 1e-12)
        assertEquals(1.0, d.quantile(0.75), 1e-10)
        assertEquals(3.07768353717525, d.quantile(0.9), 1e-6)
        assertEquals(6.31375151467504, d.quantile(0.95), 1e-6)
        assertEquals(31.8205159537741, d.quantile(0.99), 1e-4)
    }

    @Test
    fun testMoments() {
        val d = CauchyDistribution.STANDARD
        assertTrue(d.mean.isNaN())
        assertTrue(d.variance.isNaN())
        assertTrue(d.standardDeviation.isNaN())
        assertTrue(d.skewness.isNaN())
        assertTrue(d.kurtosis.isNaN())
    }

    @Test
    fun testEntropy() {
        // scipy: stats.cauchy(0, 1).entropy() = 2.53102424696929
        assertEquals(2.53102424696929, CauchyDistribution.STANDARD.entropy, 1e-10)
        // scipy: stats.cauchy(3, 2).entropy() = 3.22417142752924
        assertEquals(3.22417142752924, CauchyDistribution(3.0, 2.0).entropy, 1e-10)
    }

    // --- Custom parameters ---

    @Test
    fun testCustomLocationScale() {
        val d = CauchyDistribution(3.0, 2.0)
        // scipy: stats.cauchy(3, 2)
        assertEquals(0.5, d.cdf(3.0), 1e-12)
        assertEquals(0.75, d.cdf(5.0), 1e-12)
        assertEquals(0.5, d.sf(3.0), 1e-12)
        assertEquals(0.25, d.sf(5.0), 1e-12)
        assertEquals(0.159154943091895, d.pdf(3.0), 1e-12)
        assertEquals(1.0, d.quantile(0.25), 1e-10)
        assertEquals(5.0, d.quantile(0.75), 1e-10)
    }

    // --- Edge cases ---

    @Test
    fun testQuantileBoundaries() {
        val d = CauchyDistribution.STANDARD
        assertEquals(Double.NEGATIVE_INFINITY, d.quantile(0.0))
        assertEquals(Double.POSITIVE_INFINITY, d.quantile(1.0))
    }

    @Test
    fun testCdfSymmetry() {
        val d = CauchyDistribution.STANDARD
        for (x in listOf(0.1, 0.5, 1.0, 2.0, 10.0)) {
            assertEquals(1.0, d.cdf(x) + d.cdf(-x), 1e-12, "cdf($x) + cdf(-$x) ≈ 1")
        }
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { CauchyDistribution(0.0, 0.0) }
        assertFailsWith<InvalidParameterException> { CauchyDistribution(0.0, -1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = CauchyDistribution.STANDARD
        assertFailsWith<InvalidParameterException> { d.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testSampleMedian() {
        val d = CauchyDistribution(3.0, 2.0) // median = location = 3
        val rng = kotlin.random.Random(42)
        val samples = d.sample(100_000, rng).sorted()
        val sampleMedian = samples[50000]
        assertEquals(3.0, sampleMedian, 1.0, "sample median ≈ 3")
    }

    @Test
    fun testPdfSymmetry() {
        val d = CauchyDistribution.STANDARD
        for (x in listOf(0.5, 1.0, 2.0, 10.0)) {
            assertEquals(d.pdf(x), d.pdf(-x), 1e-12, "pdf($x) = pdf(-$x)")
        }
    }

    @Test
    fun testExtremeParameters() {
        // scale=1e-10: extremely peaked
        val d1 = CauchyDistribution(0.0, 1e-10)
        // scipy: pdf(0) = 1/(π * 1e-10) ≈ 3.18310e9
        assertEquals(1.0 / (PI * 1e-10), d1.pdf(0.0), 1.0)
        // scipy: cdf(1e-10) = 0.75
        assertEquals(0.75, d1.cdf(1e-10), 1e-10)

        // scale=1e10: very flat
        val d2 = CauchyDistribution(0.0, 1e10)
        // scipy: pdf(0) = 1/(π * 1e10) ≈ 3.18310e-11
        assertEquals(1.0 / (PI * 1e10), d2.pdf(0.0), 1e-20)

        // location=1e15
        val d3 = CauchyDistribution(1e15, 1.0)
        // scipy: cdf(1e15) = 0.5
        assertEquals(0.5, d3.cdf(1e15), 1e-10)
    }

}
