package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.PI
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class NormalDistributionTest : ContinuousDistributionPropertyTests() {
    override fun createDistribution() = NormalDistribution(5.0, 2.0)
    override val testPoints = listOf(-1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0)

    private val std = NormalDistribution.STANDARD
    private val tol = 1e-6

    @Test
    fun testPdfAtZero() {
        assertEquals(1.0 / sqrt(2.0 * PI), std.pdf(0.0), tol)
    }

    @Test
    fun testCdfAtZero() {
        assertEquals(0.5, std.cdf(0.0), tol)
    }

    @Test
    fun testCdfKnownValues() {
        // R: pnorm(1) = 0.8413447
        assertEquals(0.8413447, std.cdf(1.0), tol)
        // R: pnorm(-1) = 0.1586553
        assertEquals(0.1586553, std.cdf(-1.0), tol)
    }

    @Test
    fun testQuantile() {
        assertEquals(0.0, std.quantile(0.5), tol)
        // R: qnorm(0.975) = 1.959964
        assertEquals(1.959964, std.quantile(0.975), tol)
    }

    @Test
    fun testMoments() {
        assertEquals(0.0, std.mean, tol)
        assertEquals(1.0, std.variance, tol)
        assertEquals(0.0, std.skewness, tol)
        assertEquals(0.0, std.kurtosis, tol)
    }

    @Test
    fun testCustomMuSigma() {
        val d = NormalDistribution(5.0, 2.0)
        assertEquals(5.0, d.mean, tol)
        assertEquals(4.0, d.variance, tol)
        assertEquals(2.0, d.standardDeviation, tol)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.norm(0, 1).entropy() = 1.418938533204673
        assertEquals(1.418938533204673, std.entropy, 1e-12)
        // scipy: stats.norm(5, 2).entropy() = 2.112085713764618
        val d = NormalDistribution(5.0, 2.0)
        assertEquals(2.112085713764618, d.entropy, 1e-12)
    }

    @Test
    fun testExtremeParameters() {
        // σ=1e6: very flat pdf
        val d1 = NormalDistribution(0.0, 1e6)
        // scipy: pdf(0) ≈ 3.98942e-7
        assertEquals(3.98942280401433e-7, d1.pdf(0.0), 1e-13)
        // scipy: cdf(1e6) = 0.841344746068543
        assertEquals(0.841344746068543, d1.cdf(1e6), 1e-6)

        // μ=1e8: shifted far from origin
        val d2 = NormalDistribution(1e8, 1.0)
        assertEquals(0.5, d2.cdf(1e8), 1e-10)

        // Deep tail: sf(8) for standard normal
        // scipy: sf(8) ≈ 6.22096e-16
        assertEquals(6.22096057427174e-16, std.sf(8.0), 1e-25)
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { NormalDistribution(0.0, 0.0) }
        assertFailsWith<InvalidParameterException> { NormalDistribution(0.0, -1.0) }
        assertFailsWith<InvalidParameterException> { NormalDistribution(0.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { NormalDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { NormalDistribution(Double.NaN, Double.NaN) }
        assertFailsWith<InvalidParameterException> { NormalDistribution(Double.POSITIVE_INFINITY, 1.0) }
        assertFailsWith<InvalidParameterException> { NormalDistribution(Double.NEGATIVE_INFINITY, 1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        assertFailsWith<InvalidParameterException> { std.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { std.quantile(1.1) }
        assertFailsWith<InvalidParameterException> { std.quantile(Double.NaN) }
    }

    @Test
    fun testQuantileBoundary() {
        assertEquals(Double.NEGATIVE_INFINITY, std.quantile(0.0))
        assertEquals(Double.POSITIVE_INFINITY, std.quantile(1.0))
    }

    @Test
    fun testCdfAtInfinity() {
        assertEquals(0.0, std.cdf(Double.NEGATIVE_INFINITY), 0.0)
        assertEquals(1.0, std.cdf(Double.POSITIVE_INFINITY), 0.0)
        assertTrue(std.cdf(Double.NaN).isNaN())
    }

    @Test
    fun testPdfAtInfinity() {
        assertEquals(0.0, std.pdf(Double.POSITIVE_INFINITY), 0.0)
        assertEquals(0.0, std.pdf(Double.NEGATIVE_INFINITY), 0.0)
        assertTrue(std.pdf(Double.NaN).isNaN())
    }

    @Test
    fun testSfAtInfinity() {
        assertEquals(1.0, std.sf(Double.NEGATIVE_INFINITY), 0.0)
        assertEquals(0.0, std.sf(Double.POSITIVE_INFINITY), 0.0)
        assertTrue(std.sf(Double.NaN).isNaN())
    }
}
