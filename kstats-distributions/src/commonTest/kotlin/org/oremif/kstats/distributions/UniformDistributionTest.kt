package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class UniformDistributionTest : ContinuousDistributionPropertyTests() {

    override fun createDistribution(): ContinuousDistribution = UniformDistribution(0.0, 10.0)
    override val testPoints = listOf(-1.0, 0.0, 2.5, 5.0, 7.5, 10.0, 11.0)

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

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { UniformDistribution(5.0, 5.0) }
        assertFailsWith<InvalidParameterException> { UniformDistribution(10.0, 5.0) }
        assertFailsWith<InvalidParameterException> { UniformDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { UniformDistribution(0.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { UniformDistribution(Double.NaN, Double.NaN) }
        assertFailsWith<InvalidParameterException> { UniformDistribution(Double.POSITIVE_INFINITY, 1.0) }
        assertFailsWith<InvalidParameterException> { UniformDistribution(0.0, Double.NEGATIVE_INFINITY) }
        assertFailsWith<InvalidParameterException> {
            UniformDistribution(
                Double.NEGATIVE_INFINITY,
                Double.POSITIVE_INFINITY
            )
        }
    }

    @Test
    fun testQuantileInvalidP() {
        val u = UniformDistribution(0.0, 10.0)
        assertFailsWith<InvalidParameterException> { u.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { u.quantile(1.1) }
        assertFailsWith<InvalidParameterException> { u.quantile(Double.NaN) }
    }

    @Test
    fun testPdfNaN() {
        val u = UniformDistribution(0.0, 10.0)
        assertTrue(u.pdf(Double.NaN).isNaN())
    }

    @Test
    fun testLogPdfNaN() {
        val u = UniformDistribution(0.0, 10.0)
        assertTrue(u.logPdf(Double.NaN).isNaN())
    }

    @Test
    fun testCdfNaNAndInfinity() {
        val u = UniformDistribution(0.0, 10.0)
        assertTrue(u.cdf(Double.NaN).isNaN())
        assertEquals(0.0, u.cdf(Double.NEGATIVE_INFINITY), 0.0)
        assertEquals(1.0, u.cdf(Double.POSITIVE_INFINITY), 0.0)
    }

    @Test
    fun testSfNaNAndInfinity() {
        val u = UniformDistribution(0.0, 10.0)
        assertTrue(u.sf(Double.NaN).isNaN())
        assertEquals(1.0, u.sf(Double.NEGATIVE_INFINITY), 0.0)
        assertEquals(0.0, u.sf(Double.POSITIVE_INFINITY), 0.0)
    }

}
