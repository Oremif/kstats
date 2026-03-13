package org.oremif.kstats.distributions

import kotlin.test.Test
import kotlin.test.assertEquals

class DistributionTest {

    private val tol = 1e-15

    @Test
    fun standardDeviationDefaultSqrtVariance() {
        val d = StubDistribution(variance = 4.0)
        assertEquals(2.0, d.standardDeviation, tol)
    }

    @Test
    fun standardDeviationZeroVariance() {
        val d = StubDistribution(variance = 0.0)
        assertEquals(0.0, d.standardDeviation, tol)
    }

    @Test
    fun standardDeviationLargeVariance() {
        val d = StubDistribution(variance = 1e12)
        assertEquals(1e6, d.standardDeviation, tol)
    }

    @Test
    fun sfDefaultOneMinusCdf() {
        val d = StubDistribution(cdfValue = 0.3)
        assertEquals(0.7, d.sf(0.0), tol)
    }

    @Test
    fun sfBoundaryCdfZero() {
        val d = StubDistribution(cdfValue = 0.0)
        assertEquals(1.0, d.sf(0.0), tol)
    }

    @Test
    fun sfBoundaryCdfOne() {
        val d = StubDistribution(cdfValue = 1.0)
        assertEquals(0.0, d.sf(0.0), tol)
    }
}
