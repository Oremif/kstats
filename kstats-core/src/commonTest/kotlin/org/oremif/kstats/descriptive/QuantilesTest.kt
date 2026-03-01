package org.oremif.kstats.descriptive

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class QuantilesTest {

    @Test
    fun testPercentileMedian() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(3.0, data.percentile(50.0), 1e-10)
    }

    @Test
    fun testPercentileInterpolation() {
        val data = listOf(1.0, 2.0, 3.0, 4.0)
        // 25th percentile with linear interpolation
        assertEquals(1.75, data.percentile(25.0), 1e-10)
    }

    @Test
    fun testQuantileBounds() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(1.0, data.quantile(0.0), 1e-10)
        assertEquals(5.0, data.quantile(1.0), 1e-10)
    }

    @Test
    fun testQuartiles() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val (q1, q2, q3) = data.quartiles()
        assertEquals(q2, data.median(), 1e-10)
        assertTrue(q1 < q2)
        assertTrue(q2 < q3)
    }

    @Test
    fun testPercentileLower() {
        val data = listOf(1.0, 2.0, 3.0, 4.0)
        assertEquals(1.0, data.percentile(25.0, QuantileInterpolation.LOWER), 1e-10)
    }

    @Test
    fun testPercentileHigher() {
        val data = listOf(1.0, 2.0, 3.0, 4.0)
        assertEquals(2.0, data.percentile(25.0, QuantileInterpolation.HIGHER), 1e-10)
    }
}
