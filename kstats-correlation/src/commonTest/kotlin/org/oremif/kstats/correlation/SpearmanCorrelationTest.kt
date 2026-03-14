package org.oremif.kstats.correlation

import kotlin.test.Test
import kotlin.test.assertEquals

class SpearmanCorrelationTest {

    @Test
    fun testMonotonicallyIncreasing() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(1.0, 4.0, 9.0, 16.0, 25.0) // monotonically increasing
        val result = spearmanCorrelation(x, y)
        assertEquals(1.0, result.coefficient, 1e-10)
    }
}
