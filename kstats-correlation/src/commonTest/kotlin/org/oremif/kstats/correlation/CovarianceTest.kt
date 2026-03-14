package org.oremif.kstats.correlation

import kotlin.test.Test
import kotlin.test.assertEquals

class CovarianceTest {

    @Test
    fun testCovarianceSample() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        // cov(x, y) = 5.0
        assertEquals(5.0, covariance(x, y), 1e-10)
    }
}
