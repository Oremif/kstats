package org.oremif.kstats.correlation

import kotlin.test.Test
import kotlin.test.assertEquals

class CorrelationMatrixTest {

    @Test
    fun testDiagonal() {
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(4.0, 5.0, 6.0)
        val matrix = correlationMatrix(x, y)
        assertEquals(1.0, matrix[0][0], 1e-10)
        assertEquals(1.0, matrix[1][1], 1e-10)
    }

    @Test
    fun testSymmetry() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val y = doubleArrayOf(2.0, 3.0, 5.0, 7.0)
        val matrix = correlationMatrix(x, y)
        assertEquals(matrix[0][1], matrix[1][0], 1e-10)
    }
}
