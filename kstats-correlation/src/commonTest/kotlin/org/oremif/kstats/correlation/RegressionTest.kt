package org.oremif.kstats.correlation

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RegressionTest {

    @Test
    fun testPerfectFit() {
        // y = 2x + 1
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = simpleLinearRegression(x, y)
        assertEquals(2.0, result.slope, 1e-10)
        assertEquals(1.0, result.intercept, 1e-10)
        assertEquals(1.0, result.rSquared, 1e-10)
    }

    @Test
    fun testPredict() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = simpleLinearRegression(x, y)
        assertEquals(13.0, result.predict(6.0), 1e-10)
    }

    @Test
    fun testResiduals() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = simpleLinearRegression(x, y)
        for (r in result.residuals) {
            assertEquals(0.0, r, 1e-10)
        }
    }

    @Test
    fun testRSquaredImperfect() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 5.0, 6.0, 9.0, 10.0)
        val result = simpleLinearRegression(x, y)
        assertTrue(result.rSquared > 0.9)
        assertTrue(result.rSquared < 1.0)
    }
}
