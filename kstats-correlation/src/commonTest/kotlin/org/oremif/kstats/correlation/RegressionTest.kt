package org.oremif.kstats.correlation

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class RegressionTest {

    private val tol = 1e-10

    // --- Basic correctness ---

    @Test
    fun testPerfectFit() {
        // y = 2x + 1
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = simpleLinearRegression(x, y)
        assertEquals(2.0, result.slope, tol)
        assertEquals(1.0, result.intercept, tol)
        assertEquals(1.0, result.rSquared, tol)
    }

    @Test
    fun testNegativeSlope() {
        // y = -3x + 20
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(17.0, 14.0, 11.0, 8.0, 5.0)
        val result = simpleLinearRegression(x, y)
        assertEquals(-3.0, result.slope, tol)
        assertEquals(20.0, result.intercept, tol)
        assertEquals(1.0, result.rSquared, tol)
    }

    @Test
    fun testRSquaredImperfect() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 5.0, 6.0, 9.0, 10.0)
        val result = simpleLinearRegression(x, y)
        assertTrue(result.rSquared > 0.9)
        assertTrue(result.rSquared < 1.0)
    }

    @Test
    fun testNReturned() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        assertEquals(5, simpleLinearRegression(x, y).n)
    }

    // --- Predict ---

    @Test
    fun testPredictSingle() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = simpleLinearRegression(x, y)
        assertEquals(13.0, result.predict(6.0), tol)
    }

    @Test
    fun testPredictBatch() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = simpleLinearRegression(x, y)
        val predictions = result.predict(doubleArrayOf(6.0, 7.0, 0.0))
        assertEquals(13.0, predictions[0], tol)
        assertEquals(15.0, predictions[1], tol)
        assertEquals(1.0, predictions[2], tol)
    }

    // --- Residuals ---

    @Test
    fun testResidualsPerfectFit() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = simpleLinearRegression(x, y)
        for (r in result.residuals) {
            assertEquals(0.0, r, tol)
        }
    }

    @Test
    fun testResidualsSumToZero() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 5.0, 6.0, 9.0, 10.0)
        val result = simpleLinearRegression(x, y)
        val residualSum = result.residuals.sum()
        assertEquals(0.0, residualSum, 1e-10)
    }

    // --- Standard errors ---

    @Test
    fun testStandardErrorsPerfectFit() {
        // Perfect fit → residuals = 0 → standard errors = 0
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = simpleLinearRegression(x, y)
        assertEquals(0.0, result.standardErrorSlope, tol)
        assertEquals(0.0, result.standardErrorIntercept, tol)
    }

    @Test
    fun testStandardErrorsPositive() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 5.0, 6.0, 9.0, 10.0)
        val result = simpleLinearRegression(x, y)
        assertTrue(result.standardErrorSlope > 0.0)
        assertTrue(result.standardErrorIntercept > 0.0)
    }

    // --- Validation ---

    @Test
    fun testTooFewObservations() {
        assertFailsWith<InsufficientDataException> {
            simpleLinearRegression(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        }
    }

    @Test
    fun testDifferentSizes() {
        assertFailsWith<InvalidParameterException> {
            simpleLinearRegression(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0))
        }
    }

    @Test
    fun testAllIdenticalX() {
        assertFailsWith<DegenerateDataException> {
            simpleLinearRegression(
                doubleArrayOf(5.0, 5.0, 5.0, 5.0),
                doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            )
        }
    }

    // --- Edge cases ---

    @Test
    fun testMinimumSampleSize() {
        // n=3
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0)
        val result = simpleLinearRegression(x, y)
        assertEquals(2.0, result.slope, tol)
        assertEquals(0.0, result.intercept, tol)
    }

    @Test
    fun testZeroSlope() {
        // y constant → slope = 0, R² = 1 (no variance left to explain)
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(7.0, 7.0, 7.0, 7.0, 7.0)
        val result = simpleLinearRegression(x, y)
        assertEquals(0.0, result.slope, tol)
        assertEquals(7.0, result.intercept, tol)
        // syy = 0 → rSquared = 1.0 by convention (no variance to explain)
        assertEquals(1.0, result.rSquared, tol)
    }

    // --- Numerical stability ---

    @Test
    fun testLargeOffset() {
        // Verify numerical stability with large constant offset
        val offset = 1e12
        val x = DoubleArray(10) { offset + (it + 1).toDouble() }
        val y = DoubleArray(10) { 2.0 * (offset + (it + 1).toDouble()) + 1.0 }
        val result = simpleLinearRegression(x, y)
        assertEquals(2.0, result.slope, 1e-6)
        assertEquals(1.0, result.intercept, 1e-2)
        assertTrue(result.rSquared > 0.999999, "R² should be near 1.0, got ${result.rSquared}")
    }

    // --- Non-finite input ---

    @Test
    fun testNaNInX() {
        val x = doubleArrayOf(1.0, Double.NaN, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = simpleLinearRegression(x, y)
        assertTrue(result.slope.isNaN() || result.intercept.isNaN())
    }

    @Test
    fun testNaNInY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, Double.NaN, 6.0, 8.0, 10.0)
        val result = simpleLinearRegression(x, y)
        assertTrue(result.slope.isNaN() || result.intercept.isNaN())
    }

    @Test
    fun testInfinityInX() {
        val x = doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = simpleLinearRegression(x, y)
        assertTrue(result.slope.isNaN() || result.slope.isInfinite())
    }

    @Test
    fun testNegativeInfinityInY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, Double.NEGATIVE_INFINITY, 6.0, 8.0, 10.0)
        val result = simpleLinearRegression(x, y)
        assertTrue(result.slope.isNaN() || result.slope.isInfinite())
    }

    // --- Equals / hashCode ---

    @Test
    fun testEquality() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val r1 = simpleLinearRegression(x, y)
        val r2 = simpleLinearRegression(x, y)
        assertEquals(r1, r2)
        assertEquals(r1.hashCode(), r2.hashCode())
    }
}
