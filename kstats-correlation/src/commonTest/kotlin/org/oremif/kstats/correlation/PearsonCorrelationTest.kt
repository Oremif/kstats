package org.oremif.kstats.correlation

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PearsonCorrelationTest {

    private val tol = 1e-10

    // --- Basic correctness ---

    @Test
    fun testPerfectPositive() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = pearsonCorrelation(x, y)
        assertEquals(1.0, result.coefficient, tol)
        assertEquals(0.0, result.pValue, tol)
    }

    @Test
    fun testPerfectNegative() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(10.0, 8.0, 6.0, 4.0, 2.0)
        val result = pearsonCorrelation(x, y)
        assertEquals(-1.0, result.coefficient, tol)
        assertEquals(0.0, result.pValue, tol)
    }

    @Test
    fun testGeneralCorrelation() {
        // scipy: pearsonr([1..10], [2.1,3.9,6.2,7.8,10.1,12.0,14.1,15.9,18.2,19.8])
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1, 12.0, 14.1, 15.9, 18.2, 19.8)
        val result = pearsonCorrelation(x, y)
        assertEquals(0.999688789811339, result.coefficient, tol)
        assertTrue(result.pValue < 1e-13)
    }

    @Test
    fun testWeakCorrelation() {
        // scipy: r=0.939393939393939, p=5.48405299851367e-05
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val y = doubleArrayOf(2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5)
        val result = pearsonCorrelation(x, y)
        assertEquals(0.939393939393939, result.coefficient, tol)
        assertEquals(5.48405299851367e-05, result.pValue, 1e-14)
    }

    @Test
    fun testNReturned() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        assertEquals(5, pearsonCorrelation(x, y).n)
    }

    // --- Edge cases ---

    @Test
    fun testMinimumSampleSize() {
        // n=3 with perfect correlation
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(1.5, 2.5, 3.5)
        val result = pearsonCorrelation(x, y)
        assertEquals(1.0, result.coefficient, tol)
        assertEquals(0.0, result.pValue, tol)
    }

    @Test
    fun testPerfectPositivePValueZero() {
        // When r = ±1, p-value should be exactly 0.0
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val y = DoubleArray(10) { 2.0 * x[it] + 1.0 }
        val result = pearsonCorrelation(x, y)
        assertEquals(1.0, result.coefficient, tol)
        assertEquals(0.0, result.pValue)
    }

    @Test
    fun testPerfectNegativePValueZero() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val y = DoubleArray(10) { -3.0 * x[it] + 100.0 }
        val result = pearsonCorrelation(x, y)
        assertEquals(-1.0, result.coefficient, tol)
        assertEquals(0.0, result.pValue)
    }

    // --- Degenerate input ---

    @Test
    fun testConstantX() {
        val x = doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = pearsonCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
        assertTrue(result.pValue.isNaN())
    }

    @Test
    fun testConstantY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(7.0, 7.0, 7.0, 7.0, 7.0)
        val result = pearsonCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
        assertTrue(result.pValue.isNaN())
    }

    @Test
    fun testBothConstant() {
        val x = doubleArrayOf(3.0, 3.0, 3.0, 3.0)
        val y = doubleArrayOf(7.0, 7.0, 7.0, 7.0)
        val result = pearsonCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
    }

    @Test
    fun testTooFewObservations() {
        assertFailsWith<InsufficientDataException> {
            pearsonCorrelation(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        }
    }

    @Test
    fun testDifferentSizes() {
        assertFailsWith<InvalidParameterException> {
            pearsonCorrelation(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0))
        }
    }

    // --- Extreme parameters (t-statistic stability when r → ±1) ---

    @Test
    fun testNearPerfectPositiveStable() {
        // r very close to 1 — old formula (1 - r²) loses precision, new formula (1-r)(1+r) is stable
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val y = DoubleArray(10) { x[it] + 1e-12 * (it % 2 * 2 - 1) }
        val result = pearsonCorrelation(x, y)
        assertTrue(result.coefficient > 0.999999)
        assertTrue(result.pValue < 1e-10)
        assertTrue(result.pValue.isFinite())
    }

    @Test
    fun testNearPerfectNegativeStable() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val y = DoubleArray(10) { -x[it] + 1e-12 * (it % 2 * 2 - 1) }
        val result = pearsonCorrelation(x, y)
        assertTrue(result.coefficient < -0.999999)
        assertTrue(result.pValue < 1e-10)
        assertTrue(result.pValue.isFinite())
    }

    @Test
    fun testLargeNHighCorrelation() {
        // Large sample with near-perfect correlation — must produce finite p-value
        val n = 1000
        val x = DoubleArray(n) { it.toDouble() }
        val y = DoubleArray(n) { 2.5 * it + 3.0 + 1e-10 * (it % 3 - 1) }
        val result = pearsonCorrelation(x, y)
        assertTrue(result.coefficient > 0.9999999)
        assertEquals(0.0, result.pValue)
        assertTrue(result.pValue.isFinite())
    }

    // --- Non-finite input ---

    @Test
    fun testNaNPropagation() {
        val x = doubleArrayOf(1.0, 2.0, Double.NaN, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = pearsonCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
    }

    @Test
    fun testInfinityInX() {
        val x = doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = pearsonCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
    }

    @Test
    fun testNegativeInfinityInY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, Double.NEGATIVE_INFINITY, 6.0, 8.0, 10.0)
        val result = pearsonCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
    }
}
