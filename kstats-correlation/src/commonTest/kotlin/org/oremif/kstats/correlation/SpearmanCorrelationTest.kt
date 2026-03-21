package org.oremif.kstats.correlation

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SpearmanCorrelationTest {

    private val tol = 1e-10

    // --- Basic correctness ---

    @Test
    fun testMonotonicallyIncreasing() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(1.0, 4.0, 9.0, 16.0, 25.0) // monotonically increasing
        val result = spearmanCorrelation(x, y)
        assertEquals(1.0, result.coefficient, tol)
    }

    @Test
    fun testPerfectPositive() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0)
        val result = spearmanCorrelation(x, y)
        assertEquals(1.0, result.coefficient, tol)
        assertEquals(0.0, result.pValue, tol)
    }

    @Test
    fun testPerfectNegative() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(50.0, 40.0, 30.0, 20.0, 10.0)
        val result = spearmanCorrelation(x, y)
        assertEquals(-1.0, result.coefficient, tol)
        assertEquals(0.0, result.pValue, tol)
    }

    @Test
    fun testGeneralCorrelation() {
        // scipy: spearmanr([1,2,3,4,5,6,7,8,9,10], [2,1,4,3,6,5,8,7,10,9])
        // rho = 0.9393939393939394, p = 5.484052998513668e-05
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val y = doubleArrayOf(2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0)
        val result = spearmanCorrelation(x, y)
        assertEquals(0.9393939393939394, result.coefficient, tol)
        assertEquals(5.484052998513668e-05, result.pValue, 1e-14)
    }

    @Test
    fun testNonlinearMonotonic() {
        // Exponential relationship — Spearman should detect perfect monotonicity
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        val y = doubleArrayOf(2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)
        val result = spearmanCorrelation(x, y)
        assertEquals(1.0, result.coefficient, tol)
    }

    @Test
    fun testNReturned() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(5.0, 4.0, 3.0, 2.0, 1.0)
        assertEquals(5, spearmanCorrelation(x, y).n)
    }

    // --- Ties ---

    @Test
    fun testTiesInX() {
        // Tied x values should use average ranks
        val x = doubleArrayOf(1.0, 1.0, 2.0, 3.0, 4.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = spearmanCorrelation(x, y)
        assertTrue(result.coefficient > 0.9)
        assertTrue(result.coefficient < 1.0)
    }

    @Test
    fun testTiesInY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(1.0, 1.0, 2.0, 3.0, 4.0)
        val result = spearmanCorrelation(x, y)
        assertTrue(result.coefficient > 0.9)
        assertTrue(result.coefficient < 1.0)
    }

    @Test
    fun testHeavyTies() {
        val x = doubleArrayOf(1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0)
        val result = spearmanCorrelation(x, y)
        // No monotonic relationship
        assertEquals(0.0, result.coefficient, 1e-10)
    }

    // --- Edge cases ---

    @Test
    fun testMinimumSampleSize() {
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(3.0, 1.0, 2.0)
        val result = spearmanCorrelation(x, y)
        assertTrue(result.coefficient in -1.0..1.0)
        assertTrue(result.pValue in 0.0..1.0)
    }

    @Test
    fun testConstantX() {
        val x = doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = spearmanCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
        assertTrue(result.pValue.isNaN())
    }

    @Test
    fun testConstantY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(7.0, 7.0, 7.0, 7.0, 7.0)
        val result = spearmanCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
        assertTrue(result.pValue.isNaN())
    }

    // --- Validation ---

    @Test
    fun testTooFewObservations() {
        assertFailsWith<InsufficientDataException> {
            spearmanCorrelation(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        }
    }

    @Test
    fun testDifferentSizes() {
        assertFailsWith<InvalidParameterException> {
            spearmanCorrelation(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0))
        }
    }

    // --- Properties ---

    @Test
    fun testSymmetric() {
        val x = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 4.0)
        val y = doubleArrayOf(2.0, 1.0, 4.0, 3.0, 5.0)
        val r1 = spearmanCorrelation(x, y)
        val r2 = spearmanCorrelation(y, x)
        assertEquals(r1.coefficient, r2.coefficient, tol)
        assertEquals(r1.pValue, r2.pValue, tol)
    }

    @Test
    fun testSignFlipSymmetry() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 1.0, 4.0, 3.0, 5.0)
        val r1 = spearmanCorrelation(x, y)
        val yNeg = DoubleArray(y.size) { -y[it] }
        val r2 = spearmanCorrelation(x, yNeg)
        assertEquals(-r1.coefficient, r2.coefficient, tol)
    }

    @Test
    fun testCoefficientBounded() {
        val random = kotlin.random.Random(42)
        repeat(20) {
            val n = random.nextInt(3, 50)
            val x = DoubleArray(n) { random.nextDouble() }
            val y = DoubleArray(n) { random.nextDouble() }
            val result = spearmanCorrelation(x, y)
            assertTrue(result.coefficient in -1.0..1.0, "r out of bounds: ${result.coefficient}")
            assertTrue(result.pValue in 0.0..1.0, "p out of bounds: ${result.pValue}")
        }
    }

    // --- Non-finite input ---

    @Test
    fun testNaNInInputProducesFiniteResult() {
        // rank() assigns NaN a valid rank (placed after all finite values),
        // so Spearman on ranks produces a finite correlation coefficient
        val x = doubleArrayOf(1.0, 2.0, Double.NaN, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = spearmanCorrelation(x, y)
        assertTrue(result.coefficient.isFinite())
        assertTrue(result.coefficient in -1.0..1.0)
    }

    // --- Larger dataset ---

    @Test
    fun testLargeN() {
        val n = 1000
        val x = DoubleArray(n) { it.toDouble() }
        val y = DoubleArray(n) { it.toDouble() * 2.0 + 1.0 }
        val result = spearmanCorrelation(x, y)
        assertEquals(1.0, result.coefficient, tol)
        assertEquals(0.0, result.pValue, tol)
    }
}
