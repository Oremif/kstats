package org.oremif.kstats.correlation

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PointBiserialCorrelationTest {

    private val tol = 1e-10

    // --- Basic correctness (scipy reference values) ---

    @Test
    fun testPositiveCorrelation() {
        // scipy: pointbiserialr([0,0,0,0,0,1,1,1,1,1], [1.2,2.3,1.8,2.1,1.5,5.4,6.1,5.8,6.3,5.0])
        val x = doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        val y = doubleArrayOf(1.2, 2.3, 1.8, 2.1, 1.5, 5.4, 6.1, 5.8, 6.3, 5.0)
        val result = pointBiserialCorrelation(x, y)
        assertEquals(0.9764327706793505, result.coefficient, tol)
        assertEquals(1.3118273746219274e-06, result.pValue, 1e-16)
        assertEquals(10, result.n)
    }

    @Test
    fun testNegativeCorrelation() {
        // scipy: pointbiserialr([0,0,0,0,0,1,1,1,1,1], [5.4,6.1,5.8,6.3,5.0,1.2,2.3,1.8,2.1,1.5])
        val x = doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        val y = doubleArrayOf(5.4, 6.1, 5.8, 6.3, 5.0, 1.2, 2.3, 1.8, 2.1, 1.5)
        val result = pointBiserialCorrelation(x, y)
        assertEquals(-0.9764327706793505, result.coefficient, tol)
        assertEquals(1.3118273746219274e-06, result.pValue, 1e-16)
    }

    @Test
    fun testNearZeroCorrelation() {
        // scipy: pointbiserialr([0,1,0,1,0,1,0,1,0,1], [3.1,2.9,3.5,3.2,2.8,3.6,3.0,3.3,3.4,2.7])
        val x = doubleArrayOf(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        val y = doubleArrayOf(3.1, 2.9, 3.5, 3.2, 2.8, 3.6, 3.0, 3.3, 3.4, 2.7)
        val result = pointBiserialCorrelation(x, y)
        assertEquals(-0.03481553119113946, result.coefficient, tol)
        assertEquals(0.9239332723098315, result.pValue, tol)
    }

    @Test
    fun testEquivalenceToPearson() {
        val x = doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        val y = doubleArrayOf(1.2, 2.3, 1.8, 2.1, 1.5, 5.4, 6.1, 5.8, 6.3, 5.0)
        val pbr = pointBiserialCorrelation(x, y)
        val pr = pearsonCorrelation(x, y)
        assertEquals(pr.coefficient, pbr.coefficient, tol)
        assertEquals(pr.pValue, pbr.pValue, tol)
    }

    @Test
    fun testNReturned() {
        val x = doubleArrayOf(0.0, 0.0, 1.0, 1.0, 1.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(5, pointBiserialCorrelation(x, y).n)
    }

    // --- Edge cases ---

    @Test
    fun testMinimumN() {
        // scipy: pointbiserialr([0,0,1], [1.0,2.0,5.0])
        val x = doubleArrayOf(0.0, 0.0, 1.0)
        val y = doubleArrayOf(1.0, 2.0, 5.0)
        val result = pointBiserialCorrelation(x, y)
        assertEquals(0.970725343394151, result.coefficient, tol)
        assertEquals(0.15442095831126654, result.pValue, tol)
    }

    @Test
    fun testUnequalGroupSizes() {
        // scipy: pointbiserialr([0,0,0,0,0,0,0,0,0,1], [1..10])
        val x = doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val result = pointBiserialCorrelation(x, y)
        assertEquals(0.5222329678670934, result.coefficient, tol)
        assertEquals(0.12150291881711325, result.pValue, tol)
    }

    @Test
    fun testNonStandardBinaryValues() {
        // Binary variable coded as 5.0/10.0 — should remap to 0/1
        val x = doubleArrayOf(5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 5.0, 10.0)
        val y = doubleArrayOf(2.1, 3.4, 2.8, 7.5, 8.1, 6.9, 3.0, 7.2)
        val result = pointBiserialCorrelation(x, y)
        assertEquals(0.9807790228998718, result.coefficient, tol)
        assertEquals(1.749784533306829e-05, result.pValue, 1e-15)
    }

    @Test
    fun testNonStandardNegativeBinaryValues() {
        // Binary variable coded as -1.0/1.0 — should remap to 0/1
        val x = doubleArrayOf(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0)
        val y = doubleArrayOf(2.1, 3.4, 2.8, 7.5, 8.1, 6.9, 3.0, 7.2)
        val result = pointBiserialCorrelation(x, y)
        assertEquals(0.9807790228998718, result.coefficient, tol)
        assertEquals(1.749784533306829e-05, result.pValue, 1e-15)
    }

    // --- BooleanArray and IntArray overloads ---

    @Test
    fun testBooleanArrayOverload() {
        val xBool = booleanArrayOf(false, false, false, false, false, true, true, true, true, true)
        val y = doubleArrayOf(1.2, 2.3, 1.8, 2.1, 1.5, 5.4, 6.1, 5.8, 6.3, 5.0)
        val boolResult = pointBiserialCorrelation(xBool, y)

        val xDouble = doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        val doubleResult = pointBiserialCorrelation(xDouble, y)

        assertEquals(doubleResult.coefficient, boolResult.coefficient, tol)
        assertEquals(doubleResult.pValue, boolResult.pValue, tol)
        assertEquals(doubleResult.n, boolResult.n)
    }

    @Test
    fun testIntArrayOverload() {
        val xInt = intArrayOf(0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
        val y = doubleArrayOf(1.2, 2.3, 1.8, 2.1, 1.5, 5.4, 6.1, 5.8, 6.3, 5.0)
        val intResult = pointBiserialCorrelation(xInt, y)

        val xDouble = doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        val doubleResult = pointBiserialCorrelation(xDouble, y)

        assertEquals(doubleResult.coefficient, intResult.coefficient, tol)
        assertEquals(doubleResult.pValue, intResult.pValue, tol)
        assertEquals(doubleResult.n, intResult.n)
    }

    // --- Degenerate input ---

    @Test
    fun testConstantContinuousVariable() {
        val x = doubleArrayOf(0.0, 0.0, 1.0, 1.0)
        val y = doubleArrayOf(3.0, 3.0, 3.0, 3.0)
        val result = pointBiserialCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
        assertTrue(result.pValue.isNaN())
    }

    @Test
    fun testSingleDistinctValueInX() {
        assertFailsWith<InvalidParameterException> {
            pointBiserialCorrelation(
                doubleArrayOf(1.0, 1.0, 1.0, 1.0),
                doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            )
        }
    }

    @Test
    fun testThreeDistinctValuesInX() {
        assertFailsWith<InvalidParameterException> {
            pointBiserialCorrelation(
                doubleArrayOf(0.0, 1.0, 2.0, 0.0),
                doubleArrayOf(1.0, 2.0, 3.0, 4.0)
            )
        }
    }

    @Test
    fun testTooFewObservations() {
        assertFailsWith<InsufficientDataException> {
            pointBiserialCorrelation(
                doubleArrayOf(0.0, 1.0),
                doubleArrayOf(1.0, 2.0)
            )
        }
    }

    @Test
    fun testMismatchedSizes() {
        assertFailsWith<InvalidParameterException> {
            pointBiserialCorrelation(
                doubleArrayOf(0.0, 1.0, 0.0),
                doubleArrayOf(1.0, 2.0)
            )
        }
    }

    @Test
    fun testBooleanAllTrue() {
        // All-same boolean → Pearson on constant x → NaN
        val x = booleanArrayOf(true, true, true, true)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val result = pointBiserialCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
        assertTrue(result.pValue.isNaN())
    }

    @Test
    fun testBooleanAllFalse() {
        val x = booleanArrayOf(false, false, false, false)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val result = pointBiserialCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
        assertTrue(result.pValue.isNaN())
    }

    @Test
    fun testBooleanTooFewObservations() {
        assertFailsWith<InsufficientDataException> {
            pointBiserialCorrelation(
                booleanArrayOf(true, false),
                doubleArrayOf(1.0, 2.0)
            )
        }
    }

    @Test
    fun testBooleanMismatchedSizes() {
        assertFailsWith<InvalidParameterException> {
            pointBiserialCorrelation(
                booleanArrayOf(true, false, true),
                doubleArrayOf(1.0, 2.0)
            )
        }
    }

    @Test
    fun testIntArrayTooFewObservations() {
        assertFailsWith<InsufficientDataException> {
            pointBiserialCorrelation(
                intArrayOf(0, 1),
                doubleArrayOf(1.0, 2.0)
            )
        }
    }

    @Test
    fun testIntArrayMismatchedSizes() {
        assertFailsWith<InvalidParameterException> {
            pointBiserialCorrelation(
                intArrayOf(0, 1, 0),
                doubleArrayOf(1.0, 2.0)
            )
        }
    }

    // --- Extreme parameters ---

    @Test
    fun testVeryLargeContinuousValues() {
        // scipy: r=0.9649012813540153, p=0.0018262606682599755
        val x = doubleArrayOf(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        val y = doubleArrayOf(1e15, 2e15, 1.5e15, 4e15, 5e15, 4.5e15)
        val result = pointBiserialCorrelation(x, y)
        assertEquals(0.9649012813540153, result.coefficient, tol)
        assertEquals(0.0018262606682599755, result.pValue, tol)
    }

    @Test
    fun testVerySmallContinuousValues() {
        // scipy: r=0.9649012813540154, p=0.0018262606682599755
        val x = doubleArrayOf(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        val y = doubleArrayOf(1e-15, 2e-15, 1.5e-15, 4e-15, 5e-15, 4.5e-15)
        val result = pointBiserialCorrelation(x, y)
        assertEquals(0.9649012813540154, result.coefficient, tol)
        assertEquals(0.0018262606682599755, result.pValue, tol)
    }

    @Test
    fun testLargeN() {
        // scipy: r=0.7035639735028607, p=3.1328978494911836e-150
        val random = kotlin.random.Random(42)
        val n = 1000
        val x = DoubleArray(n) { if (random.nextDouble() < 0.5) 1.0 else 0.0 }
        // Use numpy-compatible seed 42 reference; just check bounds since RNG differs
        val y = DoubleArray(n) { if (x[it] == 1.0) 5.0 + random.nextDouble() else 3.0 + random.nextDouble() }
        val result = pointBiserialCorrelation(x, y)
        assertTrue(abs(result.coefficient) > 0.3)
        assertTrue(result.coefficient in -1.0..1.0)
        assertTrue(result.pValue in 0.0..1.0)
        assertTrue(result.pValue < 0.01) // should be highly significant
        assertEquals(n, result.n)
    }

    // --- Non-finite input ---

    @Test
    fun testNaNInContinuousVariable() {
        val x = doubleArrayOf(0.0, 0.0, 1.0, 1.0, 0.0)
        val y = doubleArrayOf(1.0, 2.0, Double.NaN, 4.0, 5.0)
        val result = pointBiserialCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
    }

    @Test
    fun testNaNInBinaryVariable() {
        // NaN is not finite, so ignored during distinct value scan.
        // But it passes through to Pearson which propagates NaN.
        val x = doubleArrayOf(0.0, 0.0, 1.0, 1.0, Double.NaN)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = pointBiserialCorrelation(x, y)
        assertTrue(result.coefficient.isNaN())
    }

    // --- Property-based ---

    @Test
    fun testCoefficientBounded() {
        val random = kotlin.random.Random(123)
        repeat(20) {
            val n = random.nextInt(3, 50)
            val x = DoubleArray(n) { if (random.nextBoolean()) 1.0 else 0.0 }
            val y = DoubleArray(n) { random.nextDouble() }
            // Skip if x is all-same
            if (x.toSet().size < 2) return@repeat
            val result = pointBiserialCorrelation(x, y)
            assertTrue(result.coefficient in -1.0..1.0, "r out of bounds: ${result.coefficient}")
            assertTrue(result.pValue in 0.0..1.0, "p out of bounds: ${result.pValue}")
        }
    }

    @Test
    fun testSwappingBinaryCodingNegatesR() {
        val x = doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        val y = doubleArrayOf(1.2, 2.3, 1.8, 2.1, 1.5, 5.4, 6.1, 5.8, 6.3, 5.0)
        val r1 = pointBiserialCorrelation(x, y)

        // Swap 0 <-> 1
        val xSwapped = DoubleArray(x.size) { 1.0 - x[it] }
        val r2 = pointBiserialCorrelation(xSwapped, y)

        assertEquals(-r1.coefficient, r2.coefficient, tol)
        assertEquals(r1.pValue, r2.pValue, tol)
    }

    @Test
    fun testAllThreeOverloadsProduceSameResults() {
        val xDouble = doubleArrayOf(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0)
        val xBool = booleanArrayOf(false, false, false, true, true, true, false, true)
        val xInt = intArrayOf(0, 0, 0, 1, 1, 1, 0, 1)
        val y = doubleArrayOf(2.1, 3.4, 2.8, 7.5, 8.1, 6.9, 3.0, 7.2)

        val rd = pointBiserialCorrelation(xDouble, y)
        val rb = pointBiserialCorrelation(xBool, y)
        val ri = pointBiserialCorrelation(xInt, y)

        assertEquals(rd.coefficient, rb.coefficient, tol)
        assertEquals(rd.coefficient, ri.coefficient, tol)
        assertEquals(rd.pValue, rb.pValue, tol)
        assertEquals(rd.pValue, ri.pValue, tol)
        assertEquals(rd.n, rb.n)
        assertEquals(rd.n, ri.n)
    }
}
