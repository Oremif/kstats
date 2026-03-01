package org.oremif.kstats.correlation

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PartialCorrelationTest {

    private val tol = 1e-10

    // --- Basic correctness (scipy/numpy reference values) ---

    @Test
    fun testSingleControlPositivePartialR() {
        // scipy verified: R = np.corrcoef([x,y,z]); P = np.linalg.inv(R)
        // pcor = -P[0,1]/np.sqrt(P[0,0]*P[1,1]) = 0.9669072151199085
        // df=7; t=10.027; pval=2.1015869597166104e-05
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.2, 18.1, 19.8)
        val z = doubleArrayOf(1.5, 3.2, 4.1, 6.8, 7.0, 9.5, 10.1, 12.8, 13.0, 15.5)
        val result = partialCorrelation(x, y, z)
        assertEquals(0.9669072151199085, result.coefficient, tol)
        assertEquals(2.1015869597166104e-05, result.pValue, 1e-15)
        assertEquals(10, result.n)
    }

    @Test
    fun testSingleControlNegativePartialR() {
        // scipy verified: pcor = -0.3217979514674713, pval = 0.398406863074564, df = 7
        val x = doubleArrayOf(2.0, 4.0, 5.0, 7.0, 8.0, 10.0, 12.0, 13.0, 15.0, 17.0)
        val y = doubleArrayOf(18.0, 16.0, 14.0, 13.0, 11.0, 9.0, 7.0, 6.0, 4.0, 2.0)
        val z = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val result = partialCorrelation(x, y, z)
        assertEquals(-0.3217979514674713, result.coefficient, tol)
        assertEquals(0.398406863074564, result.pValue, 1e-8)
    }

    @Test
    fun testMultipleControls() {
        // scipy verified: pcor = -0.12712799623235305, pval = 0.7263584935605172, df = 8
        val x = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 11.0, 10.0, 12.0)
        val y = doubleArrayOf(2.0, 4.0, 3.5, 6.0, 5.5, 8.0, 7.0, 10.0, 9.5, 12.0, 11.0, 14.0)
        val z1 = doubleArrayOf(1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5, 7.0)
        val z2 = doubleArrayOf(10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0)
        val result = partialCorrelation(x, y, z1, z2)
        assertEquals(-0.12712799623235305, result.coefficient, tol)
        assertEquals(0.7263584935605172, result.pValue, 1e-8)
        assertEquals(12, result.n)
    }

    @Test
    fun testThreeControls() {
        val n = 20
        val random = kotlin.random.Random(42)
        val z1 = DoubleArray(n) { random.nextDouble() * 10.0 }
        val z2 = DoubleArray(n) { random.nextDouble() * 10.0 }
        val z3 = DoubleArray(n) { random.nextDouble() * 10.0 }
        // x and y both depend on z1, z2, z3 plus noise
        val x = DoubleArray(n) { z1[it] + 0.5 * z2[it] - 0.3 * z3[it] + random.nextDouble() }
        val y = DoubleArray(n) { 2.0 * z1[it] - z2[it] + 0.7 * z3[it] + random.nextDouble() }
        val result = partialCorrelation(x, y, z1, z2, z3)
        // df = 20 - 2 - 3 = 15
        assertTrue(result.coefficient in -1.0..1.0)
        assertTrue(result.pValue in 0.0..1.0)
        assertEquals(20, result.n)
    }

    // --- Edge cases ---

    @Test
    fun testNoControlsDelegatesToPearson() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val partial = partialCorrelation(x, y)
        val pearson = pearsonCorrelation(x, y)
        assertEquals(pearson.coefficient, partial.coefficient, tol)
        assertEquals(pearson.pValue, partial.pValue, tol)
        assertEquals(pearson.n, partial.n)
    }

    @Test
    fun testEmptyControlsSpread() {
        // partialCorrelation(x, y, *emptyArray()) should delegate to Pearson
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val partial = partialCorrelation(x, y, *emptyArray())
        val pearson = pearsonCorrelation(x, y)
        assertEquals(pearson.coefficient, partial.coefficient, tol)
        assertEquals(pearson.pValue, partial.pValue, tol)
    }

    @Test
    fun testMinimumNForOneControl() {
        // k=1, need n >= 4 (df = n - 2 - k = 1)
        val x = doubleArrayOf(1.0, 2.5, 3.0, 5.0)
        val y = doubleArrayOf(2.0, 3.5, 5.0, 7.0)
        val z = doubleArrayOf(0.5, 4.0, 1.5, 3.0)
        val result = partialCorrelation(x, y, z)
        assertTrue(result.coefficient in -1.0..1.0, "r=${result.coefficient}")
        assertTrue(result.pValue in 0.0..1.0, "p=${result.pValue}")
        assertEquals(4, result.n)
    }

    @Test
    fun testMinimumNForTwoControls() {
        // k=2, need n >= 5 (df = n - 2 - k = 1)
        val x = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 4.0)
        val y = doubleArrayOf(2.0, 5.0, 3.5, 7.0, 8.0)
        val z1 = doubleArrayOf(0.5, 4.0, 1.5, 3.0, 5.5)
        val z2 = doubleArrayOf(10.0, 5.0, 8.0, 3.0, 6.0)
        val result = partialCorrelation(x, y, z1, z2)
        assertTrue(result.coefficient in -1.0..1.0, "r=${result.coefficient}")
        assertEquals(5, result.n)
    }

    @Test
    fun testNearZeroPartialCorrelation() {
        // x and y are both functions of z with independent noise
        // After controlling for z, partial r should be near zero
        val n = 100
        val random = kotlin.random.Random(123)
        val z = DoubleArray(n) { it.toDouble() }
        val x = DoubleArray(n) { z[it] + random.nextDouble() * 100.0 }
        val y = DoubleArray(n) { z[it] + random.nextDouble() * 100.0 }
        val result = partialCorrelation(x, y, z)
        // Partial r should be much smaller than Pearson r
        val pearson = pearsonCorrelation(x, y)
        assertTrue(
            abs(result.coefficient) < abs(pearson.coefficient),
            "Partial r (${result.coefficient}) should be closer to zero than Pearson r (${pearson.coefficient})"
        )
    }

    @Test
    fun testSpuriousCorrelationRemoved() {
        // Classic confounding: ice cream sales and drownings both caused by temperature
        // x = ice cream, y = drownings, z = temperature
        val z = doubleArrayOf(60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 70.0, 75.0, 80.0, 85.0)
        val x = DoubleArray(z.size) { 2.0 * z[it] + 10.0 + (it % 3) * 5.0 } // ice cream ~ temperature
        val y = DoubleArray(z.size) { 1.5 * z[it] - 50.0 + (it % 4) * 3.0 } // drownings ~ temperature
        val pearson = pearsonCorrelation(x, y)
        val partial = partialCorrelation(x, y, z)
        // Pearson should be high, partial should be lower
        assertTrue(abs(pearson.coefficient) > 0.5, "Pearson should show correlation")
        assertTrue(
            abs(partial.coefficient) < abs(pearson.coefficient),
            "Partial correlation should be reduced after controlling for confound"
        )
    }

    // --- Degenerate input ---

    @Test
    fun testConstantControl() {
        // Constant control → singular correlation matrix → NaN
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val z = doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0) // constant
        val result = partialCorrelation(x, y, z)
        assertTrue(result.coefficient.isNaN(), "Expected NaN for constant control")
        assertTrue(result.pValue.isNaN(), "Expected NaN p-value for constant control")
    }

    @Test
    fun testCollinearControlAndX() {
        // z = 2*x → perfectly collinear → singular matrix
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 3.0, 5.0, 7.0, 8.0)
        val z = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0) // z = 2*x
        val result = partialCorrelation(x, y, z)
        assertTrue(result.coefficient.isNaN(), "Expected NaN for collinear control")
        assertTrue(result.pValue.isNaN())
    }

    @Test
    fun testAllConstantArrays() {
        val x = doubleArrayOf(3.0, 3.0, 3.0, 3.0)
        val y = doubleArrayOf(3.0, 3.0, 3.0, 3.0)
        val z = doubleArrayOf(3.0, 3.0, 3.0, 3.0)
        val result = partialCorrelation(x, y, z)
        assertTrue(result.coefficient.isNaN())
        assertTrue(result.pValue.isNaN())
    }

    @Test
    fun testConstantX() {
        val x = doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val z = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = partialCorrelation(x, y, z)
        assertTrue(result.coefficient.isNaN())
        assertTrue(result.pValue.isNaN())
    }

    @Test
    fun testConstantY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(7.0, 7.0, 7.0, 7.0, 7.0)
        val z = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = partialCorrelation(x, y, z)
        assertTrue(result.coefficient.isNaN())
        assertTrue(result.pValue.isNaN())
    }

    @Test
    fun testTooFewObservationsOneControl() {
        // k=1, need n >= 4, have 3
        assertFailsWith<InsufficientDataException> {
            partialCorrelation(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0, 6.0),
                doubleArrayOf(7.0, 8.0, 9.0)
            )
        }
    }

    @Test
    fun testTooFewObservationsTwoControls() {
        // k=2, need n >= 5, have 4
        assertFailsWith<InsufficientDataException> {
            partialCorrelation(
                doubleArrayOf(1.0, 2.0, 3.0, 4.0),
                doubleArrayOf(5.0, 6.0, 7.0, 8.0),
                doubleArrayOf(9.0, 10.0, 11.0, 12.0),
                doubleArrayOf(13.0, 14.0, 15.0, 16.0)
            )
        }
    }

    @Test
    fun testMismatchedSizesXY() {
        assertFailsWith<InvalidParameterException> {
            partialCorrelation(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0),
                doubleArrayOf(7.0, 8.0, 9.0)
            )
        }
    }

    @Test
    fun testMismatchedSizesControl() {
        assertFailsWith<InvalidParameterException> {
            partialCorrelation(
                doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0),
                doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0),
                doubleArrayOf(11.0, 12.0, 13.0) // wrong size
            )
        }
    }

    // --- Extreme parameters ---

    @Test
    fun testNearCollinearControls() {
        // Two controls that are nearly identical but not quite
        val n = 20
        val z1 = DoubleArray(n) { it.toDouble() }
        val z2 = DoubleArray(n) { it.toDouble() + 1e-8 * it }
        val x = DoubleArray(n) { it.toDouble() * 2.0 + 1.0 }
        val y = DoubleArray(n) { it.toDouble() * 3.0 - 2.0 }
        val result = partialCorrelation(x, y, z1, z2)
        // Near-singular matrix — result may be NaN or a valid value
        // Either way, it shouldn't crash
        assertTrue(result.coefficient.isNaN() || result.coefficient in -1.0..1.0)
    }

    @Test
    fun testLargeN() {
        val n = 1000
        val random = kotlin.random.Random(99)
        val z = DoubleArray(n) { random.nextDouble() }
        val x = DoubleArray(n) { z[it] * 2.0 + random.nextDouble() }
        val y = DoubleArray(n) { z[it] * 3.0 + random.nextDouble() }
        val result = partialCorrelation(x, y, z)
        assertTrue(result.coefficient in -1.0..1.0)
        assertTrue(result.pValue in 0.0..1.0)
        assertEquals(1000, result.n)
    }

    @Test
    fun testManyControls() {
        // 5 control variables
        val n = 30
        val random = kotlin.random.Random(77)
        val controls = Array(5) { DoubleArray(n) { random.nextDouble() * 10.0 } }
        val x = DoubleArray(n) { controls.sumOf { c -> c[it] } + random.nextDouble() }
        val y = DoubleArray(n) { controls.sumOf { c -> c[it] * 0.5 } + random.nextDouble() }
        // df = 30 - 2 - 5 = 23
        val result = partialCorrelation(x, y, *controls)
        assertTrue(result.coefficient in -1.0..1.0)
        assertTrue(result.pValue in 0.0..1.0)
        assertEquals(30, result.n)
    }

    // --- Non-finite input ---

    @Test
    fun testNaNInX() {
        val x = doubleArrayOf(1.0, Double.NaN, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val z = doubleArrayOf(1.5, 2.5, 3.5, 4.5, 5.5)
        val result = partialCorrelation(x, y, z)
        assertTrue(result.coefficient.isNaN())
    }

    @Test
    fun testNaNInY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, Double.NaN, 6.0, 8.0, 10.0)
        val z = doubleArrayOf(1.5, 2.5, 3.5, 4.5, 5.5)
        val result = partialCorrelation(x, y, z)
        assertTrue(result.coefficient.isNaN())
    }

    @Test
    fun testNaNInControl() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val z = doubleArrayOf(1.5, Double.NaN, 3.5, 4.5, 5.5)
        val result = partialCorrelation(x, y, z)
        assertTrue(result.coefficient.isNaN())
    }

    @Test
    fun testInfinityInX() {
        val x = doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val z = doubleArrayOf(1.5, 2.5, 3.5, 4.5, 5.5)
        val result = partialCorrelation(x, y, z)
        assertTrue(result.coefficient.isNaN())
    }

    @Test
    fun testInfinityInControl() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val z = doubleArrayOf(1.5, Double.NEGATIVE_INFINITY, 3.5, 4.5, 5.5)
        val result = partialCorrelation(x, y, z)
        assertTrue(result.coefficient.isNaN())
    }

    // --- Property-based ---

    @Test
    fun testCoefficientBounded() {
        val random = kotlin.random.Random(456)
        repeat(20) {
            val n = random.nextInt(5, 30)
            val x = DoubleArray(n) { random.nextDouble() }
            val y = DoubleArray(n) { random.nextDouble() }
            val z = DoubleArray(n) { random.nextDouble() }
            val result = partialCorrelation(x, y, z)
            if (!result.coefficient.isNaN()) {
                assertTrue(
                    result.coefficient in -1.0..1.0,
                    "r out of bounds: ${result.coefficient}"
                )
                assertTrue(
                    result.pValue in 0.0..1.0,
                    "p out of bounds: ${result.pValue}"
                )
            }
        }
    }

    @Test
    fun testSymmetryInXY() {
        // partialCorrelation(x, y, z) should equal partialCorrelation(y, x, z)
        val x = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0, 8.0, 10.0)
        val y = doubleArrayOf(2.1, 4.3, 3.5, 6.1, 5.5, 8.2, 7.0, 10.3, 9.5, 11.0)
        val z = doubleArrayOf(1.5, 2.5, 2.0, 3.5, 3.0, 4.5, 4.0, 5.5, 5.0, 6.0)
        val rXY = partialCorrelation(x, y, z)
        val rYX = partialCorrelation(y, x, z)
        assertEquals(rXY.coefficient, rYX.coefficient, tol)
        assertEquals(rXY.pValue, rYX.pValue, tol)
    }

    @Test
    fun testPartialRReducedByConfound() {
        // When the correlation is mostly due to the confound,
        // partial r should have smaller absolute value than Pearson r
        val random = kotlin.random.Random(789)
        repeat(10) {
            val n = random.nextInt(10, 50)
            val z = DoubleArray(n) { random.nextDouble() * 10.0 }
            // x and y both strongly depend on z
            val x = DoubleArray(n) { z[it] * 3.0 + random.nextDouble() * 0.1 }
            val y = DoubleArray(n) { z[it] * 2.0 + random.nextDouble() * 0.1 }
            val pearson = pearsonCorrelation(x, y)
            val partial = partialCorrelation(x, y, z)
            if (!partial.coefficient.isNaN()) {
                assertTrue(
                    abs(partial.coefficient) <= abs(pearson.coefficient) + 1e-6,
                    "Partial r (${partial.coefficient}) > Pearson r (${pearson.coefficient})"
                )
            }
        }
    }

    @Test
    fun testHighPartialCorrelation() {
        // x and y share a hidden relationship beyond z
        // x = noise_x + 3*w, y = noise_y + 5*w, z is independent
        val random = kotlin.random.Random(2024)
        val n = 50
        val z = DoubleArray(n) { random.nextDouble() * 10.0 }
        val w = DoubleArray(n) { random.nextDouble() * 10.0 }
        val x = DoubleArray(n) { random.nextDouble() + 3.0 * w[it] }
        val y = DoubleArray(n) { random.nextDouble() + 5.0 * w[it] }
        val result = partialCorrelation(x, y, z)
        // Since z is independent, controlling for z shouldn't reduce the high correlation
        assertTrue(result.coefficient > 0.9, "Expected high partial correlation, got ${result.coefficient}")
        assertTrue(result.pValue < 0.001)
    }
}
