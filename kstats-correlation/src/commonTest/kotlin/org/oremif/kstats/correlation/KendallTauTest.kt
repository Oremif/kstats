package org.oremif.kstats.correlation

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class KendallTauTest {

    // ========== Basic correctness ==========

    @Test
    fun testPerfectConcordant() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = kendallTau(x, y)
        assertEquals(1.0, result.coefficient, 1e-10)
        assertTrue(result.pValue < 0.05)
    }

    @Test
    fun testPerfectDiscordant() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(5.0, 4.0, 3.0, 2.0, 1.0)
        val result = kendallTau(x, y)
        assertEquals(-1.0, result.coefficient, 1e-10)
        assertTrue(result.pValue < 0.05)
    }

    @Test
    fun testPartialCorrelation() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 4.0)
        val result = kendallTau(x, y)
        assertEquals(0.6, result.coefficient, 1e-10)
    }

    @Test
    fun testLiteratureDataset() {
        // n=19 dataset with repeated y values
        val x = doubleArrayOf(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0
        )
        val y = doubleArrayOf(
            12.0, 14.0, 14.0, 17.0, 13.0, 9.0, 8.0, 10.0, 15.0, 6.0,
            11.0, 5.0, 2.0, 16.0, 7.0, 1.0, 3.0, 4.0, 18.0
        )
        val result = kendallTau(x, y)
        // Verify tau is negative (inversely correlated) and roughly in expected range
        assertTrue(result.coefficient < 0.0)
        assertTrue(result.coefficient > -1.0)
        assertTrue(result.pValue < 0.05, "Expected significant correlation")
    }

    @Test
    fun testWeakCorrelation() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        val y = doubleArrayOf(2.0, 1.0, 4.0, 3.0, 6.0, 8.0, 5.0, 7.0)
        val result = kendallTau(x, y)
        // 23 concordant, 5 discordant, no ties: tau = 18/28 = 9/14
        assertEquals(9.0 / 14.0, result.coefficient, 1e-10)
    }

    // ========== Ties ==========

    @Test
    fun testTiesInX() {
        val x = doubleArrayOf(1.0, 1.0, 2.0, 3.0, 4.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = kendallTau(x, y)
        assertEquals(0.9487, result.coefficient, 1e-3)
        assertEquals(0.02298, result.pValue, 1e-3)
    }

    @Test
    fun testTiesInY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(1.0, 1.0, 2.0, 3.0, 4.0)
        val result = kendallTau(x, y)
        assertEquals(0.9487, result.coefficient, 1e-3)
        assertEquals(0.02298, result.pValue, 1e-3)
    }

    @Test
    fun testTiesInBoth() {
        val x = doubleArrayOf(1.0, 1.0, 2.0, 2.0, 3.0)
        val y = doubleArrayOf(1.0, 2.0, 1.0, 2.0, 3.0)
        val result = kendallTau(x, y)
        // 5 concordant, 1 discordant, 2 x-tied pairs, 2 y-tied pairs
        // tau-b = (5-1)/sqrt((10-2)*(10-2)) = 4/8 = 0.5
        assertEquals(0.5, result.coefficient, 1e-10)
    }

    @Test
    fun testHeavyTies() {
        // 3x3 grid: all ranks tied equally -> tau=0.0
        val x = doubleArrayOf(1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0)
        val result = kendallTau(x, y)
        assertEquals(0.0, result.coefficient, 1e-10)
        assertEquals(1.0, result.pValue, 1e-3)
    }

    // ========== Edge / degenerate cases ==========

    @Test
    fun testMinimumSize() {
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(1.0, 3.0, 2.0)
        val result = kendallTau(x, y)
        // 2 concordant, 1 discordant: tau = 1/3
        assertEquals(1.0 / 3.0, result.coefficient, 1e-10)
    }

    @Test
    fun testAllSameX() {
        val x = doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = kendallTau(x, y)
        assertTrue(result.coefficient.isNaN())
    }

    @Test
    fun testAllSameY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0)
        val result = kendallTau(x, y)
        assertTrue(result.coefficient.isNaN())
    }

    @Test
    fun testInsufficientData() {
        assertFailsWith<InsufficientDataException> {
            kendallTau(doubleArrayOf(1.0, 2.0), doubleArrayOf(1.0, 2.0))
        }
    }

    @Test
    fun testMismatchedSizes() {
        assertFailsWith<InvalidParameterException> {
            kendallTau(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(1.0, 2.0))
        }
    }

    // ========== Extreme values ==========

    @Test
    fun testVeryLargeValues() {
        val x = doubleArrayOf(1e15, 2e15, 3e15, 4e15, 5e15)
        val y = doubleArrayOf(1e15, 2e15, 3e15, 4e15, 5e15)
        val result = kendallTau(x, y)
        assertEquals(1.0, result.coefficient, 1e-10)
    }

    @Test
    fun testVerySmallValues() {
        val x = doubleArrayOf(1e-15, 2e-15, 3e-15, 4e-15, 5e-15)
        val y = doubleArrayOf(1e-15, 2e-15, 3e-15, 4e-15, 5e-15)
        val result = kendallTau(x, y)
        assertEquals(1.0, result.coefficient, 1e-10)
    }

    @Test
    fun testNegativeValues() {
        val x = doubleArrayOf(-5.0, -4.0, -3.0, -2.0, -1.0)
        val y = doubleArrayOf(-5.0, -4.0, -3.0, -2.0, -1.0)
        val result = kendallTau(x, y)
        assertEquals(1.0, result.coefficient, 1e-10)
    }

    @Test
    fun testMixedPositiveNegative() {
        val x = doubleArrayOf(-2.0, -1.0, 0.0, 1.0, 2.0)
        val y = doubleArrayOf(2.0, 1.0, 0.0, -1.0, -2.0)
        val result = kendallTau(x, y)
        assertEquals(-1.0, result.coefficient, 1e-10)
    }

    // ========== Properties ==========

    @Test
    fun testTauBounded() {
        val x = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0)
        val y = doubleArrayOf(2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 7.0)
        val result = kendallTau(x, y)
        assertTrue(result.coefficient >= -1.0 && result.coefficient <= 1.0)
        assertTrue(result.pValue in 0.0..1.0)
    }

    @Test
    fun testSignFlipSymmetry() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 1.0, 4.0, 3.0, 5.0)
        val r1 = kendallTau(x, y)
        // Negating y should negate tau
        val yNeg = DoubleArray(y.size) { -y[it] }
        val r2 = kendallTau(x, yNeg)
        assertEquals(-r1.coefficient, r2.coefficient, 1e-10)
    }

    @Test
    fun testSymmetric() {
        val x = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 4.0)
        val y = doubleArrayOf(2.0, 1.0, 4.0, 3.0, 5.0)
        val r1 = kendallTau(x, y)
        val r2 = kendallTau(y, x)
        assertEquals(r1.coefficient, r2.coefficient, 1e-10)
        assertEquals(r1.pValue, r2.pValue, 1e-10)
    }

    @Test
    fun testPerfectRankGivesPlusOrMinusOne() {
        val n = 20
        val x = DoubleArray(n) { it.toDouble() }
        val y = DoubleArray(n) { it.toDouble() }
        assertEquals(1.0, kendallTau(x, y).coefficient, 1e-10)
        val yRev = DoubleArray(n) { (n - 1 - it).toDouble() }
        assertEquals(-1.0, kendallTau(x, yRev).coefficient, 1e-10)
    }

    // ========== P-value validation ==========

    @Test
    fun testPValueSignificant() {
        // Strong correlation with enough data should give small p-value
        val n = 20
        val x = DoubleArray(n) { it.toDouble() }
        val y = DoubleArray(n) { it.toDouble() + 0.1 * (it % 3) }
        val result = kendallTau(x, y)
        assertTrue(result.pValue < 0.05, "Expected p < 0.05, got ${result.pValue}")
    }

    @Test
    fun testPValueNoCorrelation() {
        // Uncorrelated data should give large p-value
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 1.0, 5.0, 2.0, 4.0)
        val result = kendallTau(x, y)
        // tau = 0.2, not significant for n=5
        assertTrue(result.pValue > 0.3, "Expected large p-value, got ${result.pValue}")
    }

    @Test
    fun testPValuePerfectCorrelationLargerN() {
        // With n=10, perfect correlation should be highly significant
        val n = 10
        val x = DoubleArray(n) { it.toDouble() }
        val y = DoubleArray(n) { it.toDouble() }
        val result = kendallTau(x, y)
        assertTrue(result.pValue < 0.001, "Expected p < 0.001, got ${result.pValue}")
    }

    // ========== Larger dataset ==========

    @Test
    fun testLargerDataset() {
        val n = 100
        val x = DoubleArray(n) { it.toDouble() }
        val y = DoubleArray(n) { it.toDouble() }
        // Swap adjacent pairs for first 20 elements
        for (i in 0 until 20 step 2) {
            val tmp = y[i]; y[i] = y[i + 1]; y[i + 1] = tmp
        }
        val result = kendallTau(x, y)
        assertTrue(result.coefficient > 0.5, "Expected strong positive correlation")
        assertTrue(result.coefficient < 1.0, "Expected not perfect correlation")
        assertTrue(result.pValue < 0.001, "Expected highly significant p-value")
    }

    @Test
    fun testNaNInXProducesNaN() {
        val x = doubleArrayOf(1.0, 2.0, Double.NaN, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = kendallTau(x, y)
        assertTrue(result.coefficient.isNaN())
        assertTrue(result.pValue.isNaN())
    }

    @Test
    fun testNaNInYProducesNaN() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, Double.NaN, 6.0, 8.0, 10.0)
        val result = kendallTau(x, y)
        assertTrue(result.coefficient.isNaN())
        assertTrue(result.pValue.isNaN())
    }

    @Test
    fun testInfinityInX() {
        val x = doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = kendallTau(x, y)
        assertTrue(result.coefficient in -1.0..1.0)
        assertTrue(result.pValue in 0.0..1.0)
    }

    @Test
    fun testNegativeInfinityInY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, Double.NEGATIVE_INFINITY, 6.0, 8.0, 10.0)
        val result = kendallTau(x, y)
        assertTrue(result.coefficient in -1.0..1.0)
        assertTrue(result.pValue in 0.0..1.0)
    }

    @Test
    fun testConsistencyWithBruteForce() {
        // Verify O(n log n) algorithm matches O(n²) brute force
        val x = doubleArrayOf(3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0)
        val y = doubleArrayOf(2.0, 7.0, 1.0, 8.0, 2.0, 8.0, 1.0, 8.0)
        val result = kendallTau(x, y)

        // Brute-force calculation
        val n = x.size
        var con = 0; var disc = 0; var tx = 0; var ty = 0
        for (i in 0 until n) {
            for (j in i + 1 until n) {
                val xd = x[i].compareTo(x[j])
                val yd = y[i].compareTo(y[j])
                if (xd == 0 && yd == 0) { tx++; ty++ }
                else if (xd == 0) tx++
                else if (yd == 0) ty++
                else if (xd * yd > 0) con++
                else disc++
            }
        }
        val n0 = n.toLong() * (n - 1) / 2
        val bruteTau = (con - disc).toDouble() /
            kotlin.math.sqrt((n0 - tx).toDouble() * (n0 - ty).toDouble())
        assertEquals(bruteTau, result.coefficient, 1e-10)
    }
}
