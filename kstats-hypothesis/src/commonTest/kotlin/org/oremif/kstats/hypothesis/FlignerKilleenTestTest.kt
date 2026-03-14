package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FlignerKilleenTestTest {

    private val tolStat = 1e-10
    private val tolP = 1e-3

    // ── 1. Basic correctness (scipy reference values) ──────────────────────

    @Test
    fun twoGroupsEqualVariance() {
        // scipy: fligner([1,2,3,4,5], [6,7,8,9,10]) → (6.59e-31, 1.0)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = flignerKilleenTest(g1, g2)
        assertEquals("Fligner-Killeen Test", result.testName)
        assertEquals(0.0, result.statistic, 1e-6)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun twoGroupsDifferentVariance() {
        // scipy: fligner([10,11,12,9,10], [5,15,10,20,0]) → (3.6283, 0.05680)
        val g1 = doubleArrayOf(10.0, 11.0, 12.0, 9.0, 10.0)
        val g2 = doubleArrayOf(5.0, 15.0, 10.0, 20.0, 0.0)
        val result = flignerKilleenTest(g1, g2)
        assertEquals(3.628298116075668, result.statistic, tolStat)
        assertEquals(0.05680487660778472, result.pValue, tolP)
    }

    @Test
    fun threeGroupsEqualVariance() {
        // scipy: fligner([1..5], [6..10], [11..15]) → (0.0, 1.0)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
        val result = flignerKilleenTest(g1, g2, g3)
        assertEquals(0.0, result.statistic, 1e-6)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun threeGroupsDifferentVariance() {
        // scipy: fligner([10,11,12,9,10], [5,15,10,20,0], [8,9,10,11,12])
        //   → (5.8192, 0.05450)
        val g1 = doubleArrayOf(10.0, 11.0, 12.0, 9.0, 10.0)
        val g2 = doubleArrayOf(5.0, 15.0, 10.0, 20.0, 0.0)
        val g3 = doubleArrayOf(8.0, 9.0, 10.0, 11.0, 12.0)
        val result = flignerKilleenTest(g1, g2, g3)
        assertEquals(5.819209146478717, result.statistic, tolStat)
        assertEquals(0.05449727529012288, result.pValue, tolP)
    }

    @Test
    fun fiveGroups() {
        // scipy: fligner([1..5], [2,4,6,8,10], [1,3,5,7,9], [3,6,9,12,15], [0.5,1,1.5,2,2.5])
        //   → (8.0778, 0.08877)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val g3 = doubleArrayOf(1.0, 3.0, 5.0, 7.0, 9.0)
        val g4 = doubleArrayOf(3.0, 6.0, 9.0, 12.0, 15.0)
        val g5 = doubleArrayOf(0.5, 1.0, 1.5, 2.0, 2.5)
        val result = flignerKilleenTest(g1, g2, g3, g4, g5)
        assertEquals(8.077844101257757, result.statistic, tolStat)
        assertEquals(0.08876792756759197, result.pValue, tolP)
    }

    @Test
    fun unequalGroupSizes() {
        // scipy: fligner([1,2,3], [4,5,6,7,8]) → (1.1138, 0.29125)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0)
        val g2 = doubleArrayOf(4.0, 5.0, 6.0, 7.0, 8.0)
        val result = flignerKilleenTest(g1, g2)
        assertEquals(1.113841119690382, result.statistic, tolStat)
        assertEquals(0.291248495959798, result.pValue, tolP)
    }

    // ── 2. Edge cases ──────────────────────────────────────────────────────

    @Test
    fun minimumSize2x2() {
        // scipy: fligner([1,3], [2,8]) → (3.0, 0.08326)
        val g1 = doubleArrayOf(1.0, 3.0)
        val g2 = doubleArrayOf(2.0, 8.0)
        val result = flignerKilleenTest(g1, g2)
        assertEquals(3.0, result.statistic, 1e-6)
        assertEquals(0.08326451666355052, result.pValue, tolP)
    }

    @Test
    fun largeGroupsDifferentVariance() {
        val g1 = DoubleArray(50) { i -> (i - 25.0) / 25.0 }
        val g2 = DoubleArray(50) { i -> (i - 25.0) * 4.0 }
        val result = flignerKilleenTest(g1, g2)
        // scipy: stat=63.2299, p≈1.84e-15
        assertEquals(63.2298731493289, result.statistic, 1e-6)
        assertTrue(result.pValue < 1e-12, "Very different variances should be highly significant")
    }

    @Test
    fun largeGroupsEqualVariance() {
        val g1 = DoubleArray(100) { i -> i.toDouble() }
        val g2 = DoubleArray(100) { i -> i.toDouble() + 1000.0 }
        val result = flignerKilleenTest(g1, g2)
        assertEquals(0.0, result.statistic, 1e-6)
        assertTrue(result.pValue > 0.99, "Equal variances should have p-value near 1")
    }

    @Test
    fun dfVerification() {
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
        val result = flignerKilleenTest(g1, g2, g3)
        assertEquals(2.0, result.degreesOfFreedom) // k-1 = 3-1 = 2
    }

    // ── 3. Degenerate cases ────────────────────────────────────────────────

    @Test
    fun allConstantSameValue() {
        // All groups constant with same value → all deviations zero → stat=0, p=1
        val g1 = doubleArrayOf(5.0, 5.0, 5.0)
        val g2 = doubleArrayOf(5.0, 5.0, 5.0)
        val result = flignerKilleenTest(g1, g2)
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    @Test
    fun allConstantDifferentValues() {
        // Groups constant with different values → all deviations zero → stat=0, p=1
        val g1 = doubleArrayOf(3.0, 3.0, 3.0)
        val g2 = doubleArrayOf(7.0, 7.0, 7.0)
        val result = flignerKilleenTest(g1, g2)
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    // ── 4. Input validation ────────────────────────────────────────────────

    @Test
    fun fewerThan2Groups() {
        assertFailsWith<InsufficientDataException> {
            flignerKilleenTest(doubleArrayOf(1.0, 2.0))
        }
    }

    @Test
    fun zeroGroups() {
        assertFailsWith<InsufficientDataException> {
            @Suppress("SpreadOperator")
            flignerKilleenTest()
        }
    }

    @Test
    fun groupSizeLessThan2() {
        assertFailsWith<InsufficientDataException> {
            flignerKilleenTest(doubleArrayOf(1.0), doubleArrayOf(2.0, 3.0))
        }
    }

    @Test
    fun emptyGroup() {
        assertFailsWith<InsufficientDataException> {
            flignerKilleenTest(doubleArrayOf(), doubleArrayOf(2.0, 3.0))
        }
    }

    // ── 5. Non-finite input ────────────────────────────────────────────────

    @Test
    fun nanInGroup() {
        val g1 = doubleArrayOf(1.0, 2.0, Double.NaN, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = flignerKilleenTest(g1, g2)
        assertTrue(result.pValue.isNaN(), "NaN in input should produce NaN p-value")
    }

    @Test
    fun infinityInGroup() {
        val g1 = doubleArrayOf(1.0, 2.0, Double.POSITIVE_INFINITY, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = flignerKilleenTest(g1, g2)
        assertTrue(result.pValue.isNaN(), "Infinity in input should produce NaN p-value")
    }

    // ── 6. Property-based ──────────────────────────────────────────────────

    @Test
    fun identicalGroupsStatisticNearZero() {
        val data = doubleArrayOf(1.0, 3.0, 5.0, 7.0, 9.0)
        val result = flignerKilleenTest(data, data.copyOf(), data.copyOf())
        assertEquals(0.0, result.statistic, 1e-10)
        assertTrue(result.pValue > 0.99, "Identical groups should have p-value near 1")
    }

    @Test
    fun increasingVarianceDivergenceDecreasingPValue() {
        val baseline = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0)

        // Small spread difference
        val small = doubleArrayOf(8.0, 10.0, 12.0, 14.0, 16.0)
        val r1 = flignerKilleenTest(baseline, small)

        // Large spread difference (different rank structure, not just scaled)
        val large = doubleArrayOf(0.0, 5.0, 12.0, 20.0, 30.0)
        val r2 = flignerKilleenTest(baseline, large)

        assertTrue(
            r2.statistic > r1.statistic,
            "Larger variance divergence should give larger statistic"
        )
        assertTrue(
            r2.pValue < r1.pValue,
            "Larger variance divergence should give smaller p-value"
        )
    }

    @Test
    fun isSignificantConsistency() {
        // Use groups with clearly different variances for a significant result
        val g1 = doubleArrayOf(10.0, 10.5, 11.0, 9.5, 10.0)
        val g2 = doubleArrayOf(0.0, 20.0, 5.0, 25.0, 10.0)
        val result = flignerKilleenTest(g1, g2)
        // Verify isSignificant is consistent with pValue
        assertEquals(result.pValue < 0.05, result.isSignificant(0.05))
        assertEquals(result.pValue < 0.01, result.isSignificant(0.01))
    }
}
