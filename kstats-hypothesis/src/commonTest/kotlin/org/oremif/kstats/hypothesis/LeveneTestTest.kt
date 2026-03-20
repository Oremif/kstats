package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LeveneTestTest {

    private val tolF = 1e-10
    private val tolP = 1e-3

    // ── 1. Basic correctness (scipy reference values) ──────────────────────

    @Test
    fun twoGroupMeanCenter() {
        // Groups with equal variance but different means → F≈0, p≈1
        // scipy: levene([1,2,3,4,5], [6,7,8,9,10], center='mean') → (0.0, 1.0)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = leveneTest(g1, g2, center = LeveneCenter.MEAN)
        assertEquals("Levene's Test", result.testName)
        assertEquals(0.0, result.statistic, tolF)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun twoGroupMedianCenter() {
        // scipy: levene([1,2,3,4,5], [6,7,8,9,10], center='median') → (0.0, 1.0)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = leveneTest(g1, g2, center = LeveneCenter.MEDIAN)
        assertEquals("Brown-Forsythe Test", result.testName)
        assertEquals(0.0, result.statistic, tolF)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun threeGroupMeanCenter() {
        // scipy: levene([1..5], [6..10], [11..15], center='mean') → (0.0, 1.0)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
        val result = leveneTest(g1, g2, g3, center = LeveneCenter.MEAN)
        assertEquals(0.0, result.statistic, tolF)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun threeGroupMedianCenter() {
        // scipy: levene([1..5], [6..10], [11..15], center='median') → (0.0, 1.0)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
        val result = leveneTest(g1, g2, g3, center = LeveneCenter.MEDIAN)
        assertEquals(0.0, result.statistic, tolF)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun threeGroupDifferentVariancesMean() {
        // scipy: levene([10,11,12,9,10], [5,15,10,20,0], [8,9,10,11,12], center='mean')
        //   → (6.658320742499461, 0.011340988949827)
        val g1 = doubleArrayOf(10.0, 11.0, 12.0, 9.0, 10.0)
        val g2 = doubleArrayOf(5.0, 15.0, 10.0, 20.0, 0.0)
        val g3 = doubleArrayOf(8.0, 9.0, 10.0, 11.0, 12.0)
        val result = leveneTest(g1, g2, g3, center = LeveneCenter.MEAN)
        assertEquals(6.658320742499461, result.statistic, tolF)
        assertEquals(0.011340988949827, result.pValue, tolP)
    }

    @Test
    fun threeGroupDifferentVariancesMedian() {
        // scipy: levene([10,11,12,9,10], [5,15,10,20,0], [8,9,10,11,12], center='median')
        //   → (6.645502645502647, 0.011410138646959)
        val g1 = doubleArrayOf(10.0, 11.0, 12.0, 9.0, 10.0)
        val g2 = doubleArrayOf(5.0, 15.0, 10.0, 20.0, 0.0)
        val g3 = doubleArrayOf(8.0, 9.0, 10.0, 11.0, 12.0)
        val result = leveneTest(g1, g2, g3, center = LeveneCenter.MEDIAN)
        assertEquals(6.645502645502647, result.statistic, tolF)
        assertEquals(0.011410138646959, result.pValue, tolP)
    }

    @Test
    fun defaultCenterIsMedian() {
        val g1 = doubleArrayOf(10.0, 11.0, 12.0, 9.0, 10.0)
        val g2 = doubleArrayOf(5.0, 15.0, 10.0, 20.0, 0.0)
        val defaultResult = leveneTest(g1, g2)
        val medianResult = leveneTest(g1, g2, center = LeveneCenter.MEDIAN)
        assertEquals(medianResult.statistic, defaultResult.statistic, 0.0)
        assertEquals(medianResult.pValue, defaultResult.pValue, 0.0)
        assertEquals("Brown-Forsythe Test", defaultResult.testName)
    }

    @Test
    fun equalVarianceGroups() {
        // Groups with same spread but shifted means
        // scipy: levene([2,4,6,8,10], [3,5,7,9,11], center='mean') → (0.0, 1.0)
        val g1 = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val g2 = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = leveneTest(g1, g2, center = LeveneCenter.MEAN)
        assertEquals(0.0, result.statistic, tolF)
        assertEquals(1.0, result.pValue, tolP)
    }

    // ── 2. Edge cases ──────────────────────────────────────────────────────

    @Test
    fun minimumSize2x2() {
        // 2 elements per group: ssWithin=0, ssBetween>0 → W=Inf, p=0
        // scipy: levene([1,3], [2,8], center='mean') → (inf, 0.0)
        val g1 = doubleArrayOf(1.0, 3.0)
        val g2 = doubleArrayOf(2.0, 8.0)
        val result = leveneTest(g1, g2, center = LeveneCenter.MEAN)
        assertTrue(result.statistic.isInfinite(), "W should be infinite when ssWithin=0 and ssBetween>0")
        assertEquals(0.0, result.pValue, tolP)
    }

    @Test
    fun fiveGroups() {
        // scipy: levene([1..5], [2,4,6,8,10], [1,3,5,7,9], [3,6,9,12,15], [0.5,1,1.5,2,2.5], center='mean')
        //   → (2.677103718199608, 0.061616119598530)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val g3 = doubleArrayOf(1.0, 3.0, 5.0, 7.0, 9.0)
        val g4 = doubleArrayOf(3.0, 6.0, 9.0, 12.0, 15.0)
        val g5 = doubleArrayOf(0.5, 1.0, 1.5, 2.0, 2.5)
        val result = leveneTest(g1, g2, g3, g4, g5, center = LeveneCenter.MEAN)
        assertEquals(2.677103718199608, result.statistic, tolF)
        assertEquals(0.061616119598530, result.pValue, tolP)
    }

    @Test
    fun fiveGroupsMedian() {
        // scipy: same groups, center='median' → (2.677103718199608, 0.061616119598530)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val g3 = doubleArrayOf(1.0, 3.0, 5.0, 7.0, 9.0)
        val g4 = doubleArrayOf(3.0, 6.0, 9.0, 12.0, 15.0)
        val g5 = doubleArrayOf(0.5, 1.0, 1.5, 2.0, 2.5)
        val result = leveneTest(g1, g2, g3, g4, g5, center = LeveneCenter.MEDIAN)
        assertEquals(2.677103718199608, result.statistic, tolF)
        assertEquals(0.061616119598530, result.pValue, tolP)
    }

    @Test
    fun unequalGroupSizes() {
        // scipy: levene([1,2,3], [4,5,6,7,8], center='mean') → (0.923076923076923, 0.373770542585760)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0)
        val g2 = doubleArrayOf(4.0, 5.0, 6.0, 7.0, 8.0)
        val result = leveneTest(g1, g2, center = LeveneCenter.MEAN)
        assertEquals(0.923076923076923, result.statistic, tolF)
        assertEquals(0.373770542585760, result.pValue, tolP)
    }

    @Test
    fun dfVerification() {
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
        val result = leveneTest(g1, g2, g3, center = LeveneCenter.MEAN)
        assertEquals(2.0, result.degreesOfFreedom) // dfBetween = k-1 = 2
        assertEquals(2.0, result.additionalInfo["dfBetween"])
        assertEquals(12.0, result.additionalInfo["dfWithin"]) // N-k = 15-3 = 12
    }

    // ── 3. Degenerate cases ────────────────────────────────────────────────

    @Test
    fun constantGroupsSameValue() {
        // All groups constant with same value → all Z=0, W=0, p=1
        val g1 = doubleArrayOf(5.0, 5.0, 5.0)
        val g2 = doubleArrayOf(5.0, 5.0, 5.0)
        val result = leveneTest(g1, g2, center = LeveneCenter.MEAN)
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    @Test
    fun constantGroupsDifferentValues() {
        // Groups constant but different values → still all Z=0, W=0, p=1
        val g1 = doubleArrayOf(3.0, 3.0, 3.0)
        val g2 = doubleArrayOf(7.0, 7.0, 7.0)
        val result = leveneTest(g1, g2, center = LeveneCenter.MEAN)
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    @Test
    fun constantGroupsMedian() {
        val g1 = doubleArrayOf(3.0, 3.0, 3.0)
        val g2 = doubleArrayOf(7.0, 7.0, 7.0)
        val result = leveneTest(g1, g2, center = LeveneCenter.MEDIAN)
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    @Test
    fun fewerThan2Groups() {
        assertFailsWith<InsufficientDataException> {
            leveneTest(doubleArrayOf(1.0, 2.0))
        }
    }

    @Test
    fun singleElementGroup() {
        assertFailsWith<InsufficientDataException> {
            leveneTest(doubleArrayOf(1.0), doubleArrayOf(2.0, 3.0))
        }
    }

    @Test
    fun emptyGroup() {
        assertFailsWith<InsufficientDataException> {
            leveneTest(doubleArrayOf(), doubleArrayOf(2.0, 3.0))
        }
    }

    @Test
    fun zeroGroups() {
        assertFailsWith<InsufficientDataException> {
            leveneTest()
        }
    }

    // ── 4. Extreme cases ───────────────────────────────────────────────────

    @Test
    fun largeSamplesVeryDifferentVariances() {
        // Construct deterministic groups: g1 has SD≈1, g2 has SD≈100
        // Group 1: values around 0 with small spread
        val g1 = DoubleArray(50) { i -> (i - 25.0) / 25.0 }  // range [-1, 0.96]
        // Group 2: values around 0 with large spread
        val g2 = DoubleArray(50) { i -> (i - 25.0) * 4.0 }   // range [-100, 96]
        val result = leveneTest(g1, g2, center = LeveneCenter.MEAN)
        assertTrue(result.statistic > 50.0, "Very different variances should give large F")
        assertTrue(result.pValue < 0.001, "Very different variances should be highly significant")
    }

    @Test
    fun largeSamplesEqualVariances() {
        // Same spread, different location
        val g1 = DoubleArray(100) { i -> i.toDouble() }
        val g2 = DoubleArray(100) { i -> i.toDouble() + 1000.0 }
        val result = leveneTest(g1, g2, center = LeveneCenter.MEAN)
        assertEquals(0.0, result.statistic, 1e-6)
        assertTrue(result.pValue > 0.99, "Equal variances should have p-value near 1")
    }

    // ── 5. Non-finite input ────────────────────────────────────────────────

    @Test
    fun nanInGroup() {
        val g1 = doubleArrayOf(1.0, 2.0, Double.NaN, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = leveneTest(g1, g2, center = LeveneCenter.MEAN)
        assertTrue(result.pValue.isNaN(), "NaN in input should produce NaN p-value")
    }

    @Test
    fun infinityInGroup() {
        val g1 = doubleArrayOf(1.0, 2.0, Double.POSITIVE_INFINITY, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = leveneTest(g1, g2, center = LeveneCenter.MEAN)
        assertTrue(result.pValue.isNaN(), "Infinity in input should produce NaN p-value")
    }

    @Test
    fun nanInGroupMedian() {
        val g1 = doubleArrayOf(1.0, 2.0, Double.NaN, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = leveneTest(g1, g2, center = LeveneCenter.MEDIAN)
        assertTrue(result.pValue.isNaN(), "NaN in input should produce NaN p-value")
    }
}
