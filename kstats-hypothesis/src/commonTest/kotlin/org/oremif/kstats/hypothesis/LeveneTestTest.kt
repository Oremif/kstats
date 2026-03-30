package org.oremif.kstats.hypothesis

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class LeveneTestTest : AbstractVarianceTestTest() {

    override fun runTest(vararg groups: DoubleArray): TestResult =
        leveneTest(*groups, center = LeveneCenter.MEAN)

    // ── Levene-specific: center variants ───────────────────────────────────

    @Test
    fun testNameMean() {
        val result = leveneTest(TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10, center = LeveneCenter.MEAN)
        assertEquals("Levene's Test", result.testName)
    }

    @Test
    fun testNameMedian() {
        val result = leveneTest(TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10, center = LeveneCenter.MEDIAN)
        assertEquals("Brown-Forsythe Test", result.testName)
    }

    @Test
    fun twoGroupMedianCenter() {
        // scipy: levene([1,2,3,4,5], [6,7,8,9,10], center='median') → (0.0, 1.0)
        val result = leveneTest(TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10, center = LeveneCenter.MEDIAN)
        assertEquals(0.0, result.statistic, tolStat)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun threeGroupMedianCenter() {
        // scipy: levene([1..5], [6..10], [11..15], center='median') → (0.0, 1.0)
        val result = leveneTest(
            TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10, TestData.SEQUENTIAL_11_15,
            center = LeveneCenter.MEDIAN
        )
        assertEquals(0.0, result.statistic, tolStat)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun defaultCenterIsMedian() {
        val defaultResult = leveneTest(TestData.LOW_VARIANCE, TestData.HIGH_VARIANCE)
        val medianResult = leveneTest(TestData.LOW_VARIANCE, TestData.HIGH_VARIANCE, center = LeveneCenter.MEDIAN)
        assertEquals(medianResult.statistic, defaultResult.statistic, 0.0)
        assertEquals(medianResult.pValue, defaultResult.pValue, 0.0)
        assertEquals("Brown-Forsythe Test", defaultResult.testName)
    }

    // ── Levene-specific scipy golden values ────────────────────────────────

    @Test
    fun threeGroupDifferentVariancesMean() {
        // scipy: levene([10,11,12,9,10], [5,15,10,20,0], [8,9,10,11,12], center='mean')
        //   → (6.658320742499461, 0.011340988949827)
        val result = leveneTest(
            TestData.LOW_VARIANCE, TestData.HIGH_VARIANCE, TestData.MEDIUM_VARIANCE,
            center = LeveneCenter.MEAN
        )
        assertEquals(6.658320742499461, result.statistic, tolStat)
        assertEquals(0.011340988949827, result.pValue, tolP)
    }

    @Test
    fun threeGroupDifferentVariancesMedian() {
        // scipy: levene([10,11,12,9,10], [5,15,10,20,0], [8,9,10,11,12], center='median')
        //   → (6.645502645502647, 0.011410138646959)
        val result = leveneTest(
            TestData.LOW_VARIANCE, TestData.HIGH_VARIANCE, TestData.MEDIUM_VARIANCE,
            center = LeveneCenter.MEDIAN
        )
        assertEquals(6.645502645502647, result.statistic, tolStat)
        assertEquals(0.011410138646959, result.pValue, tolP)
    }

    @Test
    fun equalVarianceGroups() {
        // scipy: levene([2,4,6,8,10], [3,5,7,9,11], center='mean') → (0.0, 1.0)
        val g1 = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val g2 = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = leveneTest(g1, g2, center = LeveneCenter.MEAN)
        assertEquals(0.0, result.statistic, tolStat)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun fiveGroupsMean() {
        // scipy: levene([1..5], [2,4,6,8,10], [1,3,5,7,9], [3,6,9,12,15], [0.5,1,1.5,2,2.5], center='mean')
        //   → (2.677103718199608, 0.061616119598530)
        val result = leveneTest(
            TestData.SEQUENTIAL_1_5, TestData.EVEN_SPREAD, TestData.ODD_SPREAD,
            TestData.TRIPLE_SPREAD, TestData.HALF_SPREAD,
            center = LeveneCenter.MEAN
        )
        assertEquals(2.677103718199608, result.statistic, tolStat)
        assertEquals(0.061616119598530, result.pValue, tolP)
    }

    @Test
    fun fiveGroupsMedian() {
        // scipy: same groups, center='median' → (2.677103718199608, 0.061616119598530)
        val result = leveneTest(
            TestData.SEQUENTIAL_1_5, TestData.EVEN_SPREAD, TestData.ODD_SPREAD,
            TestData.TRIPLE_SPREAD, TestData.HALF_SPREAD,
            center = LeveneCenter.MEDIAN
        )
        assertEquals(2.677103718199608, result.statistic, tolStat)
        assertEquals(0.061616119598530, result.pValue, tolP)
    }

    // ── Levene-specific edge cases ─────────────────────────────────────────

    @Test
    fun minimumSize2x2() {
        // 2 elements per group: ssWithin=0, ssBetween>0 → W=Inf, p=0
        // scipy: levene([1,3], [2,8], center='mean') → (inf, 0.0)
        val result = leveneTest(TestData.MIN_A, TestData.MIN_B, center = LeveneCenter.MEAN)
        assertTrue(result.statistic.isInfinite(), "W should be infinite when ssWithin=0 and ssBetween>0")
        assertEquals(0.0, result.pValue, tolP)
    }

    @Test
    override fun unequalGroupSizes() {
        // scipy: levene([1,2,3], [4,5,6,7,8], center='mean') → (0.923076923076923, 0.373770542585760)
        val result = leveneTest(TestData.SHORT_3, TestData.LONG_5, center = LeveneCenter.MEAN)
        assertEquals(0.923076923076923, result.statistic, tolStat)
        assertEquals(0.373770542585760, result.pValue, tolP)
    }

    @Test
    fun dfVerification() {
        val result = leveneTest(
            TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10, TestData.SEQUENTIAL_11_15,
            center = LeveneCenter.MEAN
        )
        assertEquals(2.0, result.degreesOfFreedom) // dfBetween = k-1 = 2
        assertEquals(2.0, result.additionalInfo["dfBetween"])
        assertEquals(12.0, result.additionalInfo["dfWithin"]) // N-k = 15-3 = 12
    }

    @Test
    fun constantGroupsMedian() {
        val result = leveneTest(TestData.CONSTANT_3x3, TestData.CONSTANT_7x3, center = LeveneCenter.MEDIAN)
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    @Test
    fun largeSamplesVeryDifferentVariances() {
        val result = leveneTest(TestData.largeSmallSpread(), TestData.largeLargeSpread(), center = LeveneCenter.MEAN)
        assertTrue(result.statistic > 50.0, "Very different variances should give large F")
        assertTrue(result.pValue < 0.001, "Very different variances should be highly significant")
    }

    @Test
    fun nanInGroupMedian() {
        val result = leveneTest(TestData.WITH_NAN, TestData.SEQUENTIAL_6_10, center = LeveneCenter.MEDIAN)
        assertTrue(result.pValue.isNaN(), "NaN in input should produce NaN p-value")
    }
}
