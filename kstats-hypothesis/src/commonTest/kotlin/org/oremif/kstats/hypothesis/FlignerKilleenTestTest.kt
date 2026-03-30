package org.oremif.kstats.hypothesis

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FlignerKilleenTestTest : AbstractVarianceTestTest() {

    override fun runTest(vararg groups: DoubleArray): TestResult = flignerKilleenTest(*groups)

    // ── Fligner-specific scipy golden values ───────────────────────────────

    @Test
    fun testName() {
        val result = flignerKilleenTest(TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10)
        assertEquals("Fligner-Killeen Test", result.testName)
    }

    @Test
    fun twoGroupsDifferentVariance() {
        // scipy: fligner([10,11,12,9,10], [5,15,10,20,0]) → (3.6283, 0.05680)
        val result = flignerKilleenTest(TestData.LOW_VARIANCE, TestData.HIGH_VARIANCE)
        assertEquals(3.628298116075668, result.statistic, tolStat)
        assertEquals(0.05680487660778472, result.pValue, tolP)
    }

    @Test
    fun threeGroupsDifferentVariance() {
        // scipy: fligner([10,11,12,9,10], [5,15,10,20,0], [8,9,10,11,12])
        //   → (5.8192, 0.05450)
        val result = flignerKilleenTest(TestData.LOW_VARIANCE, TestData.HIGH_VARIANCE, TestData.MEDIUM_VARIANCE)
        assertEquals(5.819209146478717, result.statistic, tolStat)
        assertEquals(0.05449727529012288, result.pValue, tolP)
    }

    @Test
    fun fiveGroups() {
        // scipy: fligner([1..5], [2,4,6,8,10], [1,3,5,7,9], [3,6,9,12,15], [0.5,1,1.5,2,2.5])
        //   → (8.0778, 0.08877)
        val result = flignerKilleenTest(
            TestData.SEQUENTIAL_1_5, TestData.EVEN_SPREAD, TestData.ODD_SPREAD,
            TestData.TRIPLE_SPREAD, TestData.HALF_SPREAD
        )
        assertEquals(8.077844101257757, result.statistic, tolStat)
        assertEquals(0.08876792756759197, result.pValue, tolP)
    }

    // ── Fligner-specific edge cases ────────────────────────────────────────

    @Test
    fun minimumSize2x2() {
        // scipy: fligner([1,3], [2,8]) → (3.0, 0.08326)
        val result = flignerKilleenTest(TestData.MIN_A, TestData.MIN_B)
        assertEquals(3.0, result.statistic, 1e-6)
        assertEquals(0.08326451666355052, result.pValue, tolP)
    }

    @Test
    override fun unequalGroupSizes() {
        // scipy: fligner([1,2,3], [4,5,6,7,8]) → (1.1138, 0.29125)
        val result = flignerKilleenTest(TestData.SHORT_3, TestData.LONG_5)
        assertEquals(1.113841119690382, result.statistic, tolStat)
        assertEquals(0.291248495959798, result.pValue, tolP)
    }

    @Test
    fun largeGroupsDifferentVariance() {
        val result = flignerKilleenTest(TestData.largeSmallSpread(), TestData.largeLargeSpread())
        // scipy: stat=63.2299, p≈1.84e-15
        assertEquals(63.2298731493289, result.statistic, 1e-6)
        assertTrue(result.pValue < 1e-12, "Very different variances should be highly significant")
    }

    @Test
    fun dfVerification() {
        val result = flignerKilleenTest(TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10, TestData.SEQUENTIAL_11_15)
        assertEquals(2.0, result.degreesOfFreedom) // k-1 = 3-1 = 2
    }

    @Test
    fun increasingVarianceDivergenceDecreasingPValue() {
        val small = doubleArrayOf(8.0, 10.0, 12.0, 14.0, 16.0)
        val r1 = flignerKilleenTest(TestData.BASELINE, small)

        val large = doubleArrayOf(0.0, 5.0, 12.0, 20.0, 30.0)
        val r2 = flignerKilleenTest(TestData.BASELINE, large)

        assertTrue(r2.statistic > r1.statistic, "Larger divergence → larger statistic")
        assertTrue(r2.pValue < r1.pValue, "Larger divergence → smaller p-value")
    }
}
