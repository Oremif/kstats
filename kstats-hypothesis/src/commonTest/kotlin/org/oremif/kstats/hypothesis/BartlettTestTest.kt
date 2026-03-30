package org.oremif.kstats.hypothesis

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class BartlettTestTest : AbstractVarianceTestTest() {

    override fun runTest(vararg groups: DoubleArray): TestResult = bartlettTest(*groups)

    // ── Bartlett-specific scipy golden values ──────────────────────────────

    @Test
    fun testName() {
        val result = bartlettTest(TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10)
        assertEquals("Bartlett's Test", result.testName)
    }

    @Test
    fun twoGroupsDifferentVariance() {
        // scipy: bartlett([10,11,12,9,10], [5,15,10,20,0]) → (8.987310927544002, 0.002718607642979)
        val result = bartlettTest(TestData.LOW_VARIANCE, TestData.HIGH_VARIANCE)
        assertEquals(8.987310927544002, result.statistic, tolStat)
        assertEquals(0.002718607642979, result.pValue, tolP)
    }

    @Test
    fun threeGroupsDifferentVariance() {
        // scipy: bartlett([10,11,12,9,10], [5,15,10,20,0], [8,9,10,11,12])
        //   → (14.302480580953628, 0.000783891225302)
        val result = bartlettTest(TestData.LOW_VARIANCE, TestData.HIGH_VARIANCE, TestData.MEDIUM_VARIANCE)
        assertEquals(14.302480580953628, result.statistic, tolStat)
        assertEquals(0.000783891225302, result.pValue, tolP)
    }

    @Test
    fun fiveGroups() {
        // scipy: bartlett([1..5], [2,4,6,8,10], [1,3,5,7,9], [3,6,9,12,15], [0.5,1,1.5,2,2.5])
        //   → (10.509515998239600, 0.032666161769369)
        val result = bartlettTest(
            TestData.SEQUENTIAL_1_5, TestData.EVEN_SPREAD, TestData.ODD_SPREAD,
            TestData.TRIPLE_SPREAD, TestData.HALF_SPREAD
        )
        assertEquals(10.509515998239600, result.statistic, tolStat)
        assertEquals(0.032666161769369, result.pValue, tolP)
    }

    // ── Bartlett-specific edge cases ───────────────────────────────────────

    @Test
    fun minimumSize2x2() {
        // scipy: bartlett([1,3], [2,8]) → (0.681100831687988, 0.409207914283647)
        val result = bartlettTest(TestData.MIN_A, TestData.MIN_B)
        assertEquals(0.681100831687988, result.statistic, tolStat)
        assertEquals(0.409207914283647, result.pValue, tolP)
    }

    @Test
    fun largeGroupsDifferentVariance() {
        val result = bartlettTest(TestData.largeSmallSpread(), TestData.largeLargeSpread())
        // scipy: stat=379.515447435506189, p=0.0
        assertEquals(379.515447435506189, result.statistic, 1e-6)
        assertTrue(result.pValue < 1e-15, "Very different variances should be highly significant")
    }

    @Test
    fun dfAndPooledVarianceVerification() {
        val result = bartlettTest(TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10, TestData.SEQUENTIAL_11_15)
        assertEquals(2.0, result.degreesOfFreedom) // k-1 = 3-1 = 2
        assertEquals(2.5, result.additionalInfo["pooledVariance"]!!, 1e-10)
    }

    @Test
    fun someConstantSomeNot() {
        // One group constant (var=0), one not → ln(0)=-Inf → T=Inf, p=0
        val result = bartlettTest(TestData.CONSTANT_5x3, TestData.SHORT_3)
        assertTrue(result.statistic.isInfinite(), "Statistic should be Inf when one group is constant")
        assertEquals(0.0, result.pValue, tolP)
    }

    @Test
    override fun unequalGroupSizes() {
        // scipy: bartlett([1,2,3], [4,5,6,7,8]) → (0.413347107234182, 0.520275516199639)
        val result = bartlettTest(TestData.SHORT_3, TestData.LONG_5)
        assertEquals(0.413347107234182, result.statistic, tolStat)
        assertEquals(0.520275516199639, result.pValue, tolP)
    }

    @Test
    fun increasingVarianceDivergenceDecreasingPValue() {
        val r1 = bartlettTest(TestData.BASELINE, TestData.MODERATE_SPREAD)
        val r2 = bartlettTest(TestData.BASELINE, TestData.LARGE_SPREAD)
        assertTrue(r2.statistic > r1.statistic, "Larger divergence → larger statistic")
        assertTrue(r2.pValue < r1.pValue, "Larger divergence → smaller p-value")
    }

    @Test
    fun isSignificantAtMultipleAlphas() {
        val result = bartlettTest(TestData.LOW_VARIANCE, TestData.HIGH_VARIANCE)
        // p ≈ 0.0027
        assertTrue(result.isSignificant(0.05))
        assertTrue(result.isSignificant(0.01))
        assertFalse(result.isSignificant(0.001))
    }
}
