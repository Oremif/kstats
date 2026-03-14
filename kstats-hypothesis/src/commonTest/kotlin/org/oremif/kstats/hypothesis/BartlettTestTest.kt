package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.test.*

class BartlettTestTest {

    private val tolStat = 1e-10
    private val tolP = 1e-3

    // ── 1. Basic correctness (scipy reference values) ──────────────────────

    @Test
    fun twoGroupsEqualVariance() {
        // scipy: bartlett([1,2,3,4,5], [6,7,8,9,10]) → (0.0, 1.0)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = bartlettTest(g1, g2)
        assertEquals("Bartlett's Test", result.testName)
        assertEquals(0.0, result.statistic, tolStat)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun twoGroupsDifferentVariance() {
        // scipy: bartlett([10,11,12,9,10], [5,15,10,20,0]) → (8.987310927544002, 0.002718607642979)
        val g1 = doubleArrayOf(10.0, 11.0, 12.0, 9.0, 10.0)
        val g2 = doubleArrayOf(5.0, 15.0, 10.0, 20.0, 0.0)
        val result = bartlettTest(g1, g2)
        assertEquals(8.987310927544002, result.statistic, tolStat)
        assertEquals(0.002718607642979, result.pValue, tolP)
    }

    @Test
    fun threeGroupsEqualVariance() {
        // scipy: bartlett([1..5], [6..10], [11..15]) → (0.0, 1.0)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
        val result = bartlettTest(g1, g2, g3)
        assertEquals(0.0, result.statistic, tolStat)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun threeGroupsDifferentVariance() {
        // scipy: bartlett([10,11,12,9,10], [5,15,10,20,0], [8,9,10,11,12])
        //   → (14.302480580953628, 0.000783891225302)
        val g1 = doubleArrayOf(10.0, 11.0, 12.0, 9.0, 10.0)
        val g2 = doubleArrayOf(5.0, 15.0, 10.0, 20.0, 0.0)
        val g3 = doubleArrayOf(8.0, 9.0, 10.0, 11.0, 12.0)
        val result = bartlettTest(g1, g2, g3)
        assertEquals(14.302480580953628, result.statistic, tolStat)
        assertEquals(0.000783891225302, result.pValue, tolP)
    }

    @Test
    fun fiveGroups() {
        // scipy: bartlett([1..5], [2,4,6,8,10], [1,3,5,7,9], [3,6,9,12,15], [0.5,1,1.5,2,2.5])
        //   → (10.509515998239600, 0.032666161769369)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val g3 = doubleArrayOf(1.0, 3.0, 5.0, 7.0, 9.0)
        val g4 = doubleArrayOf(3.0, 6.0, 9.0, 12.0, 15.0)
        val g5 = doubleArrayOf(0.5, 1.0, 1.5, 2.0, 2.5)
        val result = bartlettTest(g1, g2, g3, g4, g5)
        assertEquals(10.509515998239600, result.statistic, tolStat)
        assertEquals(0.032666161769369, result.pValue, tolP)
    }

    @Test
    fun unequalGroupSizes() {
        // scipy: bartlett([1,2,3], [4,5,6,7,8]) → (0.413347107234182, 0.520275516199639)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0)
        val g2 = doubleArrayOf(4.0, 5.0, 6.0, 7.0, 8.0)
        val result = bartlettTest(g1, g2)
        assertEquals(0.413347107234182, result.statistic, tolStat)
        assertEquals(0.520275516199639, result.pValue, tolP)
    }

    // ── 2. Edge cases ──────────────────────────────────────────────────────

    @Test
    fun minimumSize2x2() {
        // scipy: bartlett([1,3], [2,8]) → (0.681100831687988, 0.409207914283647)
        val g1 = doubleArrayOf(1.0, 3.0)
        val g2 = doubleArrayOf(2.0, 8.0)
        val result = bartlettTest(g1, g2)
        assertEquals(0.681100831687988, result.statistic, tolStat)
        assertEquals(0.409207914283647, result.pValue, tolP)
    }

    @Test
    fun largeGroupsDifferentVariance() {
        val g1 = DoubleArray(50) { i -> (i - 25.0) / 25.0 }
        val g2 = DoubleArray(50) { i -> (i - 25.0) * 4.0 }
        val result = bartlettTest(g1, g2)
        // scipy: stat=379.515447435506189, p=0.0
        assertEquals(379.515447435506189, result.statistic, 1e-6)
        assertTrue(result.pValue < 1e-15, "Very different variances should be highly significant")
    }

    @Test
    fun largeGroupsEqualVariance() {
        val g1 = DoubleArray(100) { i -> i.toDouble() }
        val g2 = DoubleArray(100) { i -> i.toDouble() + 1000.0 }
        val result = bartlettTest(g1, g2)
        assertEquals(0.0, result.statistic, 1e-6)
        assertTrue(result.pValue > 0.99, "Equal variances should have p-value near 1")
    }

    @Test
    fun dfAndPooledVarianceVerification() {
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
        val result = bartlettTest(g1, g2, g3)
        assertEquals(2.0, result.degreesOfFreedom) // k-1 = 3-1 = 2
        // All groups have variance 2.5, so pooled variance = 2.5
        assertEquals(2.5, result.additionalInfo["pooledVariance"]!!, 1e-10)
    }

    // ── 3. Degenerate cases ────────────────────────────────────────────────

    @Test
    fun allConstantSameValue() {
        // All groups constant with same value → variances all zero → stat=0, p=1
        val g1 = doubleArrayOf(5.0, 5.0, 5.0)
        val g2 = doubleArrayOf(5.0, 5.0, 5.0)
        val result = bartlettTest(g1, g2)
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    @Test
    fun allConstantDifferentValues() {
        // Groups constant with different values → variances all zero → stat=0, p=1
        val g1 = doubleArrayOf(3.0, 3.0, 3.0)
        val g2 = doubleArrayOf(7.0, 7.0, 7.0)
        val result = bartlettTest(g1, g2)
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    @Test
    fun someConstantSomeNot() {
        // One group constant (var=0), one not → ln(0)=-Inf → T=Inf, p=0
        val g1 = doubleArrayOf(5.0, 5.0, 5.0)
        val g2 = doubleArrayOf(1.0, 2.0, 3.0)
        val result = bartlettTest(g1, g2)
        assertTrue(result.statistic.isInfinite(), "Statistic should be Inf when one group is constant")
        assertEquals(0.0, result.pValue, tolP)
    }

    // ── 4. Input validation ────────────────────────────────────────────────

    @Test
    fun fewerThan2Groups() {
        assertFailsWith<InsufficientDataException> {
            bartlettTest(doubleArrayOf(1.0, 2.0))
        }
    }

    @Test
    fun zeroGroups() {
        assertFailsWith<InsufficientDataException> {
            @Suppress("SpreadOperator")
            bartlettTest()
        }
    }

    @Test
    fun groupSizeLessThan2() {
        assertFailsWith<InsufficientDataException> {
            bartlettTest(doubleArrayOf(1.0), doubleArrayOf(2.0, 3.0))
        }
    }

    @Test
    fun emptyGroup() {
        assertFailsWith<InsufficientDataException> {
            bartlettTest(doubleArrayOf(), doubleArrayOf(2.0, 3.0))
        }
    }

    // ── 5. Non-finite input ────────────────────────────────────────────────

    @Test
    fun nanInGroup() {
        val g1 = doubleArrayOf(1.0, 2.0, Double.NaN, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = bartlettTest(g1, g2)
        assertTrue(result.pValue.isNaN(), "NaN in input should produce NaN p-value")
    }

    @Test
    fun infinityInGroup() {
        val g1 = doubleArrayOf(1.0, 2.0, Double.POSITIVE_INFINITY, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = bartlettTest(g1, g2)
        assertTrue(result.pValue.isNaN(), "Infinity in input should produce NaN p-value")
    }

    // ── 6. Property-based ──────────────────────────────────────────────────

    @Test
    fun identicalGroupsStatisticNearZero() {
        val data = doubleArrayOf(1.0, 3.0, 5.0, 7.0, 9.0)
        val result = bartlettTest(data, data.copyOf(), data.copyOf())
        assertEquals(0.0, result.statistic, 1e-10)
        assertTrue(result.pValue > 0.99, "Identical groups should have p-value near 1")
    }

    @Test
    fun increasingVarianceDivergenceDecreasingPValue() {
        val baseline = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0)

        // Moderate spread
        val moderate = doubleArrayOf(5.0, 10.0, 15.0, 20.0, 25.0)
        val r1 = bartlettTest(baseline, moderate)

        // Large spread
        val large = doubleArrayOf(0.0, 10.0, 20.0, 30.0, 40.0)
        val r2 = bartlettTest(baseline, large)

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
        val g1 = doubleArrayOf(10.0, 11.0, 12.0, 9.0, 10.0)
        val g2 = doubleArrayOf(5.0, 15.0, 10.0, 20.0, 0.0)
        val result = bartlettTest(g1, g2)
        // p ≈ 0.0027, should be significant at alpha=0.05
        assertTrue(result.isSignificant(0.05))
        assertTrue(result.isSignificant(0.01))
        assertFalse(result.isSignificant(0.001))
    }
}
