package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.*

class FriedmanTestTest {

    private val tolStat = 1e-10
    private val tolP = 1e-3

    // ── 1. Basic correctness (scipy reference values) ──────────────────────

    @Test
    fun threeGroups() {
        // scipy: friedmanchisquare([7,9.8,6.5,7.2,8.3], [5.4,6.8,5,4.8,6.1], [8.2,10.5,7.1,8.5,9])
        //   → (10.0, 0.00673794699908547)
        val g1 = doubleArrayOf(7.0, 9.8, 6.5, 7.2, 8.3)
        val g2 = doubleArrayOf(5.4, 6.8, 5.0, 4.8, 6.1)
        val g3 = doubleArrayOf(8.2, 10.5, 7.1, 8.5, 9.0)
        val result = friedmanTest(g1, g2, g3)
        assertEquals("Friedman Test", result.testName)
        assertEquals(10.0, result.statistic, tolStat)
        assertEquals(0.00673794699908547, result.pValue, tolP)
        assertEquals(2.0, result.degreesOfFreedom)
    }

    @Test
    fun fiveGroups() {
        // scipy: friedmanchisquare → (8.48, 0.0754957812579086)
        val g1 = doubleArrayOf(3.0, 4.0, 2.0, 5.0, 1.0)
        val g2 = doubleArrayOf(4.0, 5.0, 3.0, 6.0, 2.0)
        val g3 = doubleArrayOf(2.0, 3.0, 1.0, 4.0, 3.0)
        val g4 = doubleArrayOf(5.0, 6.0, 4.0, 7.0, 4.0)
        val g5 = doubleArrayOf(1.0, 2.0, 5.0, 3.0, 5.0)
        val result = friedmanTest(g1, g2, g3, g4, g5)
        assertEquals(8.48, result.statistic, tolStat)
        assertEquals(0.0754957812579086, result.pValue, tolP)
        assertEquals(4.0, result.degreesOfFreedom)
    }

    @Test
    fun groupsWithTies() {
        // scipy: friedmanchisquare([1,2,3,4,5],[1,2,3,4,5],[5,4,3,2,1]) → (0.0, 1.0)
        val g = TestData.SEQUENTIAL_1_5
        val result = friedmanTest(g, g.copyOf(), doubleArrayOf(5.0, 4.0, 3.0, 2.0, 1.0))
        assertEquals(0.0, result.statistic, tolStat)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun threeGroupsClearSeparation() {
        // scipy: friedmanchisquare([1..8], [9..16], [17..24]) → (16.0, 0.000335462627902512)
        val g1 = DoubleArray(8) { (it + 1).toDouble() }
        val g2 = DoubleArray(8) { (it + 9).toDouble() }
        val g3 = DoubleArray(8) { (it + 17).toDouble() }
        val result = friedmanTest(g1, g2, g3)
        assertEquals(16.0, result.statistic, tolStat)
        assertEquals(0.000335462627902512, result.pValue, tolP)
    }

    @Test
    fun threeGroupsNegativeValues() {
        // scipy: friedmanchisquare([-5,-3,-1,0,2],[1,3,5,7,9],[-2,0,1,3,4])
        //   → (10.0, 0.00673794699908547)
        val g1 = doubleArrayOf(-5.0, -3.0, -1.0, 0.0, 2.0)
        val g2 = doubleArrayOf(1.0, 3.0, 5.0, 7.0, 9.0)
        val g3 = doubleArrayOf(-2.0, 0.0, 1.0, 3.0, 4.0)
        val result = friedmanTest(g1, g2, g3)
        assertEquals(10.0, result.statistic, tolStat)
        assertEquals(0.00673794699908547, result.pValue, tolP)
    }

    @Test
    fun fourGroups() {
        // scipy: friedmanchisquare → (16.4, 0.000938742055045038)
        val g1 = doubleArrayOf(2.0, 6.0, 4.0, 8.0, 3.0, 5.0)
        val g2 = doubleArrayOf(3.0, 5.0, 3.0, 7.0, 2.0, 6.0)
        val g3 = doubleArrayOf(7.0, 9.0, 8.0, 12.0, 6.0, 10.0)
        val g4 = doubleArrayOf(1.0, 4.0, 2.0, 6.0, 1.0, 3.0)
        val result = friedmanTest(g1, g2, g3, g4)
        assertEquals(16.4, result.statistic, tolStat)
        assertEquals(0.000938742055045038, result.pValue, tolP)
        assertEquals(3.0, result.degreesOfFreedom)
    }

    @Test
    fun testNameAndAdditionalInfo() {
        val g1 = doubleArrayOf(7.0, 9.8, 6.5, 7.2, 8.3)
        val g2 = doubleArrayOf(5.4, 6.8, 5.0, 4.8, 6.1)
        val g3 = doubleArrayOf(8.2, 10.5, 7.1, 8.5, 9.0)
        val result = friedmanTest(g1, g2, g3)
        assertEquals("Friedman Test", result.testName)
        assertEquals(3.0, result.additionalInfo["numGroups"])
        assertEquals(5.0, result.additionalInfo["numBlocks"])
    }

    // ── 2. Edge cases ──────────────────────────────────────────────────────

    @Test
    fun minimumValidK2N2() {
        val result = friedmanTest(doubleArrayOf(10.0, 20.0), doubleArrayOf(30.0, 40.0))
        assertEquals(2.0, result.statistic, tolStat)
        assertEquals(1.0, result.degreesOfFreedom)
        assertTrue(result.pValue > 0.1)
    }

    @Test
    fun minimumValidK3N2() {
        // scipy: friedmanchisquare([10,20],[30,40],[50,60]) → (4.0, 0.135335283236613)
        val result = friedmanTest(
            doubleArrayOf(10.0, 20.0), doubleArrayOf(30.0, 40.0), doubleArrayOf(50.0, 60.0)
        )
        assertEquals(4.0, result.statistic, tolStat)
        assertEquals(0.135335283236613, result.pValue, tolP)
    }

    @Test
    fun largeN() {
        val n = 50
        val g1 = TestData.largeSequential(n)
        val g2 = TestData.largeSequential(n, offset = 100.0)
        val g3 = TestData.largeSequential(n, offset = 200.0)
        val result = friedmanTest(g1, g2, g3)
        assertTrue(result.statistic > 50.0, "Large n with clear separation should give large statistic")
        assertTrue(result.pValue < 1e-10, "Large n with clear separation should be highly significant")
    }

    @Test
    fun dfVerification() {
        val result = friedmanTest(
            TestData.SHORT_3, doubleArrayOf(4.0, 5.0, 6.0),
            doubleArrayOf(7.0, 8.0, 9.0), doubleArrayOf(10.0, 11.0, 12.0),
            doubleArrayOf(13.0, 14.0, 15.0)
        )
        assertEquals(4.0, result.degreesOfFreedom) // k-1 = 5-1 = 4
    }

    @Test
    fun twoGroupsWithTies() {
        val g1 = doubleArrayOf(1.0, 1.0, 1.0, 3.0, 5.0)
        val g2 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = friedmanTest(g1, g2)
        assertEquals(3.0, result.statistic, tolStat)
        assertEquals(0.08326, result.pValue, tolP)
    }

    // ── 3. Degenerate cases ────────────────────────────────────────────────

    @Test
    fun allIdenticalValues() {
        val c = doubleArrayOf(5.0, 5.0, 5.0, 5.0)
        val result = friedmanTest(c, c.copyOf(), c.copyOf())
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    @Test
    fun constantWithinBlocks() {
        val g = TestData.SEQUENTIAL_1_5.sliceArray(0 until 4)
        val result = friedmanTest(g, g.copyOf(), g.copyOf())
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    @Test
    fun identicalGroups() {
        val g = TestData.SEQUENTIAL_1_5
        val result = friedmanTest(g, g.copyOf(), g.copyOf())
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    // ── 4. Input validation ────────────────────────────────────────────────

    @Test
    fun fewerThan2Groups() {
        assertFailsWith<InsufficientDataException> { friedmanTest(TestData.SHORT_3) }
    }

    @Test
    fun zeroGroups() {
        assertFailsWith<InsufficientDataException> {
            @Suppress("SpreadOperator") friedmanTest()
        }
    }

    @Test
    fun unequalGroupSizes() {
        assertFailsWith<InvalidParameterException> {
            friedmanTest(TestData.SHORT_3, doubleArrayOf(4.0, 5.0), TestData.SHORT_3)
        }
    }

    @Test
    fun blockSizeLessThan2() {
        assertFailsWith<InsufficientDataException> {
            friedmanTest(doubleArrayOf(1.0), doubleArrayOf(2.0), doubleArrayOf(3.0))
        }
    }

    @Test
    fun emptyGroups() {
        assertFailsWith<InsufficientDataException> {
            friedmanTest(doubleArrayOf(), doubleArrayOf(), doubleArrayOf())
        }
    }

    // ── 5. Non-finite input ────────────────────────────────────────────────

    @Test
    fun nanInGroup() {
        val result = friedmanTest(TestData.WITH_NAN, TestData.SEQUENTIAL_6_10, TestData.SEQUENTIAL_11_15)
        assertTrue(result.pValue.isNaN(), "NaN in input should produce NaN p-value")
    }

    @Test
    fun infinityInGroup() {
        val result = friedmanTest(TestData.WITH_POS_INF, TestData.SEQUENTIAL_6_10, TestData.SEQUENTIAL_11_15)
        assertTrue(result.pValue.isNaN(), "Infinity in input should produce NaN p-value")
    }

    // ── 6. Property-based ──────────────────────────────────────────────────

    @Test
    fun isSignificantConsistency() {
        val g1 = doubleArrayOf(7.0, 9.8, 6.5, 7.2, 8.3)
        val g2 = doubleArrayOf(5.4, 6.8, 5.0, 4.8, 6.1)
        val g3 = doubleArrayOf(8.2, 10.5, 7.1, 8.5, 9.0)
        val result = friedmanTest(g1, g2, g3)
        // p ≈ 0.0067
        assertTrue(result.isSignificant(0.05))
        assertTrue(result.isSignificant(0.01))
        assertFalse(result.isSignificant(0.001))
    }

    @Test
    fun increasingDivergenceDecreasingPValue() {
        val g1 = DoubleArray(8) { (it + 1).toDouble() }
        // Moderate effect
        val g2 = doubleArrayOf(2.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0)
        val g3 = doubleArrayOf(3.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0)
        val r1 = friedmanTest(g1, g2, g3)

        // Strong effect
        val g4 = DoubleArray(8) { (it + 11).toDouble() }
        val g5 = DoubleArray(8) { (it + 21).toDouble() }
        val r2 = friedmanTest(g1, g4, g5)

        assertTrue(r2.statistic > r1.statistic, "Larger treatment divergence → larger statistic")
        assertTrue(r2.pValue < r1.pValue, "Larger treatment divergence → smaller p-value")
    }

    @Test
    fun statisticNonNegative() {
        val g1 = doubleArrayOf(3.0, 1.0, 5.0, 2.0, 4.0)
        val g2 = doubleArrayOf(4.0, 2.0, 3.0, 5.0, 1.0)
        val g3 = doubleArrayOf(5.0, 3.0, 1.0, 4.0, 2.0)
        val result = friedmanTest(g1, g2, g3)
        assertTrue(result.statistic >= 0.0, "Friedman statistic should be non-negative")
        assertTrue(result.pValue in 0.0..1.0, "p-value should be in [0, 1]")
    }
}
