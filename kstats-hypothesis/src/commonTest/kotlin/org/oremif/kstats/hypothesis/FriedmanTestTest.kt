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
        // scipy: friedmanchisquare([3,4,2,5,1],[4,5,3,6,2],[2,3,1,4,3],[5,6,4,7,4],[1,2,5,3,5])
        //   → (8.48, 0.0754957812579086)
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
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g3 = doubleArrayOf(5.0, 4.0, 3.0, 2.0, 1.0)
        val result = friedmanTest(g1, g2, g3)
        assertEquals(0.0, result.statistic, tolStat)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun threeGroupsClearSeparation() {
        // scipy: friedmanchisquare([1..8], [9..16], [17..24]) → (16.0, 0.000335462627902512)
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        val g2 = doubleArrayOf(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
        val g3 = doubleArrayOf(17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0)
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
        // scipy: friedmanchisquare([2,6,4,8,3,5],[3,5,3,7,2,6],[7,9,8,12,6,10],[1,4,2,6,1,3])
        //   → (16.4, 0.000938742055045038)
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
        // k=2, n=2: g1=[10,20], g2=[30,40]
        // Block 0: [10,30] → ranks [1,2]; Block 1: [20,40] → ranks [1,2]
        // R_1=2, R_2=4; Q = (12/(2*2*3))*(4+16) - 3*2*3 = (1.0)*20 - 18 = 2.0
        // p = ChiSq(1).sf(2.0) ≈ 0.15730
        val g1 = doubleArrayOf(10.0, 20.0)
        val g2 = doubleArrayOf(30.0, 40.0)
        val result = friedmanTest(g1, g2)
        assertEquals(2.0, result.statistic, tolStat)
        assertEquals(1.0, result.degreesOfFreedom)
        assertTrue(result.pValue > 0.1)
    }

    @Test
    fun minimumValidK3N2() {
        // scipy: friedmanchisquare([10,20],[30,40],[50,60]) → (4.0, 0.135335283236613)
        val g1 = doubleArrayOf(10.0, 20.0)
        val g2 = doubleArrayOf(30.0, 40.0)
        val g3 = doubleArrayOf(50.0, 60.0)
        val result = friedmanTest(g1, g2, g3)
        assertEquals(4.0, result.statistic, tolStat)
        assertEquals(0.135335283236613, result.pValue, tolP)
    }

    @Test
    fun largeN() {
        // Large number of blocks (n=50) with clear treatment effect
        val n = 50
        val g1 = DoubleArray(n) { i -> i.toDouble() }
        val g2 = DoubleArray(n) { i -> i.toDouble() + 100.0 }
        val g3 = DoubleArray(n) { i -> i.toDouble() + 200.0 }
        val result = friedmanTest(g1, g2, g3)
        assertTrue(result.statistic > 50.0, "Large n with clear separation should give large statistic")
        assertTrue(result.pValue < 1e-10, "Large n with clear separation should be highly significant")
    }

    @Test
    fun dfVerification() {
        val g1 = doubleArrayOf(1.0, 2.0, 3.0)
        val g2 = doubleArrayOf(4.0, 5.0, 6.0)
        val g3 = doubleArrayOf(7.0, 8.0, 9.0)
        val g4 = doubleArrayOf(10.0, 11.0, 12.0)
        val g5 = doubleArrayOf(13.0, 14.0, 15.0)
        val result = friedmanTest(g1, g2, g3, g4, g5)
        assertEquals(4.0, result.degreesOfFreedom) // k-1 = 5-1 = 4
    }

    @Test
    fun twoGroupsWithTies() {
        // k=2 with ties within blocks
        // Block 0: [1,1] → ranks [1.5,1.5]; Block 1: [1,2] → ranks [1,2]
        // Block 2: [1,3] → ranks [1,2]; Block 3: [3,4] → ranks [1,2]
        // Block 4: [5,5] → ranks [1.5,1.5]
        // R_1=6, R_2=9; Q_raw = (12/(5*2*3))*(36+81)-45 = 1.8
        // Tie correction: blocks 0,4 each have t=2 (tieSum = 2*(8-2)=12)
        // C = 1 - 12/(5*2*(4-1)) = 1 - 12/30 = 0.6
        // Q_corrected = 1.8 / 0.6 = 3.0
        val g1 = doubleArrayOf(1.0, 1.0, 1.0, 3.0, 5.0)
        val g2 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = friedmanTest(g1, g2)
        assertEquals(3.0, result.statistic, tolStat)
        // chi2.sf(3.0, 1) ≈ 0.08326
        assertEquals(0.08326, result.pValue, tolP)
    }

    // ── 3. Degenerate cases ────────────────────────────────────────────────

    @Test
    fun allIdenticalValues() {
        // All groups constant with same value → all rank sums equal → stat=0, p=1
        val g1 = doubleArrayOf(5.0, 5.0, 5.0, 5.0)
        val g2 = doubleArrayOf(5.0, 5.0, 5.0, 5.0)
        val g3 = doubleArrayOf(5.0, 5.0, 5.0, 5.0)
        val result = friedmanTest(g1, g2, g3)
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    @Test
    fun constantWithinBlocks() {
        // Each block has the same value across treatments (but different across blocks)
        // → all rank sums equal → stat=0, p=1
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val g2 = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val g3 = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val result = friedmanTest(g1, g2, g3)
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    @Test
    fun identicalGroups() {
        // All groups are copies of the same data → rank sums equal → stat=0, p=1
        val g = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = friedmanTest(g, g.copyOf(), g.copyOf())
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    // ── 4. Input validation ────────────────────────────────────────────────

    @Test
    fun fewerThan2Groups() {
        assertFailsWith<InsufficientDataException> {
            friedmanTest(doubleArrayOf(1.0, 2.0, 3.0))
        }
    }

    @Test
    fun zeroGroups() {
        assertFailsWith<InsufficientDataException> {
            @Suppress("SpreadOperator")
            friedmanTest()
        }
    }

    @Test
    fun unequalGroupSizes() {
        assertFailsWith<InvalidParameterException> {
            friedmanTest(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0),
                doubleArrayOf(6.0, 7.0, 8.0)
            )
        }
    }

    @Test
    fun blockSizeLessThan2() {
        assertFailsWith<InsufficientDataException> {
            friedmanTest(
                doubleArrayOf(1.0),
                doubleArrayOf(2.0),
                doubleArrayOf(3.0)
            )
        }
    }

    @Test
    fun emptyGroups() {
        assertFailsWith<InsufficientDataException> {
            friedmanTest(
                doubleArrayOf(),
                doubleArrayOf(),
                doubleArrayOf()
            )
        }
    }

    // ── 5. Non-finite input ────────────────────────────────────────────────

    @Test
    fun nanInGroup() {
        val g1 = doubleArrayOf(1.0, 2.0, Double.NaN, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
        val result = friedmanTest(g1, g2, g3)
        assertTrue(result.pValue.isNaN(), "NaN in input should produce NaN p-value")
    }

    @Test
    fun infinityInGroup() {
        val g1 = doubleArrayOf(1.0, 2.0, Double.POSITIVE_INFINITY, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
        val result = friedmanTest(g1, g2, g3)
        assertTrue(result.pValue.isNaN(), "Infinity in input should produce NaN p-value")
    }

    // ── 6. Property-based ──────────────────────────────────────────────────

    @Test
    fun isSignificantConsistency() {
        val g1 = doubleArrayOf(7.0, 9.8, 6.5, 7.2, 8.3)
        val g2 = doubleArrayOf(5.4, 6.8, 5.0, 4.8, 6.1)
        val g3 = doubleArrayOf(8.2, 10.5, 7.1, 8.5, 9.0)
        val result = friedmanTest(g1, g2, g3)
        // p ≈ 0.0067, should be significant at alpha=0.05 and 0.01
        assertTrue(result.isSignificant(0.05))
        assertTrue(result.isSignificant(0.01))
        assertFalse(result.isSignificant(0.001))
    }

    @Test
    fun increasingDivergenceDecreasingPValue() {
        // Moderate effect: treatments partially overlap within blocks
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        val g2 = doubleArrayOf(2.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0)
        val g3 = doubleArrayOf(3.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0)
        val r1 = friedmanTest(g1, g2, g3)

        // Strong effect: treatments clearly separated in every block
        val g4 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        val g5 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0)
        val g6 = doubleArrayOf(21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0)
        val r2 = friedmanTest(g4, g5, g6)

        assertTrue(
            r2.statistic > r1.statistic,
            "Larger treatment divergence should give larger statistic"
        )
        assertTrue(
            r2.pValue < r1.pValue,
            "Larger treatment divergence should give smaller p-value"
        )
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
