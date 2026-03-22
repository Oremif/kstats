package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.test.*

class NonParametricTest {

    // ===== Mann-Whitney U: existing tests =====

    @Test
    fun testMannWhitneyUSignificant() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = mannWhitneyUTest(s1, s2)
        assertTrue(result.isSignificant())
    }

    @Test
    fun testMannWhitneyUNotSignificant() {
        val s1 = doubleArrayOf(1.0, 3.0, 5.0, 7.0, 9.0)
        val s2 = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = mannWhitneyUTest(s1, s2)
        assertFalse(result.isSignificant(), "Interleaved data should not be significant")
    }

    @Test
    fun testMannWhitneyUWithTies() {
        val s1 = doubleArrayOf(1.0, 2.0, 2.0, 3.0, 4.0)
        val s2 = doubleArrayOf(2.0, 3.0, 3.0, 4.0, 5.0)
        val result = mannWhitneyUTest(s1, s2)
        assertTrue(result.pValue in 0.0..1.0)
        assertTrue(result.additionalInfo["z"]!!.isFinite())
    }

    @Test
    fun testMannWhitneyUAllTied() {
        val s1 = doubleArrayOf(5.0, 5.0, 5.0)
        val s2 = doubleArrayOf(5.0, 5.0, 5.0)
        val result = mannWhitneyUTest(s1, s2)
        assertEquals(0.0, result.additionalInfo["z"]!!, 1e-15, "z should be 0 when all tied")
        assertEquals(1.0, result.pValue, 1e-6, "p should be 1 when all tied")
    }

    @Test
    fun testMannWhitneyUValidation() {
        assertFailsWith<InsufficientDataException> {
            mannWhitneyUTest(doubleArrayOf(), doubleArrayOf(1.0))
        }
        assertFailsWith<InsufficientDataException> {
            mannWhitneyUTest(doubleArrayOf(1.0), doubleArrayOf())
        }
    }

    @Test
    fun testMannWhitneyUAlternatives() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val two = mannWhitneyUTest(s1, s2)
        val less = mannWhitneyUTest(s1, s2, Alternative.LESS)
        val greater = mannWhitneyUTest(s1, s2, Alternative.GREATER)
        assertEquals(Alternative.TWO_SIDED, two.alternative)
        assertEquals(Alternative.LESS, less.alternative)
        assertEquals(Alternative.GREATER, greater.alternative)
    }

    // ===== Mann-Whitney U: scipy reference values =====

    @Test
    fun testMannWhitneyUScipyDisjoint() {
        // scipy.stats.mannwhitneyu([1,2,3,4,5], [6,7,8,9,10], alternative='two-sided')
        // scipy exact: U=0.0, p=0.007937 — our implementation uses normal approximation
        // U statistic: min(U1,U2)=0.0 matches exactly
        // p-value: normal approx gives ~0.009, scipy exact gives 0.00794
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = mannWhitneyUTest(s1, s2)
        assertEquals(0.0, result.statistic, 1e-10, "U statistic for disjoint samples")
        // Normal approximation differs from scipy's exact p-value; verify within reasonable range
        assertTrue(
            result.pValue < 0.02,
            "p-value for disjoint samples should be small, actual=${result.pValue}"
        )
        assertTrue(result.isSignificant(), "Disjoint samples should be significant at alpha=0.05")
    }

    @Test
    fun testMannWhitneyUScipyInterleaved() {
        // scipy.stats.mannwhitneyu([1,3,5,7,9], [2,4,6,8,10], alternative='two-sided')
        // scipy returns U for sample1; our implementation returns min(U1, U2)
        // R1=1+3+5+7+9=25, U1=25-15=10, U2=25-10=15, min=10
        val s1 = doubleArrayOf(1.0, 3.0, 5.0, 7.0, 9.0)
        val s2 = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = mannWhitneyUTest(s1, s2)
        assertEquals(10.0, result.statistic, 1e-10, "U statistic = min(U1, U2) for interleaved samples")
        // p-value should be not significant (interleaved data, no clear separation)
        assertFalse(result.isSignificant(), "Interleaved data should not be significant")
    }

    @Test
    fun testMannWhitneyUAdditionalInfo() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = mannWhitneyUTest(s1, s2)
        // U1 + U2 = n1 * n2 = 25
        val u1 = result.additionalInfo["U1"]!!
        val u2 = result.additionalInfo["U2"]!!
        assertEquals(25.0, u1 + u2, 1e-10, "U1 + U2 should equal n1 * n2")
        assertTrue(result.additionalInfo.containsKey("z"), "Should contain z-score")
    }

    @Test
    fun testMannWhitneyUTestName() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0)
        val s2 = doubleArrayOf(4.0, 5.0, 6.0)
        val result = mannWhitneyUTest(s1, s2)
        assertEquals("Mann-Whitney U Test", result.testName)
    }

    // ===== Wilcoxon Signed-Rank: existing tests =====

    @Test
    fun testWilcoxonSignedRank() {
        val s1 = doubleArrayOf(10.0, 12.0, 14.0, 16.0, 18.0)
        val s2 = doubleArrayOf(8.0, 9.0, 11.0, 12.0, 13.0)
        val result = wilcoxonSignedRankTest(s1, s2)
        assertTrue(result.statistic > 0)
    }

    @Test
    fun testWilcoxonOneSample() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = wilcoxonSignedRankTest(data)
        assertTrue(result.statistic > 0)
        assertEquals("Wilcoxon Signed-Rank Test", result.testName)
    }

    @Test
    fun testWilcoxonAllZeros() {
        assertFailsWith<DegenerateDataException> {
            wilcoxonSignedRankTest(doubleArrayOf(0.0, 0.0, 0.0))
        }
    }

    @Test
    fun testWilcoxonPairedAllEqual() {
        assertFailsWith<DegenerateDataException> {
            wilcoxonSignedRankTest(
                doubleArrayOf(5.0, 5.0, 5.0),
                doubleArrayOf(5.0, 5.0, 5.0)
            )
        }
    }

    @Test
    fun testWilcoxonMismatchedSizes() {
        assertFailsWith<InvalidParameterException> {
            wilcoxonSignedRankTest(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0))
        }
    }

    // ===== Wilcoxon Signed-Rank: scipy reference values =====

    @Test
    fun testWilcoxonScipyPaired() {
        // scipy.stats.wilcoxon([10,12,14,16,18], [8,9,11,12,13])
        // W+=15.0, z=2.0226, p=0.04312 (two-sided, approximate)
        val s1 = doubleArrayOf(10.0, 12.0, 14.0, 16.0, 18.0)
        val s2 = doubleArrayOf(8.0, 9.0, 11.0, 12.0, 13.0)
        val result = wilcoxonSignedRankTest(s1, s2)
        assertEquals(15.0, result.statistic, 1e-10, "W+ statistic")
        assertEquals(15.0, result.additionalInfo["wPlus"]!!, 1e-10, "wPlus")
        assertEquals(0.0, result.additionalInfo["wMinus"]!!, 1e-10, "wMinus (all diffs positive)")
        assertTrue(
            abs(result.additionalInfo["z"]!! - 2.0226) < 0.05,
            "z-score: expected~2.0226, actual=${result.additionalInfo["z"]}"
        )
        assertTrue(
            abs(result.pValue - 0.04312) < 0.01,
            "p-value: expected~0.04312, actual=${result.pValue}"
        )
    }

    @Test
    fun testWilcoxonAdditionalInfo() {
        val s1 = doubleArrayOf(10.0, 12.0, 14.0, 16.0, 18.0)
        val s2 = doubleArrayOf(8.0, 9.0, 11.0, 12.0, 13.0)
        val result = wilcoxonSignedRankTest(s1, s2)
        // wPlus + wMinus = n*(n+1)/2 = 5*6/2 = 15
        val wPlus = result.additionalInfo["wPlus"]!!
        val wMinus = result.additionalInfo["wMinus"]!!
        assertEquals(15.0, wPlus + wMinus, 1e-10, "wPlus + wMinus = n*(n+1)/2")
        assertTrue(result.additionalInfo.containsKey("z"), "Should contain z-score")
    }

    @Test
    fun testWilcoxonAlternatives() {
        val s1 = doubleArrayOf(10.0, 12.0, 14.0, 16.0, 18.0)
        val s2 = doubleArrayOf(8.0, 9.0, 11.0, 12.0, 13.0)
        val two = wilcoxonSignedRankTest(s1, s2)
        val less = wilcoxonSignedRankTest(s1, s2, Alternative.LESS)
        val greater = wilcoxonSignedRankTest(s1, s2, Alternative.GREATER)
        assertEquals(Alternative.TWO_SIDED, two.alternative)
        assertEquals(Alternative.LESS, less.alternative)
        assertEquals(Alternative.GREATER, greater.alternative)
        // s1 > s2 for all pairs, so GREATER should have small p, LESS should have large p
        assertTrue(greater.pValue < 0.05, "GREATER should be significant when s1 > s2")
        assertTrue(less.pValue > 0.5, "LESS should not be significant when s1 > s2")
    }
}
