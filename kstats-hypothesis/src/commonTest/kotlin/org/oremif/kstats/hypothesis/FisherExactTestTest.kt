package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.*

class FisherExactTestTest {

    private fun assertP(expected: Double, actual: Double, message: String = "") {
        TestAssertions.assertPValue(expected, actual, message = message)
    }

    // ===== Basic correctness: classic tables vs scipy =====

    @Test
    fun testClassicTable() {
        // scipy: fisher_exact([[1,9],[11,3]])
        val table = arrayOf(intArrayOf(1, 9), intArrayOf(11, 3))
        val two = fisherExactTest(table, Alternative.TWO_SIDED)
        val less = fisherExactTest(table, Alternative.LESS)
        val greater = fisherExactTest(table, Alternative.GREATER)
        assertEquals(0.0303030303030303, two.statistic, 1e-12, "OR classic")
        assertP(0.00275945618522008, two.pValue, "classic two-sided")
        assertP(0.00137972809261004, less.pValue, "classic less")
        assertP(0.999966348095302, greater.pValue, "classic greater")
        assertTrue(two.isSignificant())
    }

    @Test
    fun testTreatmentTable() {
        // scipy: fisher_exact([[10,5],[3,12]])
        val table = arrayOf(intArrayOf(10, 5), intArrayOf(3, 12))
        val two = fisherExactTest(table, Alternative.TWO_SIDED)
        val less = fisherExactTest(table, Alternative.LESS)
        val greater = fisherExactTest(table, Alternative.GREATER)
        assertEquals(8.0, two.statistic, 1e-12, "OR treatment")
        assertP(0.0253276870336761, two.pValue, "treatment two-sided")
        assertP(0.998745364160025, less.pValue, "treatment less")
        assertP(0.0126638435168381, greater.pValue, "treatment greater")
    }

    @Test
    fun testSmallTable() {
        // scipy: fisher_exact([[2,3],[4,1]])
        val table = arrayOf(intArrayOf(2, 3), intArrayOf(4, 1))
        val two = fisherExactTest(table, Alternative.TWO_SIDED)
        val less = fisherExactTest(table, Alternative.LESS)
        val greater = fisherExactTest(table, Alternative.GREATER)
        assertEquals(0.166666666666667, two.statistic, 1e-10, "OR small")
        assertP(0.523809523809524, two.pValue, "small two-sided")
        assertP(0.261904761904762, less.pValue, "small less")
        assertP(0.976190476190476, greater.pValue, "small greater")
    }

    // ===== Symmetric tables: stress test for tolerance =====

    @Test
    fun testSymmetric5x5() {
        // scipy: fisher_exact([[5,5],[5,5]]) -> OR=1, p=1 (two-sided)
        val table = arrayOf(intArrayOf(5, 5), intArrayOf(5, 5))
        val two = fisherExactTest(table, Alternative.TWO_SIDED)
        val less = fisherExactTest(table, Alternative.LESS)
        val greater = fisherExactTest(table, Alternative.GREATER)
        assertEquals(1.0, two.statistic, 1e-12, "OR symmetric 5x5")
        assertP(1.0, two.pValue, "symmetric 5x5 two-sided")
        assertP(0.67185910065167, less.pValue, "symmetric 5x5 less")
        assertP(0.67185910065167, greater.pValue, "symmetric 5x5 greater")
        assertFalse(two.isSignificant())
    }

    @Test
    fun testSymmetric10x10() {
        // scipy: fisher_exact([[10,10],[10,10]]) -> OR=1, p=1 (two-sided)
        val table = arrayOf(intArrayOf(10, 10), intArrayOf(10, 10))
        val two = fisherExactTest(table, Alternative.TWO_SIDED)
        val less = fisherExactTest(table, Alternative.LESS)
        val greater = fisherExactTest(table, Alternative.GREATER)
        assertEquals(1.0, two.statistic, 1e-12, "OR symmetric 10x10")
        assertP(1.0, two.pValue, "symmetric 10x10 two-sided")
        assertP(0.623814432718045, less.pValue, "symmetric 10x10 less")
        assertP(0.623814432718045, greater.pValue, "symmetric 10x10 greater")
    }

    @Test
    fun testSymmetric3x7() {
        // scipy: fisher_exact([[3,7],[7,3]])
        val table = arrayOf(intArrayOf(3, 7), intArrayOf(7, 3))
        val two = fisherExactTest(table, Alternative.TWO_SIDED)
        val less = fisherExactTest(table, Alternative.LESS)
        val greater = fisherExactTest(table, Alternative.GREATER)
        assertEquals(0.183673469387755, two.statistic, 1e-10, "OR symmetric 3x7")
        assertP(0.178895407997575, two.pValue, "symmetric 3x7 two-sided")
        assertP(0.0894477039987876, less.pValue, "symmetric 3x7 less")
        assertP(0.988492931217389, greater.pValue, "symmetric 3x7 greater")
    }

    // ===== Zero cells =====

    @Test
    fun testZeroDiagonal() {
        // scipy: fisher_exact([[0,5],[5,0]]) -> OR=0, p≈0.008
        val table = arrayOf(intArrayOf(0, 5), intArrayOf(5, 0))
        val two = fisherExactTest(table, Alternative.TWO_SIDED)
        val less = fisherExactTest(table, Alternative.LESS)
        val greater = fisherExactTest(table, Alternative.GREATER)
        assertP(0.00793650793650794, two.pValue, "zero diag two-sided")
        assertP(0.00396825396825397, less.pValue, "zero diag less")
        assertP(1.0, greater.pValue, "zero diag greater")
    }

    @Test
    fun testZeroRow() {
        // scipy: fisher_exact([[0,5],[0,5]]) -> OR=NaN, p=1
        val table = arrayOf(intArrayOf(0, 5), intArrayOf(0, 5))
        val two = fisherExactTest(table, Alternative.TWO_SIDED)
        assertP(1.0, two.pValue, "zero row two-sided")
    }

    @Test
    fun testZeroCorner() {
        // scipy: fisher_exact([[3,0],[0,4]]) -> OR=inf, p≈0.029
        val table = arrayOf(intArrayOf(3, 0), intArrayOf(0, 4))
        val two = fisherExactTest(table, Alternative.TWO_SIDED)
        val less = fisherExactTest(table, Alternative.LESS)
        val greater = fisherExactTest(table, Alternative.GREATER)
        assertEquals(Double.POSITIVE_INFINITY, two.statistic, "OR zero corner")
        assertP(0.0285714285714286, two.pValue, "zero corner two-sided")
        assertP(1.0, less.pValue, "zero corner less")
        assertP(0.0285714285714286, greater.pValue, "zero corner greater")
    }

    // ===== Large counts: numerical precision =====

    @Test
    fun testLargeCounts() {
        // scipy: fisher_exact([[200,300],[400,100]])
        val table = arrayOf(intArrayOf(200, 300), intArrayOf(400, 100))
        val two = fisherExactTest(table, Alternative.TWO_SIDED)
        val less = fisherExactTest(table, Alternative.LESS)
        val greater = fisherExactTest(table, Alternative.GREATER)
        assertEquals(0.166666666666667, two.statistic, 1e-10, "OR large")
        assertTrue(two.pValue < 1e-35, "large two-sided p should be very small, got ${two.pValue}")
        assertTrue(less.pValue < 1e-35, "large less p should be very small, got ${less.pValue}")
        assertP(1.0, greater.pValue, "large greater")
    }

    @Test
    fun testLargeEqual() {
        // scipy: fisher_exact([[50,50],[50,50]]) -> OR=1, p=1
        val table = arrayOf(intArrayOf(50, 50), intArrayOf(50, 50))
        val two = fisherExactTest(table, Alternative.TWO_SIDED)
        val less = fisherExactTest(table, Alternative.LESS)
        val greater = fisherExactTest(table, Alternative.GREATER)
        assertEquals(1.0, two.statistic, 1e-12, "OR large equal")
        assertP(1.0, two.pValue, "large equal two-sided")
        assertP(0.556207787852021, less.pValue, "large equal less")
        assertP(0.556207787852021, greater.pValue, "large equal greater")
    }

    @Test
    fun testLargeModerate() {
        // scipy: fisher_exact([[100,200],[150,250]])
        val table = arrayOf(intArrayOf(100, 200), intArrayOf(150, 250))
        val two = fisherExactTest(table, Alternative.TWO_SIDED)
        val less = fisherExactTest(table, Alternative.LESS)
        val greater = fisherExactTest(table, Alternative.GREATER)
        assertEquals(0.833333333333333, two.statistic, 1e-10, "OR large moderate")
        assertP(0.265321924812353, two.pValue, "large moderate two-sided")
        assertP(0.14479914615803, less.pValue, "large moderate less")
        assertP(0.888554332846049, greater.pValue, "large moderate greater")
    }

    // ===== Edge cases: validation and metadata =====

    @Test
    fun testInvalidTableShape() {
        assertFailsWith<InvalidParameterException> {
            fisherExactTest(arrayOf(intArrayOf(1, 2, 3), intArrayOf(4, 5, 6)))
        }
    }

    @Test
    fun testNegativeValues() {
        assertFailsWith<InvalidParameterException> {
            fisherExactTest(arrayOf(intArrayOf(-1, 2), intArrayOf(3, 4)))
        }
    }

    @Test
    fun testTestNameAndOddsRatio() {
        val table = arrayOf(intArrayOf(10, 5), intArrayOf(3, 12))
        val result = fisherExactTest(table)
        assertEquals("Fisher's Exact Test", result.testName)
        assertEquals(8.0, result.additionalInfo["oddsRatio"] as Double, 1e-12)
    }
}
