package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.test.*

class AnovaTest {

    @Test
    fun testOneWayAnovaSignificant() {
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
        val result = oneWayAnova(g1, g2, g3)
        assertTrue(result.pValue < 0.001, "Very different groups should have low p-value")
        assertEquals(2, result.dfBetween)
        assertEquals(12, result.dfWithin)
    }

    @Test
    fun testOneWayAnovaNotSignificant() {
        val g1 = doubleArrayOf(5.0, 5.1, 4.9, 5.0, 5.2)
        val g2 = doubleArrayOf(5.0, 5.0, 5.1, 4.9, 5.0)
        val result = oneWayAnova(g1, g2)
        assertFalse(result.pValue < 0.05)
    }

    @Test
    fun testReference() {
        // ssBetween=250, ssWithin=30, msBetween=125, msWithin=2.5, F=50.0
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
        val result = oneWayAnova(g1, g2, g3)
        assertEquals(50.0, result.fStatistic, 1e-10)
        assertTrue(result.pValue < 1e-5, "p-value should be very small, got ${result.pValue}")
    }

    @Test
    fun testUnequalGroupSizes() {
        val g1 = doubleArrayOf(1.0, 2.0, 3.0)
        val g2 = doubleArrayOf(4.0, 5.0, 6.0, 7.0, 8.0)
        val result = oneWayAnova(g1, g2)
        assertTrue(result.pValue < 0.05, "Unequal groups with different means should be significant")
        assertEquals(1, result.dfBetween)
        assertEquals(6, result.dfWithin)
    }

    @Test
    fun testDecomposition() {
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = oneWayAnova(g1, g2)
        // msBetween / msWithin should equal fStatistic
        assertEquals(result.fStatistic, result.msBetween / result.msWithin, 1e-10)
        // ssBetween + ssWithin should equal total SS
        assertEquals(result.ssBetween, result.msBetween * result.dfBetween, 1e-10)
        assertEquals(result.ssWithin, result.msWithin * result.dfWithin, 1e-10)
    }

    // ===== Validation =====

    @Test
    fun testFewerThanTwoGroups() {
        assertFailsWith<InsufficientDataException> {
            oneWayAnova(doubleArrayOf(1.0, 2.0, 3.0))
        }
    }

    @Test
    fun testGroupWithTooFewElements() {
        assertFailsWith<InsufficientDataException> {
            oneWayAnova(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0))
        }
    }
}
