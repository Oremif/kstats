package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.test.*

class AnovaTest {

    @Test
    fun testOneWayAnovaSignificant() {
        val result = oneWayAnova(TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10, TestData.SEQUENTIAL_11_15)
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
        val result = oneWayAnova(TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10, TestData.SEQUENTIAL_11_15)
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
        val result = oneWayAnova(TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10)
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

    // ===== Degenerate cases: constant data =====

    @Test
    fun testConstantDataAllGroupsSameValue() {
        // All values identical across all groups: msWithin=0, msBetween=0 → F=0, p=1
        val g1 = doubleArrayOf(5.0, 5.0, 5.0)
        val g2 = doubleArrayOf(5.0, 5.0, 5.0)
        val g3 = doubleArrayOf(5.0, 5.0, 5.0)
        val result = oneWayAnova(g1, g2, g3)
        assertEquals(0.0, result.fStatistic, 1e-15, "F should be 0 when all data is constant")
        assertEquals(1.0, result.pValue, 1e-10, "p should be 1 when all data is constant")
        assertEquals(0.0, result.ssWithin, 1e-15, "ssWithin should be 0")
        assertEquals(0.0, result.ssBetween, 1e-15, "ssBetween should be 0")
        assertEquals(0.0, result.msWithin, 1e-15, "msWithin should be 0")
        assertEquals(0.0, result.msBetween, 1e-15, "msBetween should be 0")
    }

    @Test
    fun testConstantWithinGroupsDifferentAcross() {
        // Each group is constant but groups differ: msWithin=0, msBetween>0 → F=Inf, p=0
        val g1 = doubleArrayOf(1.0, 1.0, 1.0)
        val g2 = doubleArrayOf(5.0, 5.0, 5.0)
        val g3 = doubleArrayOf(10.0, 10.0, 10.0)
        val result = oneWayAnova(g1, g2, g3)
        assertEquals(Double.POSITIVE_INFINITY, result.fStatistic, "F should be +Inf when msWithin=0 and msBetween>0")
        assertEquals(0.0, result.pValue, 1e-15, "p should be 0 when F is infinite")
        assertEquals(0.0, result.ssWithin, 1e-15, "ssWithin should be 0")
        assertTrue(result.ssBetween > 0.0, "ssBetween should be positive")
        assertEquals(0.0, result.msWithin, 1e-15, "msWithin should be 0")
        assertTrue(result.msBetween > 0.0, "msBetween should be positive")
    }

    @Test
    fun testConstantWithinGroupsTwoGroups() {
        // Two constant groups with different means
        val g1 = doubleArrayOf(3.0, 3.0, 3.0, 3.0)
        val g2 = doubleArrayOf(7.0, 7.0, 7.0, 7.0)
        val result = oneWayAnova(g1, g2)
        assertEquals(Double.POSITIVE_INFINITY, result.fStatistic, "F should be +Inf")
        assertEquals(0.0, result.pValue, 1e-15, "p should be 0")
        assertEquals(1, result.dfBetween)
        assertEquals(6, result.dfWithin)
    }
}
