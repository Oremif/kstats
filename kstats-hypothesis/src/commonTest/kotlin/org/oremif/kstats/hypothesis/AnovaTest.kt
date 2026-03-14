package org.oremif.kstats.hypothesis

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

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
}
