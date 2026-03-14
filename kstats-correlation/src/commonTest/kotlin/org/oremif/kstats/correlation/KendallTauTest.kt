package org.oremif.kstats.correlation

import kotlin.test.Test
import kotlin.test.assertEquals

class KendallTauTest {

    @Test
    fun testConcordant() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = kendallTau(x, y)
        assertEquals(1.0, result.coefficient, 1e-10)
    }

    @Test
    fun testDiscordant() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(5.0, 4.0, 3.0, 2.0, 1.0)
        val result = kendallTau(x, y)
        assertEquals(-1.0, result.coefficient, 1e-10)
    }
}
