package org.oremif.kstats.descriptive

import kotlin.test.Test
import kotlin.test.assertEquals

class NumberExtensionsTest {

    @Test
    fun testIntMean() {
        assertEquals(3.0, listOf(1, 2, 3, 4, 5).mean(), 1e-10)
    }

    @Test
    fun testIntMedian() {
        assertEquals(3.0, listOf(1, 2, 3, 4, 5).median(), 1e-10)
    }

    @Test
    fun testLongMean() {
        assertEquals(3.0, listOf(1L, 2L, 3L, 4L, 5L).mean(), 1e-10)
    }
}
