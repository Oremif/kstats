package org.oremif.kstats.sampling

import kotlin.test.Test
import kotlin.test.assertEquals

class BinningTest {

    @Test
    fun testBinBySize() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val bins = data.bin(5.0)
        assertEquals(2, bins.size)
    }

    @Test
    fun testBinByCount() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val bins = data.bin(3)
        assertEquals(3, bins.size)
        assertEquals(10, bins.sumOf { it.count })
    }

    @Test
    fun testFrequencyTable() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val freq = data.frequencyTable(2)
        assertEquals(1.0, freq.last().cumulativeFrequency, 1e-10)
    }
}
