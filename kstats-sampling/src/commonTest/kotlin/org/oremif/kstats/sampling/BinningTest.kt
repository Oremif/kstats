package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BinningTest {

    // --- bin by size ---

    @Test
    fun testBinBySize() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val bins = data.bin(5.0)
        assertEquals(2, bins.size)
        assertEquals(10, bins.sumOf { it.count })
    }

    @Test
    fun testBinBySizeEmpty() {
        val bins = emptyList<Double>().bin(1.0)
        assertTrue(bins.isEmpty())
    }

    @Test
    fun testBinBySizeInvalidThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1.0).bin(0.0) // binSize = 0
        }
        assertFailsWith<InvalidParameterException> {
            listOf(1.0).bin(-1.0) // binSize < 0
        }
    }

    @Test
    fun testBinBySizeNaNThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1.0, Double.NaN).bin(1.0)
        }
    }

    @Test
    fun testBinBySizeInfinityThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1.0, Double.POSITIVE_INFINITY).bin(1.0)
        }
    }

    @Test
    fun testBinBySizeNaNBinSizeThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1.0, 2.0).bin(Double.NaN)
        }
    }

    @Test
    fun testBinBySizeInfBinSizeThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1.0, 2.0).bin(Double.POSITIVE_INFINITY)
        }
    }

    // --- bin by count ---

    @Test
    fun testBinByCount() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val bins = data.bin(3)
        assertEquals(3, bins.size)
        assertEquals(10, bins.sumOf { it.count })
    }

    @Test
    fun testBinByCountExactCount() {
        // Verify the floating-point roundtrip bug is fixed:
        // range / binCount should always produce exactly binCount bins.
        for (n in listOf(3, 7, 11, 13, 17)) {
            val data = (1..100).map { it.toDouble() }
            val bins = data.bin(n)
            assertEquals(n, bins.size, "Expected $n bins")
            assertEquals(100, bins.sumOf { it.count }, "All items must be assigned for binCount=$n")
        }
    }

    @Test
    fun testBinByCountAllSameValues() {
        val data = listOf(5.0, 5.0, 5.0, 5.0)
        val bins = data.bin(3)
        // All same values -> 1 bin regardless of requested count
        assertEquals(1, bins.size)
        assertEquals(4, bins[0].count)
    }

    @Test
    fun testBinByCountEmpty() {
        val bins = emptyList<Double>().bin(3)
        assertTrue(bins.isEmpty())
    }

    @Test
    fun testBinByCountInvalidThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1.0).bin(0) // binCount = 0
        }
        assertFailsWith<InvalidParameterException> {
            listOf(1.0).bin(-1) // binCount < 0
        }
    }

    @Test
    fun testBinByCountNaNThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1.0, Double.NaN).bin(2)
        }
    }

    // --- boundary assignment ---

    @Test
    fun testBoundaryAssignment() {
        // Values on interior boundaries go to the higher bin
        val data = listOf(0.0, 5.0, 10.0)
        val bins = data.bin(5.0)
        assertEquals(2, bins.size)
        // 0.0 -> bin 0, 5.0 -> bin 1 (boundary), 10.0 -> bin 1 (last bin includes upper)
        assertEquals(1, bins[0].count) // only 0.0
        assertEquals(2, bins[1].count) // 5.0 and 10.0
    }

    @Test
    fun testLastBinIncludesMaxValue() {
        val data = listOf(1.0, 2.0, 3.0)
        val bins = data.bin(1)
        // 1 bin -> all items in it
        assertEquals(1, bins.size)
        assertEquals(3, bins[0].count)
    }

    // --- rangeStart validation ---

    @Test
    fun testBinByDoubleRangeStartValid() {
        val data = listOf(5.0, 6.0, 7.0, 8.0)
        val bins = data.binByDouble({ it }, binSize = 2.0, rangeStart = 4.0)
        // Range starts at 4.0 instead of 5.0
        assertTrue(bins.isNotEmpty())
        assertEquals(4, bins.sumOf { it.count })
    }

    @Test
    fun testBinByDoubleRangeStartExceedsMinThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1.0, 2.0, 3.0).binByDouble({ it }, binSize = 1.0, rangeStart = 2.0)
        }
    }

    @Test
    fun testBinByDoubleRangeStartNaNThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1.0, 2.0).binByDouble({ it }, binSize = 1.0, rangeStart = Double.NaN)
        }
    }

    @Test
    fun testBinByDoubleRangeStartInfThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1.0, 2.0).binByDouble({ it }, binSize = 1.0, rangeStart = Double.NEGATIVE_INFINITY)
        }
    }

    // --- frequency table ---

    @Test
    fun testFrequencyTableByCount() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val freq = data.frequencyTable(2)
        assertEquals(2, freq.size)
        assertEquals(1.0, freq.last().cumulativeFrequency, 1e-10)
        assertEquals(1.0, freq.sumOf { it.relativeFrequency }, 1e-10)
    }

    @Test
    fun testFrequencyTableBySize() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val freq = data.frequencyTable(2.5)
        assertEquals(1.0, freq.last().cumulativeFrequency, 1e-10)
        assertEquals(5, freq.sumOf { it.count })
    }

    @Test
    fun testFrequencyTableEmpty() {
        val freq = emptyList<Double>().frequencyTable(3)
        assertTrue(freq.isEmpty())
    }

    @Test
    fun testFrequencyTableRelativeFrequencies() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val freq = data.frequencyTable(2)
        for (bin in freq) {
            assertTrue(bin.relativeFrequency >= 0.0)
            assertTrue(bin.relativeFrequency <= 1.0)
        }
        // Cumulative frequency should be monotonically increasing
        for (i in 1 until freq.size) {
            assertTrue(freq[i].cumulativeFrequency >= freq[i - 1].cumulativeFrequency)
        }
    }

    @Test
    fun testBoundaryAssignmentThreeBins() {
        // Values on interior boundaries (2.0, 4.0) should go to the higher bin
        val data = listOf(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        val bins = data.bin(2.0)
        assertEquals(3, bins.size)
        // bin 0: [0, 2) -> {0.0, 1.0}
        assertEquals(2, bins[0].count)
        // bin 1: [2, 4) -> {2.0, 3.0}
        assertEquals(2, bins[1].count)
        // bin 2: [4, 6] -> {4.0, 5.0, 6.0}
        assertEquals(3, bins[2].count)
    }

    @Test
    fun testLastBinRangeCoversMaxValue() {
        val data = listOf(1.0, 10.0)
        val bins = data.bin(4.0)
        // Last bin range should include maxVal (10.0)
        assertTrue(bins.last().range.endInclusive >= 10.0)
    }

    @Test
    fun testBinLargeArray() {
        val data = (1..10000).map { it.toDouble() }
        val bins = data.bin(7)
        assertEquals(7, bins.size)
        assertEquals(10000, bins.sumOf { it.count })
    }

    // --- binByDouble with custom valueSelector ---

    // --- DoubleArray overloads ---

    @Test
    fun testDoubleArrayBinBySize() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val bins = data.bin(5.0)
        assertEquals(2, bins.size)
        assertEquals(10, bins.sumOf { it.count })
    }

    @Test
    fun testDoubleArrayBinByCount() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val bins = data.bin(3)
        assertEquals(3, bins.size)
        assertEquals(10, bins.sumOf { it.count })
    }

    @Test
    fun testDoubleArrayFrequencyTableByCount() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val freq = data.frequencyTable(2)
        assertEquals(2, freq.size)
        assertEquals(1.0, freq.last().cumulativeFrequency, 1e-10)
    }

    @Test
    fun testDoubleArrayFrequencyTableBySize() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val freq = data.frequencyTable(2.5)
        assertEquals(1.0, freq.last().cumulativeFrequency, 1e-10)
        assertEquals(5, freq.sumOf { it.count })
    }

    // --- binByDouble with custom valueSelector ---

    @Test
    fun testBinByDoubleCustomSelector() {
        data class Item(val value: Double, val name: String)

        val items = listOf(
            Item(1.0, "a"), Item(3.5, "b"), Item(7.0, "c"), Item(8.0, "d")
        )
        val bins = items.binByDouble({ it.value }, binSize = 5.0)
        assertEquals(2, bins.size)
        assertEquals(2, bins[0].count) // 1.0, 3.5
        assertEquals(2, bins[1].count) // 7.0, 8.0
    }
}
