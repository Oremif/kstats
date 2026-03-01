package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class FrequencyTest {

    // ── Basic correctness ────────────────────────────────────────────────────

    @Test
    fun testStringFrequencies() {
        val freq = Frequency<String>()
        freq.addAll(listOf("a", "a", "b", "b", "b", "c"))

        assertEquals(6L, freq.totalCount)
        assertEquals(3, freq.uniqueCount)
        assertEquals(2L, freq.count("a"))
        assertEquals(3L, freq.count("b"))
        assertEquals(1L, freq.count("c"))
    }

    @Test
    fun testProportions() {
        val freq = Frequency<String>()
        freq.addAll(listOf("a", "a", "b", "b", "b", "c"))

        assertEquals(2.0 / 6.0, freq.proportion("a"), 1e-10)
        assertEquals(3.0 / 6.0, freq.proportion("b"), 1e-10)
        assertEquals(1.0 / 6.0, freq.proportion("c"), 1e-10)
    }

    @Test
    fun testCumulativeCount() {
        val freq = Frequency<String>()
        freq.addAll(listOf("a", "a", "b", "b", "b", "c"))

        assertEquals(2L, freq.cumulativeCount("a"))  // a=2
        assertEquals(5L, freq.cumulativeCount("b"))  // a=2 + b=3
        assertEquals(6L, freq.cumulativeCount("c"))  // a=2 + b=3 + c=1
    }

    @Test
    fun testCumulativeProportion() {
        val freq = Frequency<String>()
        freq.addAll(listOf("a", "a", "b", "b", "b", "c"))

        assertEquals(2.0 / 6.0, freq.cumulativeProportion("a"), 1e-10)
        assertEquals(5.0 / 6.0, freq.cumulativeProportion("b"), 1e-10)
        assertEquals(1.0, freq.cumulativeProportion("c"), 1e-10)
    }

    @Test
    fun testEntries() {
        val freq = Frequency<String>()
        freq.addAll(listOf("a", "a", "b", "b", "b", "c"))

        val entries = freq.entries
        assertEquals(3, entries.size)
        assertEquals(FrequencyEntry("a", 2L), entries[0])
        assertEquals(FrequencyEntry("b", 3L), entries[1])
        assertEquals(FrequencyEntry("c", 1L), entries[2])
    }

    @Test
    fun testIntegerFrequencies() {
        val freq = Frequency<Int>()
        freq.addAll(listOf(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5))

        assertEquals(11L, freq.totalCount)
        assertEquals(7, freq.uniqueCount)
        assertEquals(2L, freq.count(1))
        assertEquals(1L, freq.count(2))
        assertEquals(2L, freq.count(3))
        assertEquals(1L, freq.count(4))
        assertEquals(3L, freq.count(5))
        assertEquals(1L, freq.count(6))
        assertEquals(1L, freq.count(9))
    }

    // ── Edge cases ───────────────────────────────────────────────────────────

    @Test
    fun testEmptyFrequency() {
        val freq = Frequency<String>()

        assertEquals(0L, freq.totalCount)
        assertEquals(0, freq.uniqueCount)
        assertEquals(0L, freq.count("x"))
        assertEquals(0.0, freq.proportion("x"), 1e-15)
        assertTrue(freq.cumulativeProportion("x").isNaN())
        assertEquals(0L, freq.cumulativeCount("x"))
        assertEquals(emptySet(), freq.mode)
        assertEquals(emptyList(), freq.values)
        assertEquals(emptyList(), freq.entries)
    }

    @Test
    fun testSingleElement() {
        val freq = Frequency<Int>()
        freq.add(42)

        assertEquals(1L, freq.totalCount)
        assertEquals(1, freq.uniqueCount)
        assertEquals(1L, freq.count(42))
        assertEquals(1.0, freq.proportion(42), 1e-15)
        assertEquals(1.0, freq.cumulativeProportion(42), 1e-15)
        assertEquals(setOf(42), freq.mode)
        assertEquals(listOf(42), freq.values)
    }

    @Test
    fun testNonExistentKey() {
        val freq = Frequency<String>()
        freq.add("a")

        assertEquals(0L, freq.count("z"))
        assertEquals(0.0, freq.proportion("z"), 1e-15)
    }

    @Test
    fun testAddWithCount() {
        val freq = Frequency<String>()
        freq.add("a", 5L)
        freq.add("b", 3L)

        assertEquals(8L, freq.totalCount)
        assertEquals(5L, freq.count("a"))
        assertEquals(3L, freq.count("b"))
    }

    @Test
    fun testAddWithCountAccumulates() {
        val freq = Frequency<String>()
        freq.add("a", 2L)
        freq.add("a", 3L)

        assertEquals(5L, freq.count("a"))
        assertEquals(5L, freq.totalCount)
    }

    @Test
    fun testAddWithInvalidCount() {
        val freq = Frequency<String>()

        assertFailsWith<InvalidParameterException> { freq.add("a", 0L) }
        assertFailsWith<InvalidParameterException> { freq.add("a", -1L) }
        assertFailsWith<InvalidParameterException> { freq.add("a", -100L) }
    }

    // ── Degenerate data ──────────────────────────────────────────────────────

    @Test
    fun testAllIdenticalValues() {
        val freq = Frequency<Int>()
        freq.addAll(listOf(7, 7, 7, 7, 7))

        assertEquals(5L, freq.totalCount)
        assertEquals(1, freq.uniqueCount)
        assertEquals(5L, freq.count(7))
        assertEquals(1.0, freq.proportion(7), 1e-15)
        assertEquals(setOf(7), freq.mode)
    }

    @Test
    fun testSingleUniqueWithLargeCount() {
        val freq = Frequency<String>()
        freq.add("x", 1_000_000L)

        assertEquals(1_000_000L, freq.totalCount)
        assertEquals(1, freq.uniqueCount)
        assertEquals(1.0, freq.proportion("x"), 1e-15)
    }

    // ── Mode ─────────────────────────────────────────────────────────────────

    @Test
    fun testSingleMode() {
        val freq = Frequency<Int>()
        freq.addAll(listOf(1, 2, 2, 3))

        assertEquals(setOf(2), freq.mode)
    }

    @Test
    fun testTiedModes() {
        val freq = Frequency<Int>()
        freq.addAll(listOf(1, 1, 2, 2, 3))

        assertEquals(setOf(1, 2), freq.mode)
    }

    @Test
    fun testAllEqualFrequencies() {
        val freq = Frequency<Int>()
        freq.addAll(listOf(1, 2, 3, 4, 5))

        // All values appear once — all are modes
        assertEquals(setOf(1, 2, 3, 4, 5), freq.mode)
    }

    // ── Sorted order ─────────────────────────────────────────────────────────

    @Test
    fun testValuesAreSorted() {
        val freq = Frequency<Int>()
        freq.addAll(listOf(5, 3, 1, 4, 2))

        assertEquals(listOf(1, 2, 3, 4, 5), freq.values)
    }

    @Test
    fun testEntriesAreSortedByValue() {
        val freq = Frequency<String>()
        freq.addAll(listOf("c", "a", "b", "a", "c", "c"))

        val entries = freq.entries
        assertEquals("a", entries[0].value)
        assertEquals("b", entries[1].value)
        assertEquals("c", entries[2].value)
    }

    @Test
    fun testClearAndReuse() {
        val freq = Frequency<Int>()
        freq.addAll(listOf(1, 2, 3))

        freq.clear()

        assertEquals(0L, freq.totalCount)
        assertEquals(0, freq.uniqueCount)
        assertEquals(emptyList(), freq.values)

        // Reuse after clear
        freq.addAll(listOf(10, 20, 20))
        assertEquals(3L, freq.totalCount)
        assertEquals(2, freq.uniqueCount)
        assertEquals(2L, freq.count(20))
    }

    // ── Factory ──────────────────────────────────────────────────────────────

    @Test
    fun testToFrequency() {
        val freq = listOf("a", "b", "b", "c", "c", "c").toFrequency()

        assertEquals(6L, freq.totalCount)
        assertEquals(1L, freq.count("a"))
        assertEquals(2L, freq.count("b"))
        assertEquals(3L, freq.count("c"))
    }

    @Test
    fun testToFrequencyMatchesManualAdd() {
        val data = listOf(3, 1, 4, 1, 5, 9)

        val fromFactory = data.toFrequency()
        val manual = Frequency<Int>()
        manual.addAll(data)

        assertEquals(manual.totalCount, fromFactory.totalCount)
        assertEquals(manual.uniqueCount, fromFactory.uniqueCount)
        assertEquals(manual.values, fromFactory.values)
        assertEquals(manual.entries, fromFactory.entries)
    }

    @Test
    fun testAddAllMatchesSequentialAdd() {
        val data = listOf("x", "y", "x", "z", "y", "x")

        val bulk = Frequency<String>()
        bulk.addAll(data)

        val sequential = Frequency<String>()
        for (v in data) sequential.add(v)

        assertEquals(bulk.totalCount, sequential.totalCount)
        assertEquals(bulk.uniqueCount, sequential.uniqueCount)
        assertEquals(bulk.entries, sequential.entries)
    }

    // ── Cumulative with gaps ─────────────────────────────────────────────────

    @Test
    fun testCumulativeCountForValueBetweenKeys() {
        val freq = Frequency<Int>()
        freq.addAll(listOf(1, 1, 3, 3, 3, 5))

        // value 2 is not present, but cumulative should include keys ≤ 2
        assertEquals(2L, freq.cumulativeCount(2))  // only 1's
        assertEquals(5L, freq.cumulativeCount(4))  // 1's + 3's
    }

    @Test
    fun testCumulativeCountForValueBeforeAllKeys() {
        val freq = Frequency<Int>()
        freq.addAll(listOf(5, 10, 15))

        assertEquals(0L, freq.cumulativeCount(1))
    }

    @Test
    fun testCumulativeProportionForValueBeyondAllKeys() {
        val freq = Frequency<Int>()
        freq.addAll(listOf(1, 2, 3))

        assertEquals(1.0, freq.cumulativeProportion(100), 1e-15)
    }
}
