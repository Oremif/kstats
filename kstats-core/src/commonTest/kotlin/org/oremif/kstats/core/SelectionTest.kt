package org.oremif.kstats.core

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SelectionTest {

    @Test
    fun selectMinimum() {
        val arr = doubleArrayOf(5.0, 3.0, 1.0, 4.0, 2.0)
        arr.introSelect(0)
        assertEquals(1.0, arr[0])
    }

    @Test
    fun selectMaximum() {
        val arr = doubleArrayOf(5.0, 3.0, 1.0, 4.0, 2.0)
        arr.introSelect(arr.lastIndex)
        assertEquals(5.0, arr[arr.lastIndex])
    }

    @Test
    fun selectMedianOdd() {
        val arr = doubleArrayOf(5.0, 3.0, 1.0, 4.0, 2.0)
        arr.introSelect(2)
        assertEquals(3.0, arr[2])
    }

    @Test
    fun selectSingleElement() {
        val arr = doubleArrayOf(42.0)
        arr.introSelect(0)
        assertEquals(42.0, arr[0])
    }

    @Test
    fun selectTwoElements() {
        val a = doubleArrayOf(7.0, 3.0)
        a.introSelect(0)
        assertEquals(3.0, a[0])

        val b = doubleArrayOf(7.0, 3.0)
        b.introSelect(1)
        assertEquals(7.0, b[1])
    }

    @Test
    fun selectAlreadySorted() {
        val arr = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        arr.introSelect(2)
        assertEquals(3.0, arr[2])
    }

    @Test
    fun selectReverseSorted() {
        val arr = doubleArrayOf(5.0, 4.0, 3.0, 2.0, 1.0)
        arr.introSelect(2)
        assertEquals(3.0, arr[2])
    }

    @Test
    fun selectAllEqual() {
        val arr = doubleArrayOf(7.0, 7.0, 7.0, 7.0, 7.0)
        arr.introSelect(2)
        assertEquals(7.0, arr[2])
    }

    @Test
    fun partitionInvariant() {
        val rng = Random(12345)
        val arr = DoubleArray(101) { rng.nextDouble() }
        val k = 50
        arr.introSelect(k)
        val pivot = arr[k]
        for (i in 0 until k) {
            assertTrue(arr[i] <= pivot, "arr[$i]=${arr[i]} should be <= pivot=$pivot")
        }
        for (i in k + 1..arr.lastIndex) {
            assertTrue(arr[i] >= pivot, "arr[$i]=${arr[i]} should be >= pivot=$pivot")
        }
    }

    @Test
    fun selectLargeRandom() {
        val rng = Random(98765)
        val arr = DoubleArray(10001) { rng.nextDouble() }
        val reference = arr.sortedArray()
        val k = 5000
        arr.introSelect(k)
        assertEquals(reference[k], arr[k], 0.0)
    }

    @Test
    fun selectNaN() {
        val arr = doubleArrayOf(3.0, Double.NaN, 1.0, 2.0, Double.NaN)
        arr.introSelect(0)
        // NaN compares > everything via Double.compareTo, so non-NaN values come first
        assertEquals(1.0, arr[0])
    }

    @Test
    fun selectInfinity() {
        val arr = doubleArrayOf(Double.POSITIVE_INFINITY, 0.0, Double.NEGATIVE_INFINITY, 1.0, -1.0)
        val reference = arr.sortedArray()
        for (k in arr.indices) {
            val copy = arr.copyOf()
            copy.introSelect(k)
            assertEquals(reference[k], copy[k], 0.0)
        }
    }
}
