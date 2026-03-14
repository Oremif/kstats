package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class CentralTendencyTest {

    @Test
    fun testMean() {
        assertEquals(3.0, listOf(1.0, 2.0, 3.0, 4.0, 5.0).mean(), 1e-10)
        assertEquals(3.0, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).mean(), 1e-10)
    }

    @Test
    fun testMeanEmpty() {
        assertFailsWith<InsufficientDataException> { emptyList<Double>().mean() }
        assertFailsWith<InsufficientDataException> { doubleArrayOf().mean() }
    }

    @Test
    fun testGeometricMean() {
        // geometric mean of 1,2,4,8 = (1*2*4*8)^(1/4) = 64^0.25 = 2√2 ≈ 2.8284
        val expected = 2.82842712474619
        assertEquals(expected, listOf(1.0, 2.0, 4.0, 8.0).geometricMean(), 1e-10)
    }

    @Test
    fun testHarmonicMean() {
        // harmonic mean of 1,2,4 = 3 / (1 + 0.5 + 0.25) = 3/1.75 ≈ 1.7143
        val expected = 3.0 / 1.75
        assertEquals(expected, listOf(1.0, 2.0, 4.0).harmonicMean(), 1e-10)
    }

    @Test
    fun testWeightedMean() {
        val values = listOf(1.0, 2.0, 3.0)
        val weights = listOf(3.0, 1.0, 1.0)
        // (1*3 + 2*1 + 3*1) / (3+1+1) = 8/5 = 1.6
        assertEquals(1.6, values.weightedMean(weights), 1e-10)
    }

    @Test
    fun testMedianOdd() {
        assertEquals(3.0, listOf(5.0, 1.0, 3.0, 2.0, 4.0).median(), 1e-10)
    }

    @Test
    fun testMedianEven() {
        assertEquals(2.5, listOf(1.0, 2.0, 3.0, 4.0).median(), 1e-10)
    }

    @Test
    fun testMode() {
        assertEquals(setOf(3.0), listOf(1.0, 2.0, 3.0, 3.0, 4.0).mode())
        // multiple modes
        assertEquals(setOf(1.0, 2.0), listOf(1.0, 1.0, 2.0, 2.0, 3.0).mode())
    }

    @Test
    fun testModeIntegers() {
        assertEquals(setOf(5), listOf(1, 5, 5, 3).mode())
    }

    @Test
    fun testMeanLargeOffset() {
        val data = DoubleArray(1000) { 1e15 + it.toDouble() }
        val expected = 1e15 + 499.5
        assertEquals(expected, data.mean(), 1e-6)
        assertEquals(expected, data.toList().mean(), 1e-6)
        assertEquals(expected, data.asSequence().mean(), 1e-6)
    }

    @Test
    fun testHarmonicMeanPrecision() {
        // Spread values enough so harmonic < arithmetic is clear,
        // but use large magnitudes to stress compensated summation
        val data = DoubleArray(100) { 1e8 + it.toDouble() * 1e6 }
        val listResult = data.toList().harmonicMean()
        val arrayResult = data.harmonicMean()
        // Both paths should agree to high relative precision
        assertEquals(listResult, arrayResult, arrayResult * 1e-10)
        // Harmonic mean must be less than arithmetic mean
        assertTrue(arrayResult < data.mean())
    }

    @Test
    fun testWeightedMeanPrecision() {
        val values = DoubleArray(100) { 1e14 + it.toDouble() }
        val weights = DoubleArray(100) { 1.0 }
        val expected = 1e14 + 49.5 // uniform weights → arithmetic mean
        assertEquals(expected, values.weightedMean(weights), 1e-4)
        assertEquals(expected, values.toList().weightedMean(weights.toList()), 1e-4)
    }

    // ── median extended tests ─────────────────────────────────────────────

    @Test
    fun medianSingleElement() {
        assertEquals(42.0, doubleArrayOf(42.0).median(), 0.0)
    }

    @Test
    fun medianTwoElements() {
        assertEquals(2.5, doubleArrayOf(1.0, 4.0).median(), 0.0)
    }

    @Test
    fun medianThreeElements() {
        assertEquals(2.0, doubleArrayOf(3.0, 1.0, 2.0).median(), 0.0)
    }

    @Test
    fun medianEmpty() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf().median() }
    }

    @Test
    fun medianEmptyIterable() {
        assertFailsWith<InsufficientDataException> { emptyList<Double>().median() }
    }

    @Test
    fun medianAlreadySorted() {
        assertEquals(3.0, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).median(), 0.0)
    }

    @Test
    fun medianReverseSorted() {
        assertEquals(3.0, doubleArrayOf(5.0, 4.0, 3.0, 2.0, 1.0).median(), 0.0)
    }

    @Test
    fun medianAllIdentical() {
        assertEquals(7.0, doubleArrayOf(7.0, 7.0, 7.0, 7.0).median(), 0.0)
    }

    @Test
    fun medianDoesNotMutateInput() {
        val original = doubleArrayOf(5.0, 3.0, 1.0, 4.0, 2.0)
        val copy = original.copyOf()
        original.median()
        assertContentEquals(copy, original)
    }

    @Test
    fun medianLargeOdd() {
        val rng = Random(42)
        val data = DoubleArray(10001) { rng.nextDouble() }
        val expected = data.sortedArray()[5000]
        assertEquals(expected, data.median(), 0.0)
    }

    @Test
    fun medianLargeEven() {
        val rng = Random(42)
        val data = DoubleArray(10000) { rng.nextDouble() }
        val sorted = data.sortedArray()
        val expected = (sorted[4999] + sorted[5000]) / 2.0
        assertEquals(expected, data.median(), 1e-15)
    }

    @Test
    fun medianNaN() {
        val result = doubleArrayOf(1.0, Double.NaN, 3.0).median()
        // NaN > everything, so sorted = [1.0, 3.0, NaN] → median = 3.0
        assertEquals(3.0, result, 0.0)
    }

    @Test
    fun medianInfinity() {
        assertEquals(2.0, doubleArrayOf(Double.NEGATIVE_INFINITY, 2.0, Double.POSITIVE_INFINITY).median(), 0.0)
        assertEquals(0.5, doubleArrayOf(Double.NEGATIVE_INFINITY, 0.0, 1.0, Double.POSITIVE_INFINITY).median(), 0.0)
    }

    @Test
    fun medianIterableConsistency() {
        val data = doubleArrayOf(9.0, 1.0, 5.0, 3.0, 7.0)
        assertEquals(data.median(), data.toList().median(), 0.0)
    }

    @Test
    fun medianCrossValidation() {
        val rng = Random(777)
        repeat(10) {
            val data = DoubleArray(rng.nextInt(1, 200)) { rng.nextDouble(-100.0, 100.0) }
            val sorted = data.sortedArray()
            val n = sorted.size
            val expected = if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            assertEquals(expected, data.median(), 1e-15, "Failed for size=$n")
        }
    }
}
