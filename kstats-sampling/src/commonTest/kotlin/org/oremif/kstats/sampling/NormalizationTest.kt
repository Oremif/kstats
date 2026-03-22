package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class NormalizationTest {

    // --- z-score (DoubleArray) ---

    @Test
    fun testZScore() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val z = data.zScore()
        // Mean of z-scores should be ~0
        assertEquals(0.0, z.average(), 1e-10)
        // First element should be negative, last positive
        assertEquals(-z[4], z[0], 1e-10)
    }

    @Test
    fun testZScoreInsufficientDataThrows() {
        assertFailsWith<InsufficientDataException> {
            doubleArrayOf(1.0).zScore()
        }
        assertFailsWith<InsufficientDataException> {
            doubleArrayOf().zScore()
        }
    }

    @Test
    fun testZScoreDegenerateThrows() {
        assertFailsWith<DegenerateDataException> {
            doubleArrayOf(5.0, 5.0, 5.0).zScore()
        }
    }

    @Test
    fun testZScoreNaNThrows() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, Double.NaN, 3.0).zScore()
        }
    }

    @Test
    fun testZScoreInfinityThrows() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0).zScore()
        }
    }

    // --- z-score (Iterable) ---

    @Test
    fun testZScoreIterable() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val z = data.zScore()
        assertEquals(0.0, z.average(), 1e-10)
    }

    @Test
    fun testZScoreIterableInsufficientDataThrows() {
        assertFailsWith<InsufficientDataException> {
            listOf(1.0).zScore()
        }
    }

    @Test
    fun testZScoreIterableDegenerateThrows() {
        assertFailsWith<DegenerateDataException> {
            listOf(5.0, 5.0, 5.0).zScore()
        }
    }

    @Test
    fun testZScoreIterableNaNThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1.0, Double.NaN, 3.0).zScore()
        }
    }

    // --- minMaxNormalize ---

    @Test
    fun testMinMaxNormalize() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val norm = data.minMaxNormalize()
        assertEquals(0.0, norm[0], 1e-10)
        assertEquals(1.0, norm[4], 1e-10)
        assertEquals(0.5, norm[2], 1e-10)
    }

    @Test
    fun testMinMaxNormalizeSingleElement() {
        val data = doubleArrayOf(42.0)
        val norm = data.minMaxNormalize()
        assertEquals(0.0, norm[0], 1e-10) // range is 0, maps to 0
    }

    @Test
    fun testMinMaxNormalizeAllSame() {
        val data = doubleArrayOf(3.0, 3.0, 3.0)
        val norm = data.minMaxNormalize()
        for (v in norm) assertEquals(0.0, v, 1e-10)
    }

    @Test
    fun testMinMaxNormalizeEmptyThrows() {
        assertFailsWith<InsufficientDataException> {
            doubleArrayOf().minMaxNormalize()
        }
    }

    @Test
    fun testMinMaxNormalizeNaNThrows() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, Double.NaN, 3.0).minMaxNormalize()
        }
    }

    @Test
    fun testMinMaxNormalizeInfinityThrows() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, Double.POSITIVE_INFINITY).minMaxNormalize()
        }
    }

    // --- minMaxNormalize(newMin, newMax) ---

    @Test
    fun testMinMaxNormalizeCustomRange() {
        val data = doubleArrayOf(0.0, 5.0, 10.0)
        val norm = data.minMaxNormalize(-1.0, 1.0)
        assertEquals(-1.0, norm[0], 1e-10)
        assertEquals(0.0, norm[1], 1e-10)
        assertEquals(1.0, norm[2], 1e-10)
    }

    @Test
    fun testMinMaxNormalizeCustomRangeAllSame() {
        val data = doubleArrayOf(7.0, 7.0, 7.0)
        val norm = data.minMaxNormalize(-1.0, 1.0)
        for (v in norm) assertEquals(-1.0, v, 1e-10) // maps to newMin
    }

    @Test
    fun testMinMaxNormalizeCustomRangeInvalidThrows() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, 2.0).minMaxNormalize(5.0, 3.0) // newMin > newMax
        }
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, 2.0).minMaxNormalize(1.0, 1.0) // newMin == newMax
        }
    }

    @Test
    fun testMinMaxNormalizeCustomRangeEmptyThrows() {
        assertFailsWith<InsufficientDataException> {
            doubleArrayOf().minMaxNormalize(-1.0, 1.0)
        }
    }

    @Test
    fun testMinMaxNormalizeCustomRangeNaNBoundsThrows() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, 2.0).minMaxNormalize(Double.NaN, 1.0)
        }
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, 2.0).minMaxNormalize(0.0, Double.NaN)
        }
    }

    @Test
    fun testMinMaxNormalizeCustomRangeInfBoundsThrows() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, 2.0).minMaxNormalize(Double.NEGATIVE_INFINITY, 1.0)
        }
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, 2.0).minMaxNormalize(0.0, Double.POSITIVE_INFINITY)
        }
    }

    @Test
    fun testMinMaxNormalizeCustomRangeNaNDataThrows() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, Double.NaN).minMaxNormalize(-1.0, 1.0)
        }
    }

    // --- minMaxNormalize (Iterable) ---

    @Test
    fun testMinMaxNormalizeIterable() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val norm = data.minMaxNormalize()
        assertEquals(0.0, norm[0], 1e-10)
        assertEquals(1.0, norm[4], 1e-10)
        assertEquals(0.5, norm[2], 1e-10)
    }

    @Test
    fun testMinMaxNormalizeIterableAllSame() {
        val norm = listOf(3.0, 3.0, 3.0).minMaxNormalize()
        for (v in norm) assertEquals(0.0, v, 1e-10)
    }

    @Test
    fun testMinMaxNormalizeIterableEmptyThrows() {
        assertFailsWith<InsufficientDataException> {
            emptyList<Double>().minMaxNormalize()
        }
    }

    @Test
    fun testMinMaxNormalizeIterableCustomRange() {
        val data = listOf(0.0, 5.0, 10.0)
        val norm = data.minMaxNormalize(-1.0, 1.0)
        assertEquals(-1.0, norm[0], 1e-10)
        assertEquals(0.0, norm[1], 1e-10)
        assertEquals(1.0, norm[2], 1e-10)
    }

    @Test
    fun testMinMaxNormalizeIterableCustomRangeInvalidThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1.0, 2.0).minMaxNormalize(5.0, 3.0)
        }
    }

    // --- Large array tests ---

    @Test
    fun testZScoreLargeArray() {
        val data = DoubleArray(5000) { it.toDouble() }
        val z = data.zScore()
        assertEquals(0.0, z.average(), 1e-10)
    }

    @Test
    fun testMinMaxNormalizeLargeArray() {
        val data = DoubleArray(5000) { it.toDouble() }
        val norm = data.minMaxNormalize()
        assertEquals(0.0, norm[0], 1e-10)
        assertEquals(1.0, norm[4999], 1e-10)
    }
}
