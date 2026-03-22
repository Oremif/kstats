package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class RankingTest {

    @Test
    fun testRankAverage() {
        val data = doubleArrayOf(3.0, 1.0, 4.0, 1.0, 5.0)
        val ranks = data.rank(TieMethod.AVERAGE)
        // sorted: 1,1,3,4,5 -> ranks: 1.5,1.5,3,4,5
        assertEquals(3.0, ranks[0], 1e-10) // value 3 -> rank 3
        assertEquals(1.5, ranks[1], 1e-10) // value 1 -> rank 1.5
        assertEquals(4.0, ranks[2], 1e-10) // value 4 -> rank 4
        assertEquals(1.5, ranks[3], 1e-10) // value 1 -> rank 1.5
        assertEquals(5.0, ranks[4], 1e-10) // value 5 -> rank 5
    }

    @Test
    fun testRankMin() {
        val data = doubleArrayOf(3.0, 1.0, 1.0, 5.0)
        val ranks = data.rank(TieMethod.MIN)
        assertEquals(3.0, ranks[0], 1e-10) // value 3
        assertEquals(1.0, ranks[1], 1e-10) // value 1 (min of 1,2)
        assertEquals(1.0, ranks[2], 1e-10) // value 1
        assertEquals(4.0, ranks[3], 1e-10) // value 5
    }

    @Test
    fun testRankMax() {
        val data = doubleArrayOf(3.0, 1.0, 1.0, 5.0)
        val ranks = data.rank(TieMethod.MAX)
        assertEquals(3.0, ranks[0], 1e-10) // value 3 -> rank 3
        assertEquals(2.0, ranks[1], 1e-10) // value 1 -> max of positions 1,2
        assertEquals(2.0, ranks[2], 1e-10) // value 1 -> max of positions 1,2
        assertEquals(4.0, ranks[3], 1e-10) // value 5 -> rank 4
    }

    @Test
    fun testRankDense() {
        val data = doubleArrayOf(3.0, 1.0, 1.0, 5.0)
        val ranks = data.rank(TieMethod.DENSE)
        assertEquals(2.0, ranks[0], 1e-10) // 3 is 2nd unique value
        assertEquals(1.0, ranks[1], 1e-10) // 1 is 1st unique value
        assertEquals(1.0, ranks[2], 1e-10)
        assertEquals(3.0, ranks[3], 1e-10) // 5 is 3rd unique value
    }

    @Test
    fun testRankOrdinal() {
        val data = doubleArrayOf(3.0, 1.0, 1.0, 5.0)
        val ranks = data.rank(TieMethod.ORDINAL)
        assertEquals(3.0, ranks[0], 1e-10) // value 3 -> position 3
        assertEquals(1.0, ranks[1], 1e-10) // first 1 -> position 1
        assertEquals(2.0, ranks[2], 1e-10) // second 1 -> position 2
        assertEquals(4.0, ranks[3], 1e-10) // value 5 -> position 4
    }

    @Test
    fun testRankSingleElement() {
        val data = doubleArrayOf(42.0)
        val ranks = data.rank(TieMethod.AVERAGE)
        assertEquals(1.0, ranks[0], 1e-10)
    }

    @Test
    fun testRankAllTied() {
        val data = doubleArrayOf(5.0, 5.0, 5.0, 5.0)
        val avg = data.rank(TieMethod.AVERAGE)
        // All tied at positions 1,2,3,4 -> average = 2.5
        for (r in avg) assertEquals(2.5, r, 1e-10)

        val min = data.rank(TieMethod.MIN)
        for (r in min) assertEquals(1.0, r, 1e-10)

        val max = data.rank(TieMethod.MAX)
        for (r in max) assertEquals(4.0, r, 1e-10)

        val dense = data.rank(TieMethod.DENSE)
        for (r in dense) assertEquals(1.0, r, 1e-10)
    }

    @Test
    fun testRankNoTies() {
        val data = doubleArrayOf(4.0, 2.0, 1.0, 3.0)
        val ranks = data.rank(TieMethod.AVERAGE)
        assertEquals(4.0, ranks[0], 1e-10)
        assertEquals(2.0, ranks[1], 1e-10)
        assertEquals(1.0, ranks[2], 1e-10)
        assertEquals(3.0, ranks[3], 1e-10)
    }

    @Test
    fun testRankEmptyThrows() {
        assertFailsWith<InsufficientDataException> {
            doubleArrayOf().rank()
        }
    }

    @Test
    fun testRankNaNThrows() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, Double.NaN, 3.0).rank()
        }
    }

    @Test
    fun testRankPositiveInfinityThrows() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0).rank()
        }
    }

    @Test
    fun testRankNegativeInfinityThrows() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(Double.NEGATIVE_INFINITY, 1.0).rank()
        }
    }

    @Test
    fun testPercentileRank() {
        val data = doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0)
        val pr = data.percentileRank()
        assertEquals(0.0, pr[0], 1e-10)
        assertEquals(25.0, pr[1], 1e-10)
        assertEquals(50.0, pr[2], 1e-10)
        assertEquals(75.0, pr[3], 1e-10)
        assertEquals(100.0, pr[4], 1e-10)
    }

    @Test
    fun testPercentileRankSingleElement() {
        val data = doubleArrayOf(42.0)
        val pr = data.percentileRank()
        assertEquals(0.0, pr[0], 1e-10)
    }

    @Test
    fun testPercentileRankEmptyThrows() {
        assertFailsWith<InsufficientDataException> {
            doubleArrayOf().percentileRank()
        }
    }

    @Test
    fun testPercentileRankNaNThrows() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, Double.NaN).percentileRank()
        }
    }
}
