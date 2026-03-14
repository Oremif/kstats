package org.oremif.kstats.sampling

import kotlin.test.Test
import kotlin.test.assertEquals

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
    fun testRankDense() {
        val data = doubleArrayOf(3.0, 1.0, 1.0, 5.0)
        val ranks = data.rank(TieMethod.DENSE)
        assertEquals(2.0, ranks[0], 1e-10) // 3 is 2nd unique value
        assertEquals(1.0, ranks[1], 1e-10) // 1 is 1st unique value
        assertEquals(1.0, ranks[2], 1e-10)
        assertEquals(3.0, ranks[3], 1e-10) // 5 is 3rd unique value
    }

    @Test
    fun testPercentileRank() {
        val data = doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0)
        val pr = data.percentileRank()
        assertEquals(0.0, pr[0], 1e-10)
        assertEquals(100.0, pr[4], 1e-10)
    }
}
