package org.oremif.kstats.sampling

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

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

class NormalizationTest {

    @Test
    fun testZScore() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val z = data.zScore()
        // Mean of z-scores should be ~0
        assertEquals(0.0, z.average(), 1e-10)
    }

    @Test
    fun testMinMaxNormalize() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val norm = data.minMaxNormalize()
        assertEquals(0.0, norm[0], 1e-10)
        assertEquals(1.0, norm[4], 1e-10)
        assertEquals(0.5, norm[2], 1e-10)
    }

    @Test
    fun testMinMaxNormalizeCustomRange() {
        val data = doubleArrayOf(0.0, 5.0, 10.0)
        val norm = data.minMaxNormalize(-1.0, 1.0)
        assertEquals(-1.0, norm[0], 1e-10)
        assertEquals(0.0, norm[1], 1e-10)
        assertEquals(1.0, norm[2], 1e-10)
    }
}

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

class WeightedSamplingTest {

    @Test
    fun testWeightedCoin() {
        val coin = WeightedCoin(1.0, Random(42))
        assertTrue(coin.flip()) // p=1, always true
        val coin2 = WeightedCoin(0.0, Random(42))
        assertTrue(!coin2.flip()) // p=0, always false
    }

    @Test
    fun testWeightedDice() {
        val dice = WeightedDice(mapOf("A" to 1.0, "B" to 0.0), Random(42))
        assertEquals("A", dice.roll()) // Only A has weight
    }

    @Test
    fun testRandomSample() {
        val data = listOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        val sample = data.randomSample(3, Random(42))
        assertEquals(3, sample.size)
        // All elements should come from original
        assertTrue(sample.all { it in data })
    }

    @Test
    fun testBootstrapSample() {
        val data = listOf(1, 2, 3, 4, 5)
        val sample = data.bootstrapSample(10, Random(42))
        assertEquals(10, sample.size)
        assertTrue(sample.all { it in data })
    }
}
