package org.oremif.kstats.sampling

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

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
