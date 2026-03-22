package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class WeightedSamplingTest {

    // --- WeightedCoin ---

    @Test
    fun testWeightedCoin() {
        val coin = WeightedCoin(1.0, Random(42))
        assertTrue(coin.flip()) // p=1, always true
        val coin2 = WeightedCoin(0.0, Random(42))
        assertTrue(!coin2.flip()) // p=0, always false
    }

    @Test
    fun testWeightedCoinDistribution() {
        val coin = WeightedCoin(0.7, Random(123))
        val heads = (1..10000).count { coin.flip() }
        // Should be approximately 7000 ± some margin
        assertTrue(heads in 6700..7300, "Expected ~7000 heads, got $heads")
    }

    @Test
    fun testWeightedCoinInvalidProbabilityThrows() {
        assertFailsWith<InvalidParameterException> { WeightedCoin(-0.1) }
        assertFailsWith<InvalidParameterException> { WeightedCoin(1.1) }
    }

    @Test
    fun testWeightedCoinNaNThrows() {
        assertFailsWith<InvalidParameterException> { WeightedCoin(Double.NaN) }
    }

    // --- WeightedDice ---

    @Test
    fun testWeightedDice() {
        val dice = WeightedDice(mapOf("A" to 1.0, "B" to 0.0), Random(42))
        assertEquals("A", dice.roll()) // Only A has weight
    }

    @Test
    fun testWeightedDiceDistribution() {
        val dice = WeightedDice(mapOf("A" to 3.0, "B" to 1.0), Random(123))
        val counts = mutableMapOf("A" to 0, "B" to 0)
        repeat(10000) { counts[dice.roll()] = counts[dice.roll()]!! + 1 }
        // A should appear ~75% of the time
        // Use a wider range because we rolled twice per iteration (once to count, once to increment)
        // Let's fix: just count properly
        val countsFixed = mutableMapOf("A" to 0, "B" to 0)
        val dice2 = WeightedDice(mapOf("A" to 3.0, "B" to 1.0), Random(456))
        repeat(10000) {
            val result = dice2.roll()
            countsFixed[result] = countsFixed[result]!! + 1
        }
        val aCount = countsFixed["A"]!!
        assertTrue(aCount in 7000..8000, "Expected ~7500 A rolls, got $aCount")
    }

    @Test
    fun testWeightedDiceEmptyThrows() {
        assertFailsWith<InsufficientDataException> {
            WeightedDice(emptyMap<String, Double>())
        }
    }

    @Test
    fun testWeightedDiceNegativeWeightThrows() {
        assertFailsWith<InvalidParameterException> {
            WeightedDice(mapOf("A" to 1.0, "B" to -0.5))
        }
    }

    @Test
    fun testWeightedDiceAllZeroWeightThrows() {
        assertFailsWith<InvalidParameterException> {
            WeightedDice(mapOf("A" to 0.0, "B" to 0.0))
        }
    }

    @Test
    fun testWeightedDiceNaNWeightThrows() {
        assertFailsWith<InvalidParameterException> {
            WeightedDice(mapOf("A" to 1.0, "B" to Double.NaN))
        }
    }

    @Test
    fun testWeightedDiceInfinityWeightThrows() {
        assertFailsWith<InvalidParameterException> {
            WeightedDice(mapOf("A" to Double.POSITIVE_INFINITY))
        }
    }

    // --- randomSample ---

    @Test
    fun testRandomSample() {
        val data = listOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        val sample = data.randomSample(3, Random(42))
        assertEquals(3, sample.size)
        // All elements should come from original
        assertTrue(sample.all { it in data })
        // No duplicates (without replacement)
        assertEquals(sample.size, sample.toSet().size)
    }

    @Test
    fun testRandomSampleZero() {
        val data = listOf(1, 2, 3)
        val sample = data.randomSample(0, Random(42))
        assertTrue(sample.isEmpty())
    }

    @Test
    fun testRandomSampleAll() {
        val data = listOf(1, 2, 3)
        val sample = data.randomSample(3, Random(42))
        assertEquals(3, sample.size)
        assertEquals(data.toSet(), sample.toSet())
    }

    @Test
    fun testRandomSampleNegativeNThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1, 2, 3).randomSample(-1)
        }
    }

    @Test
    fun testRandomSampleExceedsSizeThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1, 2, 3).randomSample(5)
        }
    }

    // --- bootstrapSample (List) ---

    @Test
    fun testBootstrapSample() {
        val data = listOf(1, 2, 3, 4, 5)
        val sample = data.bootstrapSample(10, Random(42))
        assertEquals(10, sample.size)
        assertTrue(sample.all { it in data })
    }

    @Test
    fun testBootstrapSampleZero() {
        val data = listOf(1, 2, 3)
        val sample = data.bootstrapSample(0, Random(42))
        assertTrue(sample.isEmpty())
    }

    @Test
    fun testBootstrapSampleEmptyThrows() {
        assertFailsWith<InsufficientDataException> {
            emptyList<Int>().bootstrapSample(5)
        }
    }

    @Test
    fun testBootstrapSampleNegativeNThrows() {
        assertFailsWith<InvalidParameterException> {
            listOf(1, 2, 3).bootstrapSample(-1)
        }
    }

    // --- bootstrapSample (Iterable) ---

    @Test
    fun testBootstrapSampleIterable() {
        val data: Iterable<Int> = setOf(1, 2, 3, 4, 5)
        val sample = data.bootstrapSample(10, Random(42))
        assertEquals(10, sample.size)
        assertTrue(sample.all { it in data.toList() })
    }

    @Test
    fun testBootstrapSampleIterableEmptyThrows() {
        assertFailsWith<InsufficientDataException> {
            val data: Iterable<Int> = emptySet()
            data.bootstrapSample(5)
        }
    }
}
