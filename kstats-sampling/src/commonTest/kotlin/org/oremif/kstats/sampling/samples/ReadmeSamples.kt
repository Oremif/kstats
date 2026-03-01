package org.oremif.kstats.sampling.samples

import org.oremif.kstats.sampling.WeightedDice
import org.oremif.kstats.sampling.bootstrapSample
import org.oremif.kstats.sampling.rank
import org.oremif.kstats.sampling.zScore
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class ReadmeSamples {

    @Test
    fun samplingRankNormalize() {
        // SampleStart
        val data = doubleArrayOf(3.0, 1.0, 4.0, 1.0, 5.0)
        data.rank()                      // => [3.0, 1.5, 4.0, 1.5, 5.0]
        data.zScore()                    // => [-0.16, -1.47, 0.49, -1.47, 1.14]

        listOf(1, 2, 3, 4, 5).bootstrapSample(10, Random(42))

        val dice = WeightedDice(mapOf("A" to 3.0, "B" to 1.0))
        dice.roll()                      // => "A" (75% probability)
        // SampleEnd
        assertContentEquals(doubleArrayOf(3.0, 1.5, 4.0, 1.5, 5.0), data.rank())
        val zScores = data.zScore()
        assertEquals(0.1118, zScores[0], 1e-3)
        assertEquals(-1.0062, zScores[1], 1e-3)
        assertEquals(0.6708, zScores[2], 1e-3)
        assertEquals(-1.0062, zScores[3], 1e-3)
        assertEquals(1.2298, zScores[4], 1e-3)
    }
}
