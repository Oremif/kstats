package org.oremif.kstats.sampling.samples

import org.oremif.kstats.sampling.WeightedDice
import org.oremif.kstats.sampling.minMaxNormalize
import org.oremif.kstats.sampling.rank
import org.oremif.kstats.sampling.zScore
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals

class DokkaSamples {

    @Test
    fun dokkaSampling() {
        // SampleStart
        val data = doubleArrayOf(3.0, 1.0, 4.0, 1.0, 5.0)

        data.rank()              // [3.0, 1.5, 4.0, 1.5, 5.0] (average ties)
        data.zScore()            // standardized to mean=0, sd=1
        data.minMaxNormalize()   // scaled to [0.0, 1.0]

        val die = WeightedDice(mapOf("A" to 0.7, "B" to 0.2, "C" to 0.1), Random(42))
        die.roll()               // "A" (most likely)
        // SampleEnd
        assertContentEquals(doubleArrayOf(3.0, 1.5, 4.0, 1.5, 5.0), data.rank())
    }
}
