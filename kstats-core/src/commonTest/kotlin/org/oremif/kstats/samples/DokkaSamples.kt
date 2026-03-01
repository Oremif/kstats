package org.oremif.kstats.samples

import org.oremif.kstats.descriptive.OnlineStatistics
import org.oremif.kstats.descriptive.describe
import kotlin.test.Test
import kotlin.test.assertEquals

class DokkaSamples {

    @Test
    fun dokkaCoreDescriptive() {
        // SampleStart
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)

        val stats = data.describe()
        stats.mean              // 5.0
        stats.standardDeviation // 2.0
        stats.median            // 4.5
        stats.skewness          // 0.656...

        // Streaming computation — no need to hold all data in memory
        val online = OnlineStatistics()
        online.addAll(data)
        online.mean       // 5.0
        online.variance() // 4.571...
        // SampleEnd
        assertEquals(5.0, stats.mean, 1e-4)
        assertEquals(2.1381, stats.standardDeviation, 1e-4)
        assertEquals(4.5, stats.median, 1e-4)
        assertEquals(0.8185, stats.skewness, 1e-4)
        assertEquals(5.0, online.mean, 1e-4)
        assertEquals(4.571, online.variance(), 1e-3)
    }
}
