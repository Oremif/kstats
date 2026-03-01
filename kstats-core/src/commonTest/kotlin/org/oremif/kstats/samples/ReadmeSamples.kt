package org.oremif.kstats.samples

import org.oremif.kstats.descriptive.describe
import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.descriptive.standardDeviation
import org.oremif.kstats.descriptive.skewness
import org.oremif.kstats.descriptive.OnlineStatistics
import kotlin.test.Test
import kotlin.test.assertEquals

class ReadmeSamples {

    @Test
    fun quickstart() {
        // SampleStart
        val data = doubleArrayOf(2.0, 4.0, 4.0, 5.0, 7.0, 9.0)
        val summary = data.describe()
        // => DescriptiveStatistics(count=6, mean=5.17, median=4.5, standardDeviation=2.48, ...)

        data.mean()               // => 5.1667
        data.standardDeviation()  // => 2.4833
        data.skewness()           // => 0.3942
        // SampleEnd
        assertEquals(5.1667, data.mean(), 1e-4)
        assertEquals(2.4833, data.standardDeviation(), 1e-4)
        assertEquals(0.5398, data.skewness(), 1e-4)
    }

    @Test
    fun coreDescriptiveStats() {
        // SampleStart
        val data = doubleArrayOf(2.0, 4.0, 4.0, 5.0, 7.0, 9.0)
        val summary = data.describe()
        summary.mean              // => 5.1667
        summary.median            // => 4.5
        summary.standardDeviation // => 2.4833

        val stats = OnlineStatistics()
        stats.addAll(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0))
        stats.mean                // => 3.0
        stats.standardDeviation() // => 1.5811
        // SampleEnd
        assertEquals(5.1667, summary.mean, 1e-4)
        assertEquals(4.5, summary.median, 1e-4)
        assertEquals(2.4833, summary.standardDeviation, 1e-4)
        assertEquals(3.0, stats.mean, 1e-4)
        assertEquals(1.5811, stats.standardDeviation(), 1e-4)
    }
}
