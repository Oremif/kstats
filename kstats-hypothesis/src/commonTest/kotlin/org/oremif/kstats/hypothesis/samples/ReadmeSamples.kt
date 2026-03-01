package org.oremif.kstats.hypothesis.samples

import org.oremif.kstats.hypothesis.tTest
import kotlin.test.Test
import kotlin.test.assertEquals

class ReadmeSamples {

    @Test
    fun hypothesisTTest() {
        // SampleStart
        val sample = doubleArrayOf(2.0, 4.0, 4.0, 5.0, 7.0, 9.0)
        val result = tTest(sample, mu = 5.0)
        result.statistic                 // => 0.1644
        result.pValue                    // => 0.8759
        result.isSignificant(alpha = 0.05) // => false
        // SampleEnd
        assertEquals(0.1644, result.statistic, 1e-4)
        assertEquals(0.8759, result.pValue, 1e-4)
        assertEquals(false, result.isSignificant(alpha = 0.05))
    }
}
