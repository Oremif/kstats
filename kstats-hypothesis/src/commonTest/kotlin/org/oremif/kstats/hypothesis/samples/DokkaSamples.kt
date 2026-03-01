package org.oremif.kstats.hypothesis.samples

import org.oremif.kstats.hypothesis.shapiroWilkTest
import org.oremif.kstats.hypothesis.tTest
import kotlin.test.Test
import kotlin.test.assertEquals

class DokkaSamples {

    @Test
    fun dokkaHypothesis() {
        // SampleStart
        val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
        val result = tTest(sample, mu = 5.0)
        result.statistic          // t-statistic
        result.pValue             // p-value
        result.confidenceInterval // 95% CI for the mean
        result.isSignificant()    // true if p < 0.05

        // Normality check before choosing a test
        val sw = shapiroWilkTest(sample)
        if (!sw.isSignificant()) { /* data is consistent with normality */ }
        // SampleEnd
    }
}
