package org.oremif.kstats.correlation.samples

import org.oremif.kstats.correlation.pearsonCorrelation
import org.oremif.kstats.correlation.simpleLinearRegression
import kotlin.test.Test

class DokkaSamples {

    @Test
    fun dokkaCorrelation() {
        // SampleStart
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1)

        val r = pearsonCorrelation(x, y)
        r.coefficient // 0.999...
        r.pValue      // < 0.001

        val reg = simpleLinearRegression(x, y)
        reg.slope        // ~2.0
        reg.intercept    // ~0.04
        reg.rSquared     // 0.999...
        reg.predict(6.0) // predicted y for x = 6
        // SampleEnd
    }
}
