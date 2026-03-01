package org.oremif.kstats.correlation.samples

import org.oremif.kstats.correlation.pearsonCorrelation
import org.oremif.kstats.correlation.simpleLinearRegression
import kotlin.test.Test
import kotlin.test.assertEquals

class ReadmeSamples {

    @Test
    fun correlationPearsonRegression() {
        // SampleStart
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1)

        val r = pearsonCorrelation(x, y)
        r.coefficient                    // => 0.9987
        r.pValue                         // => 0.0001

        val reg = simpleLinearRegression(x, y)
        reg.slope                        // => 1.99
        reg.rSquared                     // => 0.9973
        reg.predict(6.0)                 // => 11.99
        // SampleEnd
        assertEquals(0.9987, r.coefficient, 1e-4)
        assertEquals(0.0001, r.pValue, 1e-4)
        assertEquals(1.99, reg.slope, 0.01)
        assertEquals(0.9973, reg.rSquared, 1e-4)
        assertEquals(11.99, reg.predict(6.0), 0.01)
    }
}
