package org.oremif.kstats.distributions.samples

import org.oremif.kstats.distributions.NormalDistribution
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class ReadmeSamples {

    @Test
    fun distributionsNormal() {
        // SampleStart
        val normal = NormalDistribution(mu = 0.0, sigma = 1.0)
        normal.pdf(0.0)                  // => 0.3989
        normal.cdf(1.96)                 // => 0.9750
        normal.quantile(0.975)           // => 1.9600
        normal.sample(5, Random(42))     // => [0.11, -0.87, ...]
        // SampleEnd
        assertEquals(0.3989, normal.pdf(0.0), 1e-4)
        assertEquals(0.9750, normal.cdf(1.96), 1e-4)
        assertEquals(1.9600, normal.quantile(0.975), 1e-4)
    }
}
