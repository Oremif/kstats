package org.oremif.kstats.distributions.samples

import org.oremif.kstats.distributions.NormalDistribution
import org.oremif.kstats.distributions.PoissonDistribution
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class DokkaSamples {

    @Test
    fun dokkaDistributions() {
        // SampleStart
        val normal = NormalDistribution(mu = 0.0, sigma = 1.0)
        normal.pdf(0.0)           // 0.3989...
        normal.cdf(1.96)          // 0.975...
        normal.quantile(0.975)    // 1.96
        normal.sample(Random(42)) // a single random draw

        val poisson = PoissonDistribution(rate = 4.0)
        poisson.pmf(3)            // P(X = 3)
        poisson.cdf(5)            // P(X <= 5)
        poisson.mean              // 4.0
        // SampleEnd
        assertEquals(0.3989, normal.pdf(0.0), 1e-4)
        assertEquals(0.975, normal.cdf(1.96), 1e-4)
        assertEquals(1.96, normal.quantile(0.975), 1e-4)
        assertEquals(4.0, poisson.mean, 1e-4)
    }
}
