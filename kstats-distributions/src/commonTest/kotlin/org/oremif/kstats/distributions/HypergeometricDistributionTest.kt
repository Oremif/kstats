package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class HypergeometricDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPmfKnownValues() {
        val d = HypergeometricDistribution(50, 20, 10)
        // scipy: stats.hypergeom(50, 20, 10).pmf(k)
        assertEquals(0.108257947418883, d.pmf(2), 1e-10)
        assertEquals(0.280058603105371, d.pmf(4), 1e-10)
        assertEquals(0.215085007184925, d.pmf(5), 1e-10)
        assertEquals(0.0, d.pmf(-1), 1e-15)
        assertEquals(0.0, d.pmf(11), 1e-15)
    }

    @Test
    fun testCdfKnownValues() {
        val d = HypergeometricDistribution(50, 20, 10)
        assertEquals(0.139038657380907, d.cdf(2), 1e-10)
        assertEquals(0.645026889882208, d.cdf(4), 1e-10)
    }

    @Test
    fun testLogPmfKnownValues() {
        val d = HypergeometricDistribution(50, 20, 10)
        assertEquals(-1.27275640090751, d.logPmf(4), 1e-10)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(-1))
    }

    @Test
    fun testSfKnownValues() {
        val d = HypergeometricDistribution(50, 20, 10)
        assertEquals(0.354973110117792, d.sf(4), 1e-10)
    }

    // --- Quantile ---

    @Test
    fun testQuantileIntKnownValues() {
        val d = HypergeometricDistribution(50, 20, 10)
        assertEquals(3, d.quantileInt(0.25))
        assertEquals(4, d.quantileInt(0.5))
        assertEquals(5, d.quantileInt(0.75))
    }

    // --- Moments ---

    @Test
    fun testMoments() {
        val d = HypergeometricDistribution(50, 20, 10)
        assertEquals(4.0, d.mean, 1e-10)
        assertEquals(1.95918367346939, d.variance, 1e-10)
        // scipy: stats.hypergeom(50, 20, 10).stats(moments='sk')
        assertEquals(0.08930431353897, d.skewness, 1e-8)
        assertEquals(-0.131621232269504, d.kurtosis, 1e-6)
    }

    @Test
    fun testMomentsN20K7n12() {
        val d = HypergeometricDistribution(20, 7, 12)
        assertEquals(4.2, d.mean, 1e-10)
        assertEquals(1.14947368421053, d.variance, 1e-10)
        assertEquals(-0.0621812179560988, d.skewness, 1e-8)
        assertEquals(-0.15266106442577, d.kurtosis, 1e-6)
    }

    // --- Entropy ---

    @Test
    fun testEntropy() {
        // scipy: stats.hypergeom(50, 20, 10).entropy()
        assertEquals(1.75382387925846, HypergeometricDistribution(50, 20, 10).entropy, 1e-8)
        // scipy: stats.hypergeom(20, 7, 12).entropy()
        assertEquals(1.48738058584421, HypergeometricDistribution(20, 7, 12).entropy, 1e-8)
    }

    // --- Edge cases ---

    @Test
    fun testSkewnessSmallN() {
        assertTrue(HypergeometricDistribution(2, 1, 1).skewness.isNaN())
    }

    @Test
    fun testKurtosisSmallN() {
        assertTrue(HypergeometricDistribution(3, 1, 1).kurtosis.isNaN())
    }

    @Test
    fun testDeterministicDraw() {
        // n=K=N → always get all successes
        val d = HypergeometricDistribution(5, 5, 5)
        assertEquals(1.0, d.pmf(5), 1e-15)
        assertEquals(0.0, d.pmf(4), 1e-15)
    }

    @Test
    fun testEmptyPopulation() {
        // population=0: degenerate distribution, always 0 successes
        val d = HypergeometricDistribution(0, 0, 0)
        assertEquals(1.0, d.pmf(0), 1e-15)
        assertEquals(0.0, d.mean, 1e-15)
        assertEquals(0.0, d.variance, 1e-15)
        assertEquals(0.0, d.entropy, 1e-15)
        assertTrue(d.skewness.isNaN())
        assertTrue(d.kurtosis.isNaN())
        assertEquals(0, d.sample(kotlin.random.Random(0)))
    }

    @Test
    fun testPopulationOne() {
        // population=1, successes=1, draws=1: degenerate at k=1
        val d1 = HypergeometricDistribution(1, 1, 1)
        assertEquals(1.0, d1.mean, 1e-15)
        assertEquals(0.0, d1.variance, 1e-15)
        assertTrue(d1.skewness.isNaN())
        assertTrue(d1.kurtosis.isNaN())

        // population=1, successes=0, draws=1: degenerate at k=0
        val d2 = HypergeometricDistribution(1, 0, 1)
        assertEquals(0.0, d2.mean, 1e-15)
        assertEquals(0.0, d2.variance, 1e-15)
    }

    @Test
    fun testDegenerateMomentsZeroDraws() {
        // draws=0 with large N: degenerate at k=0
        val d = HypergeometricDistribution(100, 50, 0)
        assertEquals(0.0, d.mean, 1e-15)
        assertEquals(0.0, d.variance, 1e-15)
        assertTrue(d.skewness.isNaN())
        assertTrue(d.kurtosis.isNaN())
    }

    @Test
    fun testDegenerateMomentsAllSuccesses() {
        // successes=population: always draw all successes, degenerate at k=draws
        val d = HypergeometricDistribution(10, 10, 7)
        assertEquals(7.0, d.mean, 1e-15)
        assertEquals(0.0, d.variance, 1e-15)
        assertTrue(d.skewness.isNaN())
        assertTrue(d.kurtosis.isNaN())
    }

    @Test
    fun testDegenerateMomentsZeroSuccesses() {
        // successes=0: always 0, degenerate at k=0
        val d = HypergeometricDistribution(10, 0, 5)
        assertEquals(0.0, d.mean, 1e-15)
        assertEquals(0.0, d.variance, 1e-15)
        assertTrue(d.skewness.isNaN())
        assertTrue(d.kurtosis.isNaN())
    }

    @Test
    fun testDegenerateSample() {
        // Degenerate distributions should always sample the single value
        val d1 = HypergeometricDistribution(5, 5, 5)
        val rng = kotlin.random.Random(42)
        repeat(10) { assertEquals(5, d1.sample(rng)) }

        val d2 = HypergeometricDistribution(10, 0, 5)
        repeat(10) { assertEquals(0, d2.sample(rng)) }
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { HypergeometricDistribution(-1, 5, 5) }
        assertFailsWith<InvalidParameterException> { HypergeometricDistribution(10, 11, 5) }
        assertFailsWith<InvalidParameterException> { HypergeometricDistribution(10, 5, 11) }
        assertFailsWith<InvalidParameterException> { HypergeometricDistribution(10, -1, 5) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = HypergeometricDistribution(50, 20, 10)
        assertFailsWith<InvalidParameterException> { d.quantileInt(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantileInt(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testExpLogPmfConsistency() {
        val d = HypergeometricDistribution(50, 20, 10)
        for (k in 0..10) {
            assertEquals(d.pmf(k), exp(d.logPmf(k)), 1e-12, "exp(logPmf($k)) ≈ pmf($k)")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val d = HypergeometricDistribution(50, 20, 10)
        for (k in -1..11) {
            assertEquals(1.0, d.sf(k) + d.cdf(k), 1e-10, "sf($k) + cdf($k) ≈ 1")
        }
    }

    @Test
    fun testSampleStats() {
        val d = HypergeometricDistribution(50, 20, 10)
        val rng = kotlin.random.Random(42)
        val samples = d.sample(100_000, rng)
        val doubles = samples.map { it.toDouble() }
        val sampleMean = doubles.average()
        assertEquals(4.0, sampleMean, 0.15, "sample mean ≈ 4.0")
        val sampleVar = doubles.sumOf { (it - sampleMean) * (it - sampleMean) } / (doubles.size - 1)
        assertEquals(d.variance, sampleVar, maxOf(d.variance * 0.1, 0.05), "sample variance ≈ ${d.variance}")
    }

    @Test
    fun testCdfMonotonicity() {
        val d = HypergeometricDistribution(50, 20, 10)
        var prev = 0.0
        for (k in 0..10) {
            val cdfVal = d.cdf(k)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    // --- Large parameters (naive summation underflows) ---

    @Test
    fun testCdfLargeParameters() {
        val d = HypergeometricDistribution(1000, 500, 300)
        // scipy: stats.hypergeom(1000, 500, 300).cdf(k)
        assertEquals(2.243836272440083e-05, d.cdf(120), 1e-10)
        assertEquals(9.488468056336551e-02, d.cdf(140), 1e-10)
        assertEquals(5.275037540386245e-01, d.cdf(150), 1e-10)
        assertEquals(9.263829302706118e-01, d.cdf(160), 1e-10)
        assertEquals(9.999877784284262e-01, d.cdf(180), 1e-10)
    }

    @Test
    fun testSfLargeParameters() {
        val d = HypergeometricDistribution(1000, 500, 300)
        // scipy: stats.hypergeom(1000, 500, 300).sf(k)
        assertEquals(9.999775616372757e-01, d.sf(120), 1e-10)
        assertEquals(9.051153194366346e-01, d.sf(140), 1e-10)
        assertEquals(4.724962459613756e-01, d.sf(150), 1e-10)
        assertEquals(7.361706972938824e-02, d.sf(160), 1e-10)
        assertEquals(1.222157157383291e-05, d.sf(180), 1e-10)
    }

    @Test
    fun testSfPlusCdfEqualsOneLargeParams() {
        val d = HypergeometricDistribution(1000, 500, 300)
        for (k in listOf(120, 140, 150, 160, 180)) {
            assertEquals(1.0, d.sf(k) + d.cdf(k), 1e-12, "sf($k) + cdf($k) ≈ 1")
        }
    }

    @Test
    fun testCdfQuantileRoundTrip() {
        val d = HypergeometricDistribution(50, 20, 10)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            val k = d.quantileInt(p)
            assertTrue(d.cdf(k) >= p, "cdf(quantileInt($p)) >= $p")
            if (k > 0) assertTrue(d.cdf(k - 1) < p, "cdf(quantileInt($p)-1) < $p")
        }
    }

    @Test
    fun testPmfSumsToOne() {
        val d = HypergeometricDistribution(50, 20, 10)
        val total = (0..10).sumOf { d.pmf(it) }
        assertEquals(1.0, total, 1e-10)
    }
}
