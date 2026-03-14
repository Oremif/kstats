package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BinomialDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPmfKnownValues() {
        val d = BinomialDistribution(10, 0.3)
        // scipy: stats.binom(10, 0.3).pmf(k)
        assertEquals(0.0282475249, d.pmf(0), 1e-10)
        assertEquals(0.266827932, d.pmf(3), 1e-9)
        assertEquals(0.1029193452, d.pmf(5), 1e-10)
        assertEquals(5.9049e-06, d.pmf(10), 1e-10)
        assertEquals(0.0, d.pmf(-1), 1e-15)
        assertEquals(0.0, d.pmf(11), 1e-15)
    }

    @Test
    fun testCdfKnownValues() {
        val d = BinomialDistribution(10, 0.3)
        assertEquals(0.0282475249, d.cdf(0), 1e-10)
        assertEquals(0.6496107184, d.cdf(3), 1e-10)
        assertEquals(0.9526510126, d.cdf(5), 1e-10)
        assertEquals(1.0, d.cdf(10), 1e-15)
        assertEquals(0.0, d.cdf(-1), 1e-15)
    }

    @Test
    fun testLogPmfKnownValues() {
        val d = BinomialDistribution(10, 0.3)
        assertEquals(-1.32115127776689, d.logPmf(3), 1e-10)
        assertEquals(-3.56674943938732, d.logPmf(0), 1e-10)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(-1))
    }

    @Test
    fun testSfKnownValues() {
        val d = BinomialDistribution(10, 0.3)
        // scipy: stats.binom(10, 0.3).sf(k)
        assertEquals(0.3503892816, d.sf(3), 1e-10)
        assertEquals(0.0473489874, d.sf(5), 1e-10)
        assertEquals(1.0, d.sf(-1), 1e-15)
        assertEquals(0.0, d.sf(10), 1e-15)
    }

    // --- Quantile ---

    @Test
    fun testQuantileIntKnownValues() {
        val d = BinomialDistribution(10, 0.3)
        // scipy: stats.binom(10, 0.3).ppf(p)
        assertEquals(2, d.quantileInt(0.25))
        assertEquals(3, d.quantileInt(0.5))
        assertEquals(4, d.quantileInt(0.75))
        assertEquals(7, d.quantileInt(0.99))
        assertEquals(0, d.quantileInt(0.0))
        assertEquals(10, d.quantileInt(1.0))
    }

    // --- Moments ---

    @Test
    fun testMoments() {
        val d = BinomialDistribution(10, 0.3)
        assertEquals(3.0, d.mean, 1e-15)
        assertEquals(2.1, d.variance, 1e-15)
        // scipy: stats.binom(10, 0.3).stats(moments='sk')
        assertEquals(0.276026223736942, d.skewness, 1e-12)
        assertEquals(-0.123809523809524, d.kurtosis, 1e-12)
    }

    @Test
    fun testMomentsSymmetric() {
        val d = BinomialDistribution(20, 0.5)
        assertEquals(0.0, d.skewness, 1e-12)
        assertEquals(-0.1, d.kurtosis, 1e-12)
    }

    // --- Entropy ---

    @Test
    fun testEntropy() {
        // scipy: stats.binom(10, 0.3).entropy()
        assertEquals(1.77907878409006, BinomialDistribution(10, 0.3).entropy, 1e-8)
        // scipy: stats.binom(20, 0.5).entropy()
        assertEquals(2.22342391581026, BinomialDistribution(20, 0.5).entropy, 1e-8)
    }

    @Test
    fun testEntropyDegenerate() {
        assertEquals(0.0, BinomialDistribution(0, 0.5).entropy, 1e-15)
    }

    // --- Edge cases ---

    @Test
    fun testDegenerateN0() {
        val d = BinomialDistribution(0, 0.5)
        assertEquals(1.0, d.pmf(0), 1e-15)
        assertEquals(0.0, d.pmf(1), 1e-15)
        assertEquals(0.0, d.mean, 1e-15)
    }

    @Test
    fun testDegenerateP0() {
        val d = BinomialDistribution(10, 0.0)
        assertEquals(1.0, d.pmf(0), 1e-15)
        assertEquals(0.0, d.pmf(1), 1e-15)
    }

    @Test
    fun testDegenerateP1() {
        val d = BinomialDistribution(10, 1.0)
        assertEquals(0.0, d.pmf(0), 1e-15)
        assertEquals(1.0, d.pmf(10), 1e-15)
    }

    @Test
    fun testSkewnessKurtosisDegenerate() {
        assertTrue(BinomialDistribution(0, 0.5).skewness.isNaN())
        assertTrue(BinomialDistribution(10, 0.0).skewness.isNaN())
        assertTrue(BinomialDistribution(10, 1.0).skewness.isNaN())
        assertTrue(BinomialDistribution(0, 0.5).kurtosis.isNaN())
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { BinomialDistribution(-1, 0.5) }
        assertFailsWith<InvalidParameterException> { BinomialDistribution(10, -0.1) }
        assertFailsWith<InvalidParameterException> { BinomialDistribution(10, 1.1) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = BinomialDistribution(10, 0.3)
        assertFailsWith<InvalidParameterException> { d.quantileInt(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantileInt(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val d = BinomialDistribution(10, 0.3)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            val k = d.quantileInt(p)
            assertTrue(d.cdf(k) >= p, "cdf(quantileInt($p)) >= $p")
            if (k > 0) assertTrue(d.cdf(k - 1) < p, "cdf(quantileInt($p)-1) < $p")
        }
    }

    @Test
    fun testExpLogPmfConsistency() {
        val d = BinomialDistribution(10, 0.3)
        for (k in 0..10) {
            assertEquals(d.pmf(k), exp(d.logPmf(k)), 1e-12, "exp(logPmf($k)) ≈ pmf($k)")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val d = BinomialDistribution(10, 0.3)
        for (k in -1..11) {
            assertEquals(1.0, d.sf(k) + d.cdf(k), 1e-12, "sf($k) + cdf($k) ≈ 1")
        }
    }

    @Test
    fun testSampleStats() {
        val d = BinomialDistribution(10, 0.3)
        val rng = kotlin.random.Random(42)
        val samples = d.sample(100_000, rng)
        val doubles = samples.map { it.toDouble() }
        val sampleMean = doubles.average()
        assertEquals(3.0, sampleMean, 0.15, "sample mean ≈ 3.0")
        val sampleVar = doubles.sumOf { (it - sampleMean) * (it - sampleMean) } / (doubles.size - 1)
        assertEquals(d.variance, sampleVar, maxOf(d.variance * 0.1, 0.05), "sample variance ≈ ${d.variance}")
    }

    @Test
    fun testCdfMonotonicity() {
        val d = BinomialDistribution(10, 0.3)
        var prev = 0.0
        for (k in 0..10) {
            val cdfVal = d.cdf(k)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testExtremeParameters() {
        // n=1000, p=0.5: large symmetric
        val d1 = BinomialDistribution(1000, 0.5)
        // scipy: cdf(500) = 0.512613
        assertEquals(0.512612509089181, d1.cdf(500), 1e-3)
        // scipy: sf(530) = 0.026839
        assertEquals(0.026838924822505, d1.sf(530), 1e-3)

        // n=10000, p=0.001: Poisson-like
        val d2 = BinomialDistribution(10000, 0.001)
        assertEquals(10.0, d2.mean, 1e-10)
        // scipy: cdf(10) = 0.583040
        assertEquals(0.583039760629257, d2.cdf(10), 1e-3)
        // scipy: pmf(10) = 0.125173
        assertEquals(0.125172636650239, d2.pmf(10), 1e-3)
    }

    @Test
    fun testPmfSumsToOne() {
        val d = BinomialDistribution(10, 0.3)
        val total = (0..10).sumOf { d.pmf(it) }
        assertEquals(1.0, total, 1e-10)
    }
}
