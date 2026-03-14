package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BetaBinomialDistributionTest {

    // ==================== Basic correctness (scipy 15-digit refs) ====================

    // --- Parameter set 1: BetaBinomial(10, 2.0, 5.0) — asymmetric, moderate ---

    @Test
    fun testPmfSet1() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        // scipy: stats.betabinom(10, 2.0, 5.0).pmf(k)
        assertEquals(1.250000000000000e-01, d.pmf(0), 1e-12)
        assertEquals(1.785714285714286e-01, d.pmf(1), 1e-12)
        assertEquals(1.854395604395606e-01, d.pmf(2), 1e-12)
        assertEquals(1.648351648351647e-01, d.pmf(3), 1e-12)
        assertEquals(1.311188811188811e-01, d.pmf(4), 1e-12)
        assertEquals(9.440559440559447e-02, d.pmf(5), 1e-12)
        assertEquals(6.118881118881118e-02, d.pmf(6), 1e-12)
        assertEquals(3.496503496503497e-02, d.pmf(7), 1e-12)
        assertEquals(1.685814185814187e-02, d.pmf(8), 1e-12)
        assertEquals(6.243756243756246e-03, d.pmf(9), 1e-12)
        assertEquals(1.373626373626374e-03, d.pmf(10), 1e-12)
    }

    @Test
    fun testLogPmfSet1() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        assertEquals(-2.079441541679836e+00, d.logPmf(0), 1e-10)
        assertEquals(-1.722766597741103e+00, d.logPmf(1), 1e-10)
        assertEquals(-1.685026269758256e+00, d.logPmf(2), 1e-10)
        assertEquals(-1.802809305414641e+00, d.logPmf(3), 1e-10)
        assertEquals(-2.031650877843487e+00, d.logPmf(4), 1e-10)
        assertEquals(-6.590301048196686e+00, d.logPmf(10), 1e-10)
    }

    @Test
    fun testCdfSet1() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        assertEquals(1.250000000000000e-01, d.cdf(0), 1e-10)
        assertEquals(3.035714285714286e-01, d.cdf(1), 1e-10)
        assertEquals(4.890109890109892e-01, d.cdf(2), 1e-10)
        assertEquals(6.538461538461540e-01, d.cdf(3), 1e-10)
        assertEquals(8.793706293706296e-01, d.cdf(5), 1e-10)
        assertEquals(9.755244755244757e-01, d.cdf(7), 1e-10)
        assertEquals(1.0, d.cdf(10), 1e-10)
    }

    @Test
    fun testSfSet1() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        assertEquals(8.750000000000000e-01, d.sf(0), 1e-10)
        assertEquals(6.964285714285714e-01, d.sf(1), 1e-10)
        assertEquals(5.109890109890108e-01, d.sf(2), 1e-10)
        assertEquals(3.461538461538460e-01, d.sf(3), 1e-10)
        assertEquals(1.206293706293704e-01, d.sf(5), 1e-10)
        assertEquals(2.447552447552426e-02, d.sf(7), 1e-10)
        assertEquals(0.0, d.sf(10), 1e-10)
    }

    @Test
    fun testQuantileSet1() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        assertEquals(1, d.quantileInt(0.25))
        assertEquals(3, d.quantileInt(0.5))
        assertEquals(4, d.quantileInt(0.75))
    }

    @Test
    fun testMomentsSet1() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        assertEquals(2.857142857142857e+00, d.mean, 1e-10)
        assertEquals(4.336734693877551e+00, d.variance, 1e-10)
        assertEquals(6.173949065130319e-01, d.skewness, 1e-8)
        assertEquals(-1.661176470588237e-01, d.kurtosis, 1e-6)
    }

    @Test
    fun testEntropySet1() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        assertEquals(2.064181804937187e+00, d.entropy, 1e-8)
    }

    // --- Parameter set 2: BetaBinomial(10, 0.5, 0.5) — U-shaped Beta prior ---

    @Test
    fun testPmfSet2() {
        val d = BetaBinomialDistribution(10, 0.5, 0.5)
        assertEquals(1.761970520019532e-01, d.pmf(0), 1e-12)
        assertEquals(9.273529052734376e-02, d.pmf(1), 1e-12)
        assertEquals(7.364273071289062e-02, d.pmf(2), 1e-12)
        assertEquals(6.546020507812504e-02, d.pmf(3), 1e-12)
        assertEquals(6.168365478515626e-02, d.pmf(4), 1e-12)
        assertEquals(6.056213378906251e-02, d.pmf(5), 1e-12)
        assertEquals(6.168365478515626e-02, d.pmf(6), 1e-12)
        assertEquals(6.546020507812504e-02, d.pmf(7), 1e-12)
        assertEquals(7.364273071289062e-02, d.pmf(8), 1e-12)
        assertEquals(9.273529052734376e-02, d.pmf(9), 1e-12)
        assertEquals(1.761970520019532e-01, d.pmf(10), 1e-12)
    }

    @Test
    fun testCdfSfSet2() {
        val d = BetaBinomialDistribution(10, 0.5, 0.5)
        assertEquals(1.761970520019532e-01, d.cdf(0), 1e-10)
        assertEquals(3.425750732421876e-01, d.cdf(2), 1e-10)
        assertEquals(5.302810668945314e-01, d.cdf(5), 1e-10)
        assertEquals(7.310676574707033e-01, d.cdf(8), 1e-10)
        assertEquals(8.238029479980469e-01, d.sf(0), 1e-10)
        assertEquals(6.574249267578125e-01, d.sf(2), 1e-10)
        assertEquals(4.697189331054686e-01, d.sf(5), 1e-10)
        assertEquals(2.689323425292967e-01, d.sf(8), 1e-10)
    }

    @Test
    fun testMomentsSet2() {
        val d = BetaBinomialDistribution(10, 0.5, 0.5)
        assertEquals(5.000000000000000e+00, d.mean, 1e-10)
        assertEquals(1.375000000000000e+01, d.variance, 1e-10)
        assertEquals(0.0, d.skewness, 1e-10)
        assertEquals(-1.509090909090909e+00, d.kurtosis, 1e-6)
    }

    @Test
    fun testEntropySet2() {
        val d = BetaBinomialDistribution(10, 0.5, 0.5)
        assertEquals(2.307478732387405e+00, d.entropy, 1e-8)
    }

    // --- Parameter set 3: BetaBinomial(20, 1.0, 1.0) — uniform prior (discrete uniform) ---

    @Test
    fun testPmfSet3UniformPrior() {
        val d = BetaBinomialDistribution(20, 1.0, 1.0)
        // Uniform prior: pmf(k) = 1/(n+1) for all k
        val expected = 1.0 / 21.0
        for (k in 0..20) {
            assertEquals(expected, d.pmf(k), 1e-12, "pmf($k) should be 1/21")
        }
    }

    @Test
    fun testCdfSet3() {
        val d = BetaBinomialDistribution(20, 1.0, 1.0)
        assertEquals(4.761904761904762e-02, d.cdf(0), 1e-10)
        assertEquals(2.857142857142858e-01, d.cdf(5), 1e-10)
        assertEquals(5.238095238095239e-01, d.cdf(10), 1e-10)
        assertEquals(7.619047619047621e-01, d.cdf(15), 1e-10)
        assertEquals(1.0, d.cdf(20), 1e-10)
    }

    @Test
    fun testMomentsSet3() {
        val d = BetaBinomialDistribution(20, 1.0, 1.0)
        assertEquals(1.000000000000000e+01, d.mean, 1e-10)
        assertEquals(3.666666666666666e+01, d.variance, 1e-10)
        assertEquals(0.0, d.skewness, 1e-10)
        assertEquals(-1.205454545454546e+00, d.kurtosis, 1e-6)
    }

    @Test
    fun testEntropySet3() {
        val d = BetaBinomialDistribution(20, 1.0, 1.0)
        assertEquals(3.044522437723423e+00, d.entropy, 1e-8)
    }

    // ==================== Edge cases ====================

    @Test
    fun testDegenerateN0() {
        val d = BetaBinomialDistribution(0, 2.0, 5.0)
        assertEquals(1.0, d.pmf(0), 1e-15)
        assertEquals(0.0, d.pmf(1), 1e-15)
        assertEquals(0.0, d.logPmf(0), 1e-15)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(1))
        assertEquals(1.0, d.cdf(0), 1e-15)
        assertEquals(0.0, d.sf(0), 1e-15)
        assertEquals(0, d.quantileInt(0.5))
        assertEquals(0.0, d.mean, 1e-15)
        assertEquals(0.0, d.variance, 1e-15)
        assertTrue(d.skewness.isNaN())
        assertTrue(d.kurtosis.isNaN())
        assertEquals(0.0, d.entropy, 1e-15)
    }

    @Test
    fun testBoundaryPmf() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        assertEquals(0.0, d.pmf(-1), 1e-15)
        assertEquals(0.0, d.pmf(11), 1e-15)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(-1))
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(11))
    }

    @Test
    fun testSymmetricAlphaEqualsBeta() {
        // When alpha == beta, distribution is symmetric → skewness = 0
        val d = BetaBinomialDistribution(10, 3.0, 3.0)
        assertEquals(0.0, d.skewness, 1e-10)
        assertEquals(5.0, d.mean, 1e-10)
        // pmf(k) == pmf(n-k) for symmetric
        for (k in 0..5) {
            assertEquals(d.pmf(k), d.pmf(10 - k), 1e-12, "pmf($k) == pmf(${10 - k})")
        }
    }

    @Test
    fun testCdfBoundaries() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        assertEquals(0.0, d.cdf(-1))
        assertEquals(1.0, d.cdf(10))
        assertEquals(1.0, d.cdf(11))
        assertEquals(1.0, d.sf(-1))
        assertEquals(0.0, d.sf(10))
        assertEquals(0.0, d.sf(11))
    }

    // ==================== Extreme parameters ====================

    @Test
    fun testExtremeLargeN() {
        val d = BetaBinomialDistribution(100, 0.5, 0.5)
        // scipy: stats.betabinom(100, 0.5, 0.5)
        assertEquals(5.634847900925642e-02, d.pmf(0), 1e-12)
        assertEquals(2.831581859761627e-02, d.pmf(1), 1e-12)
        assertEquals(6.334446707872718e-03, d.pmf(50), 1e-12)
        assertEquals(2.831581859761627e-02, d.pmf(99), 1e-12)
        assertEquals(5.634847900925642e-02, d.pmf(100), 1e-12)
        assertEquals(5.634847900925642e-02, d.cdf(0), 1e-10)
        assertEquals(5.031672233539372e-01, d.cdf(50), 1e-10)
        assertEquals(5.000000000000000e+01, d.mean, 1e-10)
        assertEquals(1.262500000000000e+03, d.variance, 1e-10)
        assertEquals(4.432451310676071e+00, d.entropy, 1e-8)
    }

    // ==================== Property-based tests ====================

    @Test
    fun testExpLogPmfConsistency() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        for (k in 0..10) {
            assertEquals(d.pmf(k), exp(d.logPmf(k)), 1e-12, "exp(logPmf($k)) ≈ pmf($k)")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        for (k in -1..11) {
            assertEquals(1.0, d.sf(k) + d.cdf(k), 1e-10, "sf($k) + cdf($k) ≈ 1")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOneSet2() {
        val d = BetaBinomialDistribution(10, 0.5, 0.5)
        for (k in -1..11) {
            assertEquals(1.0, d.sf(k) + d.cdf(k), 1e-10, "sf($k) + cdf($k) ≈ 1")
        }
    }

    @Test
    fun testCdfMonotonicity() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        var prev = 0.0
        for (k in 0..10) {
            val cdfVal = d.cdf(k)
            assertTrue(cdfVal >= prev, "cdf should be monotonically non-decreasing at k=$k")
            prev = cdfVal
        }
    }

    @Test
    fun testCdfQuantileRoundTrip() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            val k = d.quantileInt(p)
            assertTrue(d.cdf(k) >= p, "cdf(quantileInt($p)) >= $p")
            if (k > 0) assertTrue(d.cdf(k - 1) < p, "cdf(quantileInt($p)-1) < $p")
        }
    }

    @Test
    fun testPmfSumsToOne() {
        val d1 = BetaBinomialDistribution(10, 2.0, 5.0)
        assertEquals(1.0, (0..10).sumOf { d1.pmf(it) }, 1e-10)

        val d2 = BetaBinomialDistribution(10, 0.5, 0.5)
        assertEquals(1.0, (0..10).sumOf { d2.pmf(it) }, 1e-10)

        val d3 = BetaBinomialDistribution(20, 1.0, 1.0)
        assertEquals(1.0, (0..20).sumOf { d3.pmf(it) }, 1e-10)
    }

    @Test
    fun testPmfNonNegative() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        for (k in -1..11) {
            assertTrue(d.pmf(k) >= 0.0, "pmf($k) should be non-negative")
        }
    }

    @Test
    fun testSampleStats() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        val rng = kotlin.random.Random(42)
        val samples = d.sample(100_000, rng)
        val doubles = samples.map { it.toDouble() }
        val sampleMean = doubles.average()
        assertEquals(d.mean, sampleMean, d.mean * 0.05, "sample mean ≈ theoretical mean")
        val sampleVar = doubles.sumOf { (it - sampleMean) * (it - sampleMean) } / (doubles.size - 1)
        assertEquals(d.variance, sampleVar, d.variance * 0.10, "sample variance ≈ theoretical variance")
    }

    // ==================== Validation tests ====================

    @Test
    fun testInvalidTrials() {
        assertFailsWith<InvalidParameterException> { BetaBinomialDistribution(-1, 2.0, 5.0) }
    }

    @Test
    fun testInvalidAlpha() {
        assertFailsWith<InvalidParameterException> { BetaBinomialDistribution(10, 0.0, 5.0) }
        assertFailsWith<InvalidParameterException> { BetaBinomialDistribution(10, -1.0, 5.0) }
    }

    @Test
    fun testInvalidBeta() {
        assertFailsWith<InvalidParameterException> { BetaBinomialDistribution(10, 2.0, 0.0) }
        assertFailsWith<InvalidParameterException> { BetaBinomialDistribution(10, 2.0, -1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = BetaBinomialDistribution(10, 2.0, 5.0)
        assertFailsWith<InvalidParameterException> { d.quantileInt(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantileInt(1.1) }
    }
}
