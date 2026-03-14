package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.exp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ZipfDistributionTest {

    // --- PMF / logPMF known values (scipy 15-digit refs) ---

    @Test
    fun testPmfKnownValuesN10S1() {
        val d = ZipfDistribution(10, 1.0)
        assertEquals(3.414171521474055e-01, d.pmf(1), 1e-12)
        assertEquals(1.707085760737028e-01, d.pmf(2), 1e-12)
        assertEquals(6.828343042948111e-02, d.pmf(5), 1e-12)
        assertEquals(3.414171521474055e-02, d.pmf(10), 1e-12)
    }

    @Test
    fun testPmfKnownValuesN10S2() {
        val d = ZipfDistribution(10, 2.0)
        assertEquals(6.452579827864142e-01, d.pmf(1), 1e-12)
        assertEquals(1.613144956966036e-01, d.pmf(2), 1e-12)
        assertEquals(2.581031931145657e-02, d.pmf(5), 1e-12)
        assertEquals(6.452579827864143e-03, d.pmf(10), 1e-12)
    }

    @Test
    fun testPmfKnownValuesN100S05() {
        val d = ZipfDistribution(100, 0.5)
        assertEquals(5.379350788889721e-02, d.pmf(1), 1e-12)
        assertEquals(3.803775421205126e-02, d.pmf(2), 1e-12)
        assertEquals(2.405718807754907e-02, d.pmf(5), 1e-12)
        assertEquals(1.701100082591511e-02, d.pmf(10), 1e-12)
    }

    @Test
    fun testPmfKnownValuesN5S3() {
        val d = ZipfDistribution(5, 3.0)
        assertEquals(8.434106589926711e-01, d.pmf(1), 1e-12)
        assertEquals(1.054263323740839e-01, d.pmf(2), 1e-12)
        assertEquals(6.747285271941369e-03, d.pmf(5), 1e-12)
    }

    @Test
    fun testLogPmfKnownValues() {
        val d1 = ZipfDistribution(10, 1.0)
        assertEquals(-1.074650229264583e+00, d1.logPmf(1), 1e-12)
        assertEquals(-1.767797409824529e+00, d1.logPmf(2), 1e-12)
        assertEquals(-2.684088141698683e+00, d1.logPmf(5), 1e-12)
        assertEquals(-3.377235322258629e+00, d1.logPmf(10), 1e-12)

        val d2 = ZipfDistribution(10, 2.0)
        assertEquals(-4.381050688420970e-01, d2.logPmf(1), 1e-12)
        assertEquals(-1.824399429961988e+00, d2.logPmf(2), 1e-12)

        val d3 = ZipfDistribution(5, 3.0)
        assertEquals(-1.703012996268547e-01, d3.logPmf(1), 1e-12)
        assertEquals(-4.998615036929156e+00, d3.logPmf(5), 1e-12)
    }

    // --- CDF / SF known values ---

    @Test
    fun testCdfKnownValuesN10S1() {
        val d = ZipfDistribution(10, 1.0)
        assertEquals(3.414171521474055e-01, d.cdf(1), 1e-10)
        assertEquals(5.121257282211082e-01, d.cdf(2), 1e-10)
        assertEquals(7.795691640699093e-01, d.cdf(5), 1e-10)
        assertEquals(1.0, d.cdf(10), 1e-10)
    }

    @Test
    fun testCdfKnownValuesN10S2() {
        val d = ZipfDistribution(10, 2.0)
        assertEquals(6.452579827864142e-01, d.cdf(1), 1e-10)
        assertEquals(8.065724784830179e-01, d.cdf(2), 1e-10)
        assertEquals(9.444067531393380e-01, d.cdf(5), 1e-10)
        assertEquals(1.0, d.cdf(10), 1e-10)
    }

    @Test
    fun testCdfKnownValuesN100S05() {
        val d = ZipfDistribution(100, 0.5)
        assertEquals(5.379350788889721e-02, d.cdf(1), 1e-10)
        assertEquals(9.183126210094845e-02, d.cdf(2), 1e-10)
        assertEquals(1.738429003832552e-01, d.cdf(5), 1e-10)
        assertEquals(2.700970901057363e-01, d.cdf(10), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val d = ZipfDistribution(10, 1.0)
        assertEquals(6.585828478525945e-01, d.sf(1), 1e-10)
        assertEquals(4.878742717788917e-01, d.sf(2), 1e-10)
        assertEquals(2.204308359300907e-01, d.sf(5), 1e-10)
        assertEquals(0.0, d.sf(10), 1e-10)

        val d2 = ZipfDistribution(10, 2.0)
        assertEquals(3.547420172135856e-01, d2.sf(1), 1e-10)
        assertEquals(5.559324686066193e-02, d2.sf(5), 1e-10)
    }

    // --- Quantile ---

    @Test
    fun testQuantileIntKnownValues() {
        val d1 = ZipfDistribution(10, 1.0)
        assertEquals(1, d1.quantileInt(0.25))
        assertEquals(2, d1.quantileInt(0.5))
        assertEquals(5, d1.quantileInt(0.75))

        val d2 = ZipfDistribution(10, 2.0)
        assertEquals(1, d2.quantileInt(0.25))
        assertEquals(1, d2.quantileInt(0.5))
        assertEquals(2, d2.quantileInt(0.75))
    }

    @Test
    fun testQuantileBoundary() {
        val d = ZipfDistribution(10, 1.0)
        assertEquals(1, d.quantileInt(0.0))
        assertEquals(10, d.quantileInt(1.0))
    }

    @Test
    fun testQuantileInvalidP() {
        val d = ZipfDistribution(10, 1.0)
        assertFailsWith<InvalidParameterException> { d.quantileInt(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantileInt(1.1) }
    }

    // --- Moments ---

    @Test
    fun testMomentsN10S1() {
        val d = ZipfDistribution(10, 1.0)
        assertEquals(3.414171521474055e+00, d.mean, 1e-10)
        assertEquals(7.121376190062837e+00, d.variance, 1e-10)
        assertEquals(9.843847103550410e-01, d.skewness, 1e-8)
        assertEquals(-1.730287890606017e-01, d.kurtosis, 1e-6)
    }

    @Test
    fun testMomentsN10S2() {
        val d = ZipfDistribution(10, 2.0)
        assertEquals(1.889940147201001e+00, d.mean, 1e-10)
        assertEquals(2.880706067862000e+00, d.variance, 1e-10)
        assertEquals(2.537260080803198e+00, d.skewness, 1e-8)
        assertEquals(6.657989030448334e+00, d.kurtosis, 1e-6)
    }

    @Test
    fun testMomentsN100S05() {
        val d = ZipfDistribution(100, 0.5)
        assertEquals(3.612034734209534e+01, d.mean, 1e-8)
        assertEquals(8.740234483611915e+02, d.variance, 1e-6)
        assertEquals(5.635184176085750e-01, d.skewness, 1e-6)
        assertEquals(-9.359838301640995e-01, d.kurtosis, 1e-5)
    }

    @Test
    fun testMomentsN5S3() {
        val d = ZipfDistribution(5, 3.0)
        assertEquals(1.234425211731218e+00, d.mean, 1e-10)
        assertEquals(4.019820680089369e-01, d.variance, 1e-10)
        assertEquals(3.324815847621283e+00, d.skewness, 1e-8)
        assertEquals(1.228451233995319e+01, d.kurtosis, 1e-5)
    }

    // --- Entropy ---

    @Test
    fun testEntropy() {
        assertEquals(1.993805759248671e+00, ZipfDistribution(10, 1.0).entropy, 1e-8)
        assertEquals(1.236292605723536e+00, ZipfDistribution(10, 2.0).entropy, 1e-8)
        assertEquals(4.435711654711786e+00, ZipfDistribution(100, 0.5).entropy, 1e-8)
        assertEquals(5.798676580755550e-01, ZipfDistribution(5, 3.0).entropy, 1e-8)
    }

    // --- Edge cases ---

    @Test
    fun testDegenerateN1() {
        val d = ZipfDistribution(1, 2.0)
        assertEquals(1.0, d.pmf(1), 1e-15)
        assertEquals(0.0, d.pmf(0), 1e-15)
        assertEquals(0.0, d.pmf(2), 1e-15)
        assertEquals(1.0, d.cdf(1), 1e-15)
        assertEquals(1.0, d.mean, 1e-15)
        assertEquals(0.0, d.variance, 1e-15)
        assertTrue(d.skewness.isNaN())
        assertTrue(d.kurtosis.isNaN())
        assertEquals(0.0, d.entropy, 1e-15)
    }

    @Test
    fun testLargeExponent() {
        // With a very large exponent, almost all mass is at k=1
        val d = ZipfDistribution(10, 10.0)
        assertTrue(d.pmf(1) > 0.999)
        assertTrue(d.pmf(2) < 0.001)
    }

    @Test
    fun testKOutsideSupport() {
        val d = ZipfDistribution(10, 1.0)
        assertEquals(0.0, d.pmf(0), 1e-15)
        assertEquals(0.0, d.pmf(11), 1e-15)
        assertEquals(0.0, d.pmf(-1), 1e-15)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(0))
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(11))
        assertEquals(0.0, d.cdf(0), 1e-15)
        assertEquals(1.0, d.cdf(10), 1e-15)
        assertEquals(1.0, d.cdf(100), 1e-15)
    }

    // --- Invalid parameters ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { ZipfDistribution(0, 1.0) }
        assertFailsWith<InvalidParameterException> { ZipfDistribution(-1, 1.0) }
        assertFailsWith<InvalidParameterException> { ZipfDistribution(10, 0.0) }
        assertFailsWith<InvalidParameterException> { ZipfDistribution(10, -1.0) }
    }

    // --- Property-based ---

    @Test
    fun testExpLogPmfConsistency() {
        val d = ZipfDistribution(10, 2.0)
        for (k in 0..11) {
            assertEquals(d.pmf(k), exp(d.logPmf(k)), 1e-12, "exp(logPmf($k)) ≈ pmf($k)")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val d = ZipfDistribution(10, 1.0)
        for (k in 0..11) {
            assertEquals(1.0, d.sf(k) + d.cdf(k), 1e-10, "sf($k) + cdf($k) ≈ 1")
        }
    }

    @Test
    fun testCdfMonotonicity() {
        val d = ZipfDistribution(10, 1.0)
        var prev = 0.0
        for (k in 1..10) {
            val cdfVal = d.cdf(k)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing at k=$k")
            prev = cdfVal
        }
    }

    @Test
    fun testCdfQuantileRoundTrip() {
        val d = ZipfDistribution(10, 1.0)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            val k = d.quantileInt(p)
            assertTrue(d.cdf(k) >= p, "cdf(quantileInt($p)) >= $p")
            if (k > 1) assertTrue(d.cdf(k - 1) < p, "cdf(quantileInt($p)-1) < $p")
        }
    }

    @Test
    fun testPmfSumsToOne() {
        val d = ZipfDistribution(10, 1.0)
        val total = (1..10).sumOf { d.pmf(it) }
        assertEquals(1.0, total, 1e-10)
    }

    @Test
    fun testPmfNonNegative() {
        val d = ZipfDistribution(100, 0.5)
        for (k in 0..101) {
            assertTrue(d.pmf(k) >= 0.0, "pmf($k) >= 0")
        }
    }

    // --- Sampling ---

    @Test
    fun testSampleStats() {
        val d = ZipfDistribution(10, 1.0)
        val rng = kotlin.random.Random(42)
        val samples = d.sample(100_000, rng)
        val doubles = samples.map { it.toDouble() }
        val sampleMean = doubles.average()
        assertEquals(d.mean, sampleMean, d.mean * 0.05, "sample mean ≈ ${d.mean}")
        val sampleVar = doubles.sumOf { (it - sampleMean) * (it - sampleMean) } / (doubles.size - 1)
        assertEquals(d.variance, sampleVar, d.variance * 0.1, "sample variance ≈ ${d.variance}")
    }
}
