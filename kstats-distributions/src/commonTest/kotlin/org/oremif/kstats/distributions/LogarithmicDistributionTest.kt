package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LogarithmicDistributionTest : DiscreteDistributionPropertyTests() {

    override fun createDistribution() = LogarithmicDistribution(0.7)
    override val testKRange = 0..15
    override val supportMin = 1

    // --- PMF / logPMF known values (scipy 15-digit refs) ---

    @Test
    fun testPmfKnownValuesP03() {
        val d = LogarithmicDistribution(0.3)
        assertEquals(8.411019756171385e-01, d.pmf(1), 1e-12)
        assertEquals(1.261652963425708e-01, d.pmf(2), 1e-12)
        assertEquals(2.523305926851415e-02, d.pmf(3), 1e-12)
        assertEquals(1.362585200499764e-03, d.pmf(5), 1e-12)
        assertEquals(1.655541018607213e-06, d.pmf(10), 1e-12)
    }

    @Test
    fun testPmfKnownValuesP05() {
        val d = LogarithmicDistribution(0.5)
        assertEquals(7.213475204444817e-01, d.pmf(1), 1e-12)
        assertEquals(1.803368801111204e-01, d.pmf(2), 1e-12)
        assertEquals(6.011229337037347e-02, d.pmf(3), 1e-12)
        assertEquals(9.016844005556022e-03, d.pmf(5), 1e-12)
        assertEquals(1.408881875868128e-04, d.pmf(10), 1e-12)
    }

    @Test
    fun testPmfKnownValuesP07() {
        val d = LogarithmicDistribution(0.7)
        assertEquals(5.814084815577761e-01, d.pmf(1), 1e-12)
        assertEquals(2.034929685452216e-01, d.pmf(2), 1e-12)
        assertEquals(9.496338532110342e-02, d.pmf(3), 1e-12)
        assertEquals(2.791923528440441e-02, d.pmf(5), 1e-12)
        assertEquals(2.346192937124923e-03, d.pmf(10), 1e-12)
    }

    @Test
    fun testPmfKnownValuesP09() {
        val d = LogarithmicDistribution(0.9)
        assertEquals(3.908650337129266e-01, d.pmf(1), 1e-12)
        assertEquals(1.758892651708170e-01, d.pmf(2), 1e-12)
        assertEquals(1.055335591024902e-01, d.pmf(3), 1e-12)
        assertEquals(5.128930972381023e-02, d.pmf(5), 1e-12)
        assertEquals(1.514291224940635e-02, d.pmf(10), 1e-12)
    }

    @Test
    fun testPmfKnownValuesP099() {
        val d = LogarithmicDistribution(0.99)
        assertEquals(2.149757685421097e-01, d.pmf(1), 1e-12)
        assertEquals(1.064130054283443e-01, d.pmf(2), 1e-12)
        assertEquals(7.023258358270723e-02, d.pmf(3), 1e-12)
        assertEquals(4.130097310164681e-02, d.pmf(5), 1e-12)
        assertEquals(1.963840723542683e-02, d.pmf(10), 1e-12)
    }

    @Test
    fun testLogPmfKnownValues() {
        val d1 = LogarithmicDistribution(0.3)
        assertEquals(-1.730423711672132e-01, d1.logPmf(1), 1e-12)
        assertEquals(-2.070162356053094e+00, d1.logPmf(2), 1e-12)
        assertEquals(-3.679600268487195e+00, d1.logPmf(3), 1e-12)

        val d2 = LogarithmicDistribution(0.5)
        assertEquals(-3.266342599782810e-01, d2.logPmf(1), 1e-12)
        assertEquals(-1.712928621098172e+00, d2.logPmf(2), 1e-12)

        val d3 = LogarithmicDistribution(0.9)
        assertEquals(-9.393929609057822e-01, d3.logPmf(1), 1e-12)
        assertEquals(-4.190222694820265e+00, d3.logPmf(10), 1e-12)
    }

    // --- CDF / SF known values ---

    @Test
    fun testCdfKnownValuesP03() {
        val d = LogarithmicDistribution(0.3)
        assertEquals(8.411019756171385e-01, d.cdf(1), 1e-10)
        assertEquals(9.672672719597093e-01, d.cdf(2), 1e-10)
        assertEquals(9.925003312282235e-01, d.cdf(3), 1e-10)
        assertEquals(9.995403547641389e-01, d.cdf(5), 1e-10)
        assertEquals(9.999993766268324e-01, d.cdf(10), 1e-10)
    }

    @Test
    fun testCdfKnownValuesP05() {
        val d = LogarithmicDistribution(0.5)
        assertEquals(7.213475204444817e-01, d.cdf(1), 1e-10)
        assertEquals(9.016844005556022e-01, d.cdf(2), 1e-10)
        assertEquals(9.617966939259757e-01, d.cdf(3), 1e-10)
        assertEquals(9.933556479454217e-01, d.cdf(5), 1e-10)
    }

    @Test
    fun testCdfKnownValuesP09() {
        val d = LogarithmicDistribution(0.9)
        assertEquals(3.908650337129266e-01, d.cdf(1), 1e-10)
        assertEquals(5.667542988837436e-01, d.cdf(2), 1e-10)
        assertEquals(6.722878579862338e-01, d.cdf(3), 1e-10)
        assertEquals(7.948123201042249e-01, d.cdf(5), 1e-10)
        assertEquals(9.201603889810761e-01, d.cdf(10), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val d1 = LogarithmicDistribution(0.3)
        assertEquals(1.588980243828615e-01, d1.sf(1), 1e-10)
        assertEquals(3.273272804029070e-02, d1.sf(2), 1e-10)

        val d2 = LogarithmicDistribution(0.9)
        assertEquals(6.091349662870733e-01, d2.sf(1), 1e-10)
        assertEquals(2.051876798957751e-01, d2.sf(5), 1e-10)
        assertEquals(7.983961101892390e-02, d2.sf(10), 1e-10)
    }

    // --- Quantile ---

    @Test
    fun testQuantileIntKnownValues() {
        val d1 = LogarithmicDistribution(0.3)
        assertEquals(1, d1.quantileInt(0.25))
        assertEquals(1, d1.quantileInt(0.5))
        assertEquals(1, d1.quantileInt(0.75))

        val d2 = LogarithmicDistribution(0.5)
        assertEquals(1, d2.quantileInt(0.25))
        assertEquals(1, d2.quantileInt(0.5))
        assertEquals(2, d2.quantileInt(0.75))

        val d3 = LogarithmicDistribution(0.9)
        assertEquals(1, d3.quantileInt(0.25))
        assertEquals(2, d3.quantileInt(0.5))
        assertEquals(5, d3.quantileInt(0.75))

        val d4 = LogarithmicDistribution(0.99)
        assertEquals(2, d4.quantileInt(0.25))
        assertEquals(6, d4.quantileInt(0.5))
        assertEquals(22, d4.quantileInt(0.75))
    }

    @Test
    fun testQuantileBoundary() {
        val d = LogarithmicDistribution(0.5)
        assertEquals(1, d.quantileInt(0.0))
        assertEquals(Int.MAX_VALUE, d.quantileInt(1.0))
    }

    @Test
    fun testQuantileInvalidP() {
        val d = LogarithmicDistribution(0.5)
        assertFailsWith<InvalidParameterException> { d.quantileInt(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantileInt(1.1) }
    }

    // --- Moments ---

    @Test
    fun testMomentsP03() {
        val d = LogarithmicDistribution(0.3)
        assertEquals(1.201574250881627e+00, d.mean, 1e-10)
        assertEquals(2.727539637348675e-01, d.variance, 1e-10)
        assertEquals(3.298310082548320e+00, d.skewness, 1e-6)
        assertEquals(1.469893412509325e+01, d.kurtosis, 1e-4)
    }

    @Test
    fun testMomentsP05() {
        val d = LogarithmicDistribution(0.5)
        assertEquals(1.442695040888963e+00, d.mean, 1e-10)
        assertEquals(8.040211007723190e-01, d.variance, 1e-10)
        assertEquals(3.014824431890540e+00, d.skewness, 1e-6)
        assertEquals(1.338842081953825e+01, d.kurtosis, 1e-4)
    }

    @Test
    fun testMomentsP07() {
        val d = LogarithmicDistribution(0.7)
        assertEquals(1.938028271859254e+00, d.mean, 1e-10)
        assertEquals(2.704140657005078e+00, d.variance, 1e-10)
        assertEquals(3.059743776094487e+00, d.skewness, 1e-6)
        assertEquals(1.442379081467195e+01, d.kurtosis, 1e-4)
    }

    @Test
    fun testMomentsP09() {
        val d = LogarithmicDistribution(0.9)
        assertEquals(3.908650337129267e+00, d.mean, 1e-10)
        assertEquals(2.380895591335195e+01, d.variance, 1e-8)
        assertEquals(3.475346035525533e+00, d.skewness, 1e-6)
        assertEquals(1.890556319688110e+01, d.kurtosis, 1e-4)
    }

    @Test
    fun testMomentsP099() {
        val d = LogarithmicDistribution(0.99)
        assertEquals(2.149757685421095e+01, d.mean, 1e-8)
        assertEquals(1.687611874818387e+03, d.variance, 1e-5)
        assertEquals(4.457476534015192e+00, d.skewness, 1e-4)
        assertEquals(3.078865138230687e+01, d.kurtosis, 1e-2)
    }

    // --- Entropy ---

    @Test
    fun testEntropy() {
        // refs: mpmath (30 digits); scipy's _expect truncates for large p
        assertEquals(5.418157739122489e-01, LogarithmicDistribution(0.3).entropy, 1e-8)
        assertEquals(8.829244358028678e-01, LogarithmicDistribution(0.5).entropy, 1e-8)
        assertEquals(1.321323177770790e+00, LogarithmicDistribution(0.7).entropy, 1e-8)
        assertEquals(2.137656983403781e+00, LogarithmicDistribution(0.9).entropy, 1e-8)
        assertEquals(3.661173369939258e+00, LogarithmicDistribution(0.99).entropy, 1e-6)
    }

    // --- Edge cases ---

    @Test
    fun testSmallProbability() {
        // With very small p, almost all mass is at k=1
        val d = LogarithmicDistribution(0.01)
        assertTrue(d.pmf(1) > 0.99)
        assertTrue(d.pmf(2) < 0.01)
        assertEquals(1.0, d.cdf(1), 0.01)
    }

    @Test
    fun testKOutsideSupport() {
        val d = LogarithmicDistribution(0.5)
        assertEquals(0.0, d.pmf(0), 1e-15)
        assertEquals(0.0, d.pmf(-1), 1e-15)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(0))
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(-1))
        assertEquals(0.0, d.cdf(0), 1e-15)
        assertEquals(1.0, d.sf(0), 1e-15)
    }

    // --- Invalid parameters ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { LogarithmicDistribution(0.0) }
        assertFailsWith<InvalidParameterException> { LogarithmicDistribution(1.0) }
        assertFailsWith<InvalidParameterException> { LogarithmicDistribution(-0.1) }
        assertFailsWith<InvalidParameterException> { LogarithmicDistribution(1.5) }
        assertFailsWith<InvalidParameterException> { LogarithmicDistribution(Double.NaN) }
    }

}
