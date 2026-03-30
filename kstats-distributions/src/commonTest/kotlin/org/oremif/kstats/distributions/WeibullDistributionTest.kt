package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class WeibullDistributionTest : ContinuousDistributionPropertyTests() {
    override fun createDistribution() = WeibullDistribution(1.5, 2.0)
    override val testPoints = listOf(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
    override val roundTripTol = 1e-12

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val w = WeibullDistribution(1.5, 2.0)
        // scipy: stats.weibull_min(1.5, scale=2).pdf(x)
        assertEquals(0.330936338469223, w.pdf(0.5), 1e-12)
        assertEquals(0.372391688219422, w.pdf(1.0), 1e-12)
        assertEquals(0.275909580878582, w.pdf(2.0), 1e-12)
        assertEquals(0.146304264044542, w.pdf(3.0), 1e-12)
        assertEquals(0.0227683519028661, w.pdf(5.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val w = WeibullDistribution(1.5, 2.0)
        // scipy: stats.weibull_min(1.5, scale=2).cdf(x)
        assertEquals(0.117503097415405, w.cdf(0.5), 1e-12)
        assertEquals(0.29781149867344, w.cdf(1.0), 1e-12)
        assertEquals(0.632120558828558, w.cdf(2.0), 1e-12)
        assertEquals(0.840724091509979, w.cdf(3.0), 1e-12)
        assertEquals(0.98080003984499, w.cdf(5.0), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val w = WeibullDistribution(1.5, 2.0)
        // scipy: stats.weibull_min(1.5, scale=2).sf(x)
        assertEquals(0.882496902584595, w.sf(0.5), 1e-12)
        assertEquals(0.70218850132656, w.sf(1.0), 1e-12)
        assertEquals(0.367879441171442, w.sf(2.0), 1e-12)
        assertEquals(0.159275908490021, w.sf(3.0), 1e-12)
        assertEquals(0.0191999601550095, w.sf(5.0), 1e-12)
    }

    @Test
    fun testLogPdfKnownValues() {
        val w = WeibullDistribution(1.5, 2.0)
        assertEquals(-1.10582925301173, w.logPdf(0.5), 1e-10)
        assertEquals(-0.987809053325027, w.logPdf(1.0), 1e-10)
        assertEquals(-1.28768207245178, w.logPdf(2.0), 1e-10)
        assertEquals(-1.92206682548508, w.logPdf(3.0), 1e-10)
        assertEquals(-3.78238378172518, w.logPdf(5.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val w = WeibullDistribution(1.5, 2.0)
        // scipy: stats.weibull_min(1.5, scale=2).ppf(p)
        assertEquals(0.0931430336940394, w.quantile(0.01), 1e-10)
        assertEquals(0.446151051273834, w.quantile(0.1), 1e-10)
        assertEquals(0.871575863406048, w.quantile(0.25), 1e-10)
        assertEquals(1.5664395375493, w.quantile(0.5), 1e-10)
        assertEquals(2.48656776975034, w.quantile(0.75), 1e-10)
        assertEquals(3.48744302719282, w.quantile(0.9), 1e-10)
        assertEquals(5.53597073004505, w.quantile(0.99), 1e-10)
    }

    @Test
    fun testMoments() {
        val w = WeibullDistribution(1.5, 2.0)
        assertEquals(1.80549058590187, w.mean, 1e-10)
        assertEquals(1.50276113925573, w.variance, 1e-10)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.weibull_min(1.5, scale=2).entropy()
        assertEquals(1.48008729408563, WeibullDistribution(1.5, 2.0).entropy, 1e-10)
        // scipy: stats.weibull_min(0.5, scale=1).entropy()
        assertEquals(1.11593151565841, WeibullDistribution(0.5, 1.0).entropy, 1e-10)
        // scipy: stats.weibull_min(3, scale=5).entropy()
        assertEquals(1.89563606703368, WeibullDistribution(3.0, 5.0).entropy, 1e-10)
        // Weibull(1,1) = Exponential(1), entropy = 1
        assertEquals(1.0, WeibullDistribution(1.0, 1.0).entropy, 1e-10)
    }

    // --- Edge cases ---

    @Test
    fun testPdfAtZero() {
        // k=1: pdf(0) = 1/lambda
        assertEquals(0.5, WeibullDistribution(1.0, 2.0).pdf(0.0), 1e-12)
        // k<1: pdf(0) = +Inf
        assertEquals(Double.POSITIVE_INFINITY, WeibullDistribution(0.5, 1.0).pdf(0.0))
        // k>1: pdf(0) = 0
        assertEquals(0.0, WeibullDistribution(2.0, 1.0).pdf(0.0), 1e-12)
    }

    @Test
    fun testNegativeX() {
        val w = WeibullDistribution(1.5, 2.0)
        assertEquals(0.0, w.pdf(-1.0), 1e-12)
        assertEquals(0.0, w.cdf(-1.0), 1e-12)
        assertEquals(1.0, w.sf(-1.0), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, w.logPdf(-1.0))
    }

    @Test
    fun testLogPdfAtZero() {
        assertEquals(-ln(2.0), WeibullDistribution(1.0, 2.0).logPdf(0.0), 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, WeibullDistribution(0.5, 1.0).logPdf(0.0))
        assertEquals(Double.NEGATIVE_INFINITY, WeibullDistribution(2.0, 1.0).logPdf(0.0))
    }

    @Test
    fun testQuantileBoundaries() {
        val w = WeibullDistribution(1.5, 2.0)
        assertEquals(0.0, w.quantile(0.0), 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, w.quantile(1.0))
    }

    // --- Extreme params ---

    @Test
    fun testSmallShape() {
        val w = WeibullDistribution(0.5, 1.0)
        assertEquals(2.0, w.mean, 1e-10)
        assertEquals(20.0, w.variance, 1e-8)
        // scipy: stats.weibull_min(0.5, scale=1).sf(1.0)
        assertEquals(0.367879441171442, w.sf(1.0), 1e-12)
    }

    @Test
    fun testLargeShape() {
        val w = WeibullDistribution(3.0, 5.0)
        assertEquals(4.46489755784624, w.mean, 1e-10)
        // scipy: stats.weibull_min(3, scale=5).sf(x)
        assertEquals(0.992031914837061, w.sf(1.0), 1e-10)
        assertEquals(0.80573530187348, w.sf(3.0), 1e-10)
        assertEquals(0.367879441171442, w.sf(5.0), 1e-12)
    }

    @Test
    fun testExponentialEquivalence() {
        // Weibull(1,1) = Exponential(1)
        val w = WeibullDistribution(1.0, 1.0)
        assertEquals(1.0, w.mean, 1e-12)
        assertEquals(1.0, w.variance, 1e-12)
        assertEquals(1.0, w.entropy, 1e-12)
    }

    @Test
    fun testQuantilePrecisionNearOne() {
        // Same as ExponentialDistribution test: Weibull(1,1) = Exp(1)
        val w = WeibullDistribution(1.0, 1.0)
        // scipy: stats.weibull_min(1, scale=1).ppf(1 - 1e-15)
        assertEquals(34.5395759923409, w.quantile(1.0 - 1e-15), 1e-6)
        assertEquals(23.0258508472001, w.quantile(1.0 - 1e-10), 1e-6)
    }

    // --- NaN/inf handling ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { WeibullDistribution(0.0, 1.0) }
        assertFailsWith<InvalidParameterException> { WeibullDistribution(-1.0, 1.0) }
        assertFailsWith<InvalidParameterException> { WeibullDistribution(1.0, 0.0) }
        assertFailsWith<InvalidParameterException> { WeibullDistribution(1.0, -1.0) }
    }

    @Test
    fun testSpecialDoubleInputs() {
        val w = WeibullDistribution(2.0, 1.0) // k > 1
        // pdf
        assertEquals(0.0, w.pdf(Double.POSITIVE_INFINITY))
        assertEquals(0.0, w.pdf(Double.NEGATIVE_INFINITY))
        assertTrue(w.pdf(Double.NaN).isNaN())
        // logPdf
        assertEquals(Double.NEGATIVE_INFINITY, w.logPdf(Double.POSITIVE_INFINITY))
        assertEquals(Double.NEGATIVE_INFINITY, w.logPdf(Double.NEGATIVE_INFINITY))
        assertTrue(w.logPdf(Double.NaN).isNaN())
        // cdf
        assertEquals(1.0, w.cdf(Double.POSITIVE_INFINITY))
        assertEquals(0.0, w.cdf(Double.NEGATIVE_INFINITY))
        assertTrue(w.cdf(Double.NaN).isNaN())
        // sf
        assertEquals(0.0, w.sf(Double.POSITIVE_INFINITY))
        assertEquals(1.0, w.sf(Double.NEGATIVE_INFINITY))
        assertTrue(w.sf(Double.NaN).isNaN())
    }

    @Test
    fun testSpecialDoubleInputsSmallShape() {
        val w = WeibullDistribution(0.5, 1.0) // k < 1
        assertEquals(0.0, w.pdf(Double.POSITIVE_INFINITY))
        assertEquals(Double.NEGATIVE_INFINITY, w.logPdf(Double.POSITIVE_INFINITY))
    }

    @Test
    fun testQuantileInvalidP() {
        val w = WeibullDistribution(1.5, 2.0)
        assertFailsWith<InvalidParameterException> { w.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { w.quantile(1.1) }
    }

}
