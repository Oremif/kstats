package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class NakagamiDistributionTest : ContinuousDistributionPropertyTests() {

    override fun createDistribution(): ContinuousDistribution = NakagamiDistribution(2.0, 1.0)
    override val testPoints = listOf(0.5, 1.0, 1.5, 2.0, 3.0)
    override val pValues = listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9)
    override val roundTripTol = 1e-8

    private val tol = 1e-10
    private val pdfTol = 1e-12
    private val momentTol = 1e-10
    private val statTol = 0.05

    // ========================================
    // Config 1: mu=1, omega=1 (Rayleigh-like)
    // scipy: stats.nakagami(1, scale=1)
    // ========================================

    private val d1 = NakagamiDistribution(1.0, 1.0)

    @Test
    fun testPdfConfig1() {
        // scipy: stats.nakagami(1, scale=1).pdf(x)
        assertEquals(0.778800783071405, d1.pdf(0.5), pdfTol)
        assertEquals(0.735758882342885, d1.pdf(1.0), pdfTol)
        assertEquals(0.316197673685593, d1.pdf(1.5), pdfTol)
        assertEquals(0.0732625555549367, d1.pdf(2.0), pdfTol)
        assertEquals(0.000740458824520077, d1.pdf(3.0), pdfTol)
    }

    @Test
    fun testLogPdfConfig1() {
        // scipy: stats.nakagami(1, scale=1).logpdf(x)
        assertEquals(-0.25, d1.logPdf(0.5), pdfTol)
        assertEquals(-0.306852819440055, d1.logPdf(1.0), pdfTol)
        assertEquals(-1.15138771133189, d1.logPdf(1.5), pdfTol)
        assertEquals(-2.61370563888011, d1.logPdf(2.0), pdfTol)
        assertEquals(-7.20824053077195, d1.logPdf(3.0), pdfTol)
    }

    @Test
    fun testCdfConfig1() {
        // scipy: stats.nakagami(1, scale=1).cdf(x)
        assertEquals(0.221199216928595, d1.cdf(0.5), tol)
        assertEquals(0.632120558828558, d1.cdf(1.0), tol)
        assertEquals(0.894600775438136, d1.cdf(1.5), tol)
        assertEquals(0.981684361111266, d1.cdf(2.0), tol)
        assertEquals(0.999876590195913, d1.cdf(3.0), tol)
    }

    @Test
    fun testSfConfig1() {
        // scipy: stats.nakagami(1, scale=1).sf(x)
        assertEquals(0.778800783071405, d1.sf(0.5), tol)
        assertEquals(0.367879441171442, d1.sf(1.0), tol)
        assertEquals(0.105399224561864, d1.sf(1.5), tol)
        assertEquals(0.0183156388887342, d1.sf(2.0), tol)
        assertEquals(0.00012340980408668, d1.sf(3.0), tol)
    }

    @Test
    fun testQuantileConfig1() {
        // scipy: stats.nakagami(1, scale=1).ppf(p)
        assertEquals(0.100251363349839, d1.quantile(0.01), tol)
        assertEquals(0.324592845974501, d1.quantile(0.1), tol)
        assertEquals(0.536360021302652, d1.quantile(0.25), tol)
        assertEquals(0.832554611157698, d1.quantile(0.5), tol)
        assertEquals(1.17741002251547, d1.quantile(0.75), tol)
        assertEquals(1.51742712938515, d1.quantile(0.9), tol)
        assertEquals(2.14596602628935, d1.quantile(0.99), tol)
    }

    @Test
    fun testMomentsConfig1() {
        assertEquals(0.886226925452758, d1.mean, momentTol)
        assertEquals(0.214601836602552, d1.variance, momentTol)
        assertEquals(0.631110657818934, d1.skewness, momentTol)
        assertEquals(0.245089300687629, d1.kurtosis, momentTol)
    }

    // ========================================
    // Config 2: mu=0.5, omega=1 (half-normal)
    // scipy: stats.nakagami(0.5, scale=1)
    // ========================================

    private val d2 = NakagamiDistribution(0.5, 1.0)

    @Test
    fun testPdfConfig2() {
        // scipy: stats.nakagami(0.5, scale=1).pdf(x)
        assertEquals(0.793905094954024, d2.pdf(0.1), pdfTol)
        assertEquals(0.704130653528599, d2.pdf(0.5), pdfTol)
        assertEquals(0.483941449038287, d2.pdf(1.0), pdfTol)
        assertEquals(0.107981933026376, d2.pdf(2.0), pdfTol)
        assertEquals(0.00886369682387601, d2.pdf(3.0), pdfTol)
    }

    @Test
    fun testLogPdfConfig2() {
        // scipy: stats.nakagami(0.5, scale=1).logpdf(x)
        assertEquals(-0.230791352644727, d2.logPdf(0.1), pdfTol)
        assertEquals(-0.350791352644727, d2.logPdf(0.5), pdfTol)
        assertEquals(-0.725791352644727, d2.logPdf(1.0), pdfTol)
        assertEquals(-2.22579135264473, d2.logPdf(2.0), pdfTol)
        assertEquals(-4.72579135264473, d2.logPdf(3.0), pdfTol)
    }

    @Test
    fun testCdfConfig2() {
        // scipy: stats.nakagami(0.5, scale=1).cdf(x)
        assertEquals(0.079655674554058, d2.cdf(0.1), tol)
        assertEquals(0.382924922548026, d2.cdf(0.5), tol)
        assertEquals(0.682689492137086, d2.cdf(1.0), tol)
        assertEquals(0.954499736103642, d2.cdf(2.0), tol)
        assertEquals(0.99730020393674, d2.cdf(3.0), tol)
    }

    @Test
    fun testSfConfig2() {
        // scipy: stats.nakagami(0.5, scale=1).sf(x)
        assertEquals(0.920344325445942, d2.sf(0.1), tol)
        assertEquals(0.617075077451974, d2.sf(0.5), tol)
        assertEquals(0.317310507862911, d2.sf(1.0), tol)
        assertEquals(0.0455002638963584, d2.sf(2.0), tol)
        assertEquals(0.00269979606326019, d2.sf(3.0), tol)
    }

    @Test
    fun testQuantileConfig2() {
        // scipy: stats.nakagami(0.5, scale=1).ppf(p)
        assertEquals(0.0125334695080693, d2.quantile(0.01), tol)
        assertEquals(0.125661346855074, d2.quantile(0.1), tol)
        assertEquals(0.318639363964375, d2.quantile(0.25), tol)
        assertEquals(0.674489750196081, d2.quantile(0.5), tol)
        assertEquals(1.15034938037601, d2.quantile(0.75), tol)
        assertEquals(1.64485362695147, d2.quantile(0.9), tol)
        assertEquals(2.5758293035489, d2.quantile(0.99), tol)
    }

    @Test
    fun testMomentsConfig2() {
        assertEquals(0.797884560802865, d2.mean, momentTol)
        assertEquals(0.363380227632419, d2.variance, momentTol)
        assertEquals(0.995271746431157, d2.skewness, momentTol)
        assertEquals(0.869177303605968, d2.kurtosis, momentTol)
    }

    // ========================================
    // Config 3: mu=2, omega=3
    // scipy: stats.nakagami(2, scale=sqrt(3))
    // ========================================

    private val d3 = NakagamiDistribution(2.0, 3.0)

    @Test
    fun testPdfConfig3() {
        // scipy: stats.nakagami(2, scale=sqrt(3)).pdf(x)
        assertEquals(0.094053524987846, d3.pdf(0.5), pdfTol)
        assertEquals(0.456370772473415, d3.pdf(1.0), pdfTol)
        assertEquals(0.669390480445289, d3.pdf(1.5), pdfTol)
        assertEquals(0.494104542028811, d3.pdf(2.0), pdfTol)
        assertEquals(0.0594900522399925, d3.pdf(3.0), pdfTol)
    }

    @Test
    fun testLogPdfConfig3() {
        // scipy: stats.nakagami(2, scale=sqrt(3)).logpdf(x)
        assertEquals(-2.36389124400289, d3.logPdf(0.5), pdfTol)
        assertEquals(-0.78444970232305, d3.logPdf(1.0), pdfTol)
        assertEquals(-0.40138771133189, d3.logPdf(1.5), pdfTol)
        assertEquals(-0.705008160643215, d3.logPdf(2.0), pdfTol)
        assertEquals(-2.82194616965206, d3.logPdf(3.0), pdfTol)
    }

    @Test
    fun testCdfConfig3() {
        // scipy: stats.nakagami(2, scale=sqrt(3)).cdf(x)
        assertEquals(0.0124379876276169, d3.cdf(0.5), tol)
        assertEquals(0.144304801612347, d3.cdf(1.0), tol)
        assertEquals(0.442174599628925, d3.cdf(1.5), tol)
        assertEquals(0.745227345516394, d3.cdf(2.0), tol)
        assertEquals(0.982648734763335, d3.cdf(3.0), tol)
    }

    @Test
    fun testSfConfig3() {
        // scipy: stats.nakagami(2, scale=sqrt(3)).sf(x)
        assertEquals(0.987562012372383, d3.sf(0.5), tol)
        assertEquals(0.855695198387653, d3.sf(1.0), tol)
        assertEquals(0.557825400371075, d3.sf(1.5), tol)
        assertEquals(0.254772654483605, d3.sf(2.0), tol)
        assertEquals(0.0173512652366645, d3.sf(3.0), tol)
    }

    @Test
    fun testQuantileConfig3() {
        // scipy: stats.nakagami(2, scale=sqrt(3)).ppf(p)
        assertEquals(0.472050961634333, d3.quantile(0.01), tol)
        assertEquals(0.893150274357243, d3.quantile(0.1), tol)
        assertEquals(1.2007989609723, d3.quantile(0.25), tol)
        assertEquals(1.58666962062838, d3.quantile(0.5), tol)
        assertEquals(2.00971435615476, d3.quantile(0.75), tol)
        assertEquals(2.41548758117303, d3.quantile(0.9), tol)
        assertEquals(3.15555511788191, d3.quantile(0.99), tol)
    }

    @Test
    fun testMomentsConfig3() {
        assertEquals(1.6281028227561, d3.mean, momentTol)
        assertEquals(0.349281198533612, d3.variance, momentTol)
        assertEquals(0.40569507726267, d3.skewness, momentTol)
        assertEquals(0.0592950893996258, d3.kurtosis, momentTol)
    }

    // ========================================
    // Config 4: mu=5, omega=2
    // scipy: stats.nakagami(5, scale=sqrt(2))
    // ========================================

    private val d4 = NakagamiDistribution(5.0, 2.0)

    @Test
    fun testPdfConfig4() {
        // scipy: stats.nakagami(5, scale=sqrt(2)).pdf(x)
        assertEquals(0.00850775128235801, d4.pdf(0.5), pdfTol)
        assertEquals(0.668009428905426, d4.pdf(1.0), pdfTol)
        assertEquals(1.14733403505079, d4.pdf(1.2), pdfTol)
        assertEquals(1.25206869473722, d4.pdf(1.4), pdfTol)
        assertEquals(0.489988672104804, d4.pdf(1.8), pdfTol)
    }

    @Test
    fun testLogPdfConfig4() {
        // scipy: stats.nakagami(5, scale=sqrt(2)).logpdf(x)
        assertEquals(-4.76677761545673, d4.logPdf(0.5), pdfTol)
        assertEquals(-0.403452990417225, d4.logPdf(1.0), pdfTol)
        assertEquals(0.137441020728367, d4.logPdf(1.2), pdfTol)
        assertEquals(0.224797139173691, d4.logPdf(1.4), pdfTol)
        assertEquals(-0.713373006298154, d4.logPdf(1.8), pdfTol)
    }

    @Test
    fun testCdfConfig4() {
        // scipy: stats.nakagami(5, scale=sqrt(2)).cdf(x)
        assertEquals(0.000473987103245852, d4.cdf(0.5), tol)
        assertEquals(0.108821981085849, d4.cdf(1.0), tol)
        assertEquals(0.293561550358719, d4.cdf(1.2), tol)
        assertEquals(0.541788131772204, d4.cdf(1.4), tol)
        assertEquals(0.905951482782754, d4.cdf(1.8), tol)
    }

    @Test
    fun testSfConfig4() {
        // scipy: stats.nakagami(5, scale=sqrt(2)).sf(x)
        assertEquals(0.999526012896754, d4.sf(0.5), tol)
        assertEquals(0.891178018914151, d4.sf(1.0), tol)
        assertEquals(0.706438449641281, d4.sf(1.2), tol)
        assertEquals(0.458211868227796, d4.sf(1.4), tol)
        assertEquals(0.0940485172172462, d4.sf(1.8), tol)
    }

    @Test
    fun testQuantileConfig4() {
        // scipy: stats.nakagami(5, scale=sqrt(2)).ppf(p)
        assertEquals(0.715291850951373, d4.quantile(0.01), tol)
        assertEquals(0.986426079534126, d4.quantile(0.1), tol)
        assertEquals(1.16079289900952, d4.quantile(0.25), tol)
        assertEquals(1.36688095791784, d4.quantile(0.5), tol)
        assertEquals(1.58422608215427, d4.quantile(0.75), tol)
        assertEquals(1.78813753230031, d4.quantile(0.9), tol)
        assertEquals(2.15449535432102, d4.quantile(0.99), tol)
    }

    @Test
    fun testMomentsConfig4() {
        assertEquals(1.37935330716043, d4.mean, momentTol)
        assertEquals(0.0973844540255902, d4.variance, momentTol)
        assertEquals(0.237428814290145, d4.skewness, momentTol)
        assertEquals(0.00852028621157685, d4.kurtosis, momentTol)
    }

    // ========================================
    // Edge cases
    // ========================================

    @Test
    fun testPdfAtZero() {
        // mu > 0.5: pdf(0) = 0
        assertEquals(0.0, d1.pdf(0.0), 0.0)
        assertEquals(0.0, d3.pdf(0.0), 0.0)
        // mu = 0.5: pdf(0) = sqrt(2 / (pi * omega))
        assertEquals(sqrt(2.0 / kotlin.math.PI), d2.pdf(0.0), pdfTol)
    }

    @Test
    fun testPdfNegative() {
        assertEquals(0.0, d1.pdf(-1.0), 0.0)
        assertEquals(0.0, d2.pdf(-0.5), 0.0)
    }

    @Test
    fun testLogPdfAtBoundary() {
        assertEquals(Double.NEGATIVE_INFINITY, d1.logPdf(0.0))
        assertEquals(Double.NEGATIVE_INFINITY, d1.logPdf(-1.0))
    }

    @Test
    fun testCdfAtBoundary() {
        assertEquals(0.0, d1.cdf(0.0), 0.0)
        assertEquals(0.0, d1.cdf(-1.0), 0.0)
    }

    @Test
    fun testSfAtBoundary() {
        assertEquals(1.0, d1.sf(0.0), 0.0)
        assertEquals(1.0, d1.sf(-1.0), 0.0)
    }

    @Test
    fun testQuantileAtBoundaries() {
        assertEquals(0.0, d1.quantile(0.0), 0.0)
        assertEquals(Double.POSITIVE_INFINITY, d1.quantile(1.0))
    }

    @Test
    fun testBoundaryMu() {
        // mu = 0.5 is the minimum allowed value
        val d = NakagamiDistribution(0.5, 1.0)
        assertTrue(d.pdf(0.5) > 0.0)
        assertTrue(d.cdf(0.5) > 0.0)
    }

    // ========================================
    // Extreme parameters
    // ========================================

    @Test
    fun testLargeMu() {
        // mu=100, omega=1 — very concentrated around 1
        val d = NakagamiDistribution(100.0, 1.0)
        // scipy: stats.nakagami(100, scale=1)
        assertEquals(7.97219936182963, d.pdf(1.0), 1e-6)
        assertEquals(0.513298798279149, d.cdf(1.0), 1e-6)
        assertEquals(0.998750786126202, d.mean, 1e-6)
        assertEquals(0.0024968672122927, d.variance, 1e-6)
    }

    @Test
    fun testSmallOmega() {
        // mu=0.5, omega=0.01
        val d = NakagamiDistribution(0.5, 0.01)
        // scipy: stats.nakagami(0.5, scale=sqrt(0.01))
        assertEquals(7.93905094954024, d.pdf(0.01), 1e-8)
        assertEquals(7.04130653528599, d.pdf(0.05), 1e-8)
        assertEquals(0.079655674554058, d.cdf(0.01), tol)
        assertEquals(0.682689492137086, d.cdf(0.1), tol)
        assertEquals(0.0797884560802865, d.mean, momentTol)
        assertEquals(0.00363380227632419, d.variance, momentTol)
    }

    @Test
    fun testPdfFarFromMode() {
        val pdfVal = d1.pdf(10.0)
        assertTrue(pdfVal > 0.0, "pdf in far tail should be positive")
        assertTrue(pdfVal < 1e-30, "pdf in far tail should be tiny")
    }

    // ========================================
    // Non-finite
    // ========================================

    @Test
    fun testPdfNaN() {
        assertTrue(d1.pdf(Double.NaN).isNaN())
    }

    @Test
    fun testLogPdfNaN() {
        assertTrue(d1.logPdf(Double.NaN).isNaN())
    }

    @Test
    fun testCdfNaN() {
        assertTrue(d1.cdf(Double.NaN).isNaN())
    }

    @Test
    fun testSfNaN() {
        assertTrue(d1.sf(Double.NaN).isNaN())
    }

    @Test
    fun testCdfInfinity() {
        assertEquals(1.0, d1.cdf(Double.POSITIVE_INFINITY), 0.0)
        assertEquals(0.0, d1.cdf(Double.NEGATIVE_INFINITY), 0.0)
    }

    @Test
    fun testSfInfinity() {
        assertEquals(0.0, d1.sf(Double.POSITIVE_INFINITY), 0.0)
        assertEquals(1.0, d1.sf(Double.NEGATIVE_INFINITY), 0.0)
    }

    // ========================================
    // Property-based
    // ========================================

    @Test
    fun testEntropy() {
        assertEquals(0.875408870846436, NakagamiDistribution(2.0, 3.0).entropy, 1e-10)
        assertEquals(0.725791352644727, NakagamiDistribution(0.5, 1.0).entropy, 1e-10)
        assertEquals(0.595460651890821, NakagamiDistribution(1.0, 1.0).entropy, 1e-10)
        assertEquals(0.249231775907822, NakagamiDistribution(5.0, 2.0).entropy, 1e-10)
        assertEquals(1.746753198387844, NakagamiDistribution(1.0, 10.0).entropy, 1e-10)
        assertEquals(0.370457113107702, NakagamiDistribution(10.0, 5.0).entropy, 1e-10)
    }

    @Test
    fun testSampleMedian() {
        val samples = d1.sample(100_000, Random(42)).sorted()
        val sampleMedian = samples[samples.size / 2]
        val theoreticalMedian = d1.quantile(0.5)
        assertEquals(theoreticalMedian, sampleMedian, statTol * theoreticalMedian)
    }

    // ========================================
    // Validation
    // ========================================

    @Test
    fun testInvalidMuTooSmall() {
        assertFailsWith<InvalidParameterException> { NakagamiDistribution(0.4, 1.0) }
    }

    @Test
    fun testInvalidMuZero() {
        assertFailsWith<InvalidParameterException> { NakagamiDistribution(0.0, 1.0) }
    }

    @Test
    fun testInvalidOmegaZero() {
        assertFailsWith<InvalidParameterException> { NakagamiDistribution(1.0, 0.0) }
    }

    @Test
    fun testInvalidOmegaNegative() {
        assertFailsWith<InvalidParameterException> { NakagamiDistribution(1.0, -1.0) }
    }

    @Test
    fun testInvalidMuNaN() {
        assertFailsWith<InvalidParameterException> { NakagamiDistribution(Double.NaN, 1.0) }
    }

    @Test
    fun testInvalidOmegaNaN() {
        assertFailsWith<InvalidParameterException> { NakagamiDistribution(1.0, Double.NaN) }
    }

    @Test
    fun testInvalidQuantileProbability() {
        assertFailsWith<InvalidParameterException> { d1.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { d1.quantile(1.1) }
    }

}
