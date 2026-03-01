package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DistributionTest {

    private val tol = 1e-15

    private class StubContinuousDistribution(
        override val mean: Double = 0.0,
        override val variance: Double = 1.0,
        override val skewness: Double = 0.0,
        override val kurtosis: Double = 0.0,
        private val cdfValue: Double = 0.5,
    ) : ContinuousDistribution {
        override fun pdf(x: Double): Double = 0.0
        override fun logPdf(x: Double): Double = Double.NEGATIVE_INFINITY
        override fun cdf(x: Double): Double = cdfValue
        override fun quantile(p: Double): Double = p
        override fun sample(random: Random): Double = random.nextDouble()
    }

    @Test
    fun standardDeviationDefaultSqrtVariance() {
        val d = StubContinuousDistribution(variance = 4.0)
        assertEquals(2.0, d.standardDeviation, tol)
    }

    @Test
    fun standardDeviationZeroVariance() {
        val d = StubContinuousDistribution(variance = 0.0)
        assertEquals(0.0, d.standardDeviation, tol)
    }

    @Test
    fun standardDeviationLargeVariance() {
        val d = StubContinuousDistribution(variance = 1e12)
        assertEquals(1e6, d.standardDeviation, tol)
    }

    @Test
    fun sfDefaultOneMinusCdf() {
        val d = StubContinuousDistribution(cdfValue = 0.3)
        assertEquals(0.7, d.sf(0.0), tol)
    }

    @Test
    fun sfBoundaryCdfZero() {
        val d = StubContinuousDistribution(cdfValue = 0.0)
        assertEquals(1.0, d.sf(0.0), tol)
    }

    @Test
    fun sfBoundaryCdfOne() {
        val d = StubContinuousDistribution(cdfValue = 1.0)
        assertEquals(0.0, d.sf(0.0), tol)
    }

    @Test
    fun entropyDefaultIsNaN() {
        val d = StubContinuousDistribution()
        assertEquals(Double.NaN, d.entropy)
    }

    // --- DiscreteDistribution defaults ---

    private class StubDiscreteDistribution(
        override val mean: Double = 0.0,
        override val variance: Double = 1.0,
        private val cdfValue: Double = 0.5,
    ) : DiscreteDistribution {
        override fun pmf(k: Int): Double = 0.0
        override fun logPmf(k: Int): Double = Double.NEGATIVE_INFINITY
        override fun cdf(k: Int): Double = cdfValue
        override fun quantileInt(p: Double): Int = 0
        override fun sample(random: Random): Int = 0
    }

    @Test
    fun discreteStandardDeviationDefault() {
        val d = StubDiscreteDistribution(variance = 9.0)
        assertEquals(3.0, d.standardDeviation, tol)
    }

    @Test
    fun discreteSfIntDefault() {
        val d = StubDiscreteDistribution(cdfValue = 0.3)
        assertEquals(0.7, d.sf(0), tol)
    }

    @Test
    fun discreteSfDoubleDefault() {
        val d = StubDiscreteDistribution(cdfValue = 0.3)
        assertEquals(0.7, d.sf(0.0), tol)
    }

    @Test
    fun discreteCdfDoubleBridgesToInt() {
        val d = StubDiscreteDistribution(cdfValue = 0.8)
        assertEquals(0.8, d.cdf(2.7), tol)
    }

    @Test
    fun discreteQuantileDoubleDelegatesToQuantileInt() {
        val d = StubDiscreteDistribution()
        assertEquals(0.0, d.quantile(0.5), tol)
    }

    @Test
    fun discreteSkewnessDefaultIsNaN() {
        val d = StubDiscreteDistribution()
        assertEquals(Double.NaN, d.skewness)
    }

    @Test
    fun discreteKurtosisDefaultIsNaN() {
        val d = StubDiscreteDistribution()
        assertEquals(Double.NaN, d.kurtosis)
    }

    @Test
    fun discreteEntropyDefaultIsNaN() {
        val d = StubDiscreteDistribution()
        assertEquals(Double.NaN, d.entropy)
    }

    // --- cdf(Double) floors correctly for negative fractional values ---

    @Test
    fun discreteCdfDoubleFloorsNegativeFraction() {
        // cdf(-0.5) should delegate to cdf(-1), not cdf(0)
        val poisson = PoissonDistribution(rate = 5.0)
        assertEquals(0.0, poisson.cdf(-0.5), tol, "cdf(-0.5) should be 0 for Poisson")
        assertEquals(0.0, poisson.cdf(-0.1), tol, "cdf(-0.1) should be 0 for Poisson")
    }

    @Test
    fun discreteCdfDoublePositiveFractionFloors() {
        val binom = BinomialDistribution(10, 0.3)
        // cdf(2.7) should equal cdf(2)
        assertEquals(binom.cdf(2), binom.cdf(2.7), tol)
        // cdf(3.0) should equal cdf(3)
        assertEquals(binom.cdf(3), binom.cdf(3.0), tol)
    }

    // --- logPmf edge cases for degenerate parameters ---

    @Test
    fun binomialLogPmfDegenerateP0() {
        val d = BinomialDistribution(10, 0.0)
        assertEquals(0.0, d.logPmf(0), tol, "logPmf(0) should be 0 when p=0")
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(1), "logPmf(1) should be -Inf when p=0")
    }

    @Test
    fun binomialLogPmfDegenerateP1() {
        val d = BinomialDistribution(10, 1.0)
        assertEquals(0.0, d.logPmf(10), tol, "logPmf(n) should be 0 when p=1")
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(0), "logPmf(0) should be -Inf when p=1")
    }

    @Test
    fun geometricLogPmfDegenerateP1() {
        val d = GeometricDistribution(1.0)
        assertEquals(0.0, d.logPmf(0), tol, "logPmf(0) should be 0 when p=1")
        assertEquals(Double.NEGATIVE_INFINITY, d.logPmf(1), "logPmf(1) should be -Inf when p=1")
    }

    // --- sample() degenerate cases ---

    @Test
    fun binomialSampleDegenerateP0LargeN() {
        val d = BinomialDistribution(100, 0.0)
        assertEquals(0, d.sample(Random(42)), "sample should return 0 when p=0")
    }

    @Test
    fun binomialSampleDegenerateP1LargeN() {
        val d = BinomialDistribution(100, 1.0)
        assertEquals(100, d.sample(Random(42)), "sample should return n when p=1")
    }

    @Test
    fun geometricSampleDegenerateP1() {
        val d = GeometricDistribution(1.0)
        assertEquals(0, d.sample(Random(42)), "sample should return 0 when p=1")
    }

    // --- sample(n) validation ---

    @Test
    fun continuousSampleNegativeNThrows() {
        val d = NormalDistribution.STANDARD
        assertFailsWith<InvalidParameterException> { d.sample(-1, Random(42)) }
    }

    @Test
    fun discreteSampleNegativeNThrows() {
        val d = PoissonDistribution(5.0)
        assertFailsWith<InvalidParameterException> { d.sample(-1, Random(42)) }
    }

    @Test
    fun sampleZeroNReturnsEmpty() {
        val continuous = NormalDistribution.STANDARD
        assertTrue(continuous.sample(0, Random(42)).isEmpty())
        val discrete = PoissonDistribution(5.0)
        assertTrue(discrete.sample(0, Random(42)).isEmpty())
    }

    // --- Box-Muller / sampling robustness ---

    @Test
    fun normalSampleProducesFiniteValues() {
        val d = NormalDistribution.STANDARD
        val samples = d.sample(100_000, Random(42))
        assertTrue(samples.all { it.isFinite() }, "all samples should be finite")
    }

    @Test
    fun exponentialSampleProducesFiniteValues() {
        val d = ExponentialDistribution.STANDARD
        val samples = d.sample(100_000, Random(42))
        assertTrue(samples.all { it.isFinite() }, "all samples should be finite")
    }

    // --- Discrete cdf/sf NaN handling ---

    @Test
    fun discreteCdfDoubleNaNReturnsNaN() {
        val poisson = PoissonDistribution(rate = 5.0)
        assertTrue(poisson.cdf(Double.NaN).isNaN(), "cdf(NaN) should return NaN")
    }

    @Test
    fun discreteSfDoubleNaNReturnsNaN() {
        val binom = BinomialDistribution(10, 0.3)
        assertTrue(binom.sf(Double.NaN).isNaN(), "sf(NaN) should return NaN")
    }

    @Test
    fun discreteCdfDoubleInfinityHandled() {
        val binom = BinomialDistribution(10, 0.3)
        assertEquals(1.0, binom.cdf(Double.POSITIVE_INFINITY), 1e-15)
        assertEquals(0.0, binom.cdf(Double.NEGATIVE_INFINITY), 1e-15)
    }

    // --- logPdf boundary consistency ---

    @Test
    fun betaLogPdfBoundaryConsistency() {
        // alpha=1: pdf(0)=beta, logPdf(0)=ln(beta)
        val b1 = BetaDistribution(1.0, 5.0)
        assertEquals(b1.pdf(0.0), kotlin.math.exp(b1.logPdf(0.0)), 1e-12, "Beta(1,5): exp(logPdf(0)) ≈ pdf(0)")
        // beta=1: pdf(1)=alpha, logPdf(1)=ln(alpha)
        val b2 = BetaDistribution(3.0, 1.0)
        assertEquals(b2.pdf(1.0), kotlin.math.exp(b2.logPdf(1.0)), 1e-12, "Beta(3,1): exp(logPdf(1)) ≈ pdf(1)")
        // alpha>1: pdf(0)=0, logPdf(0)=-Inf
        val b3 = BetaDistribution(2.0, 5.0)
        assertEquals(0.0, b3.pdf(0.0), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, b3.logPdf(0.0))
    }

    @Test
    fun gammaLogPdfBoundaryConsistency() {
        // shape=1: pdf(0)=rate, logPdf(0)=ln(rate)
        val g1 = GammaDistribution(1.0, 2.0)
        assertEquals(g1.pdf(0.0), kotlin.math.exp(g1.logPdf(0.0)), 1e-12, "Gamma(1,2): exp(logPdf(0)) ≈ pdf(0)")
        // shape>1: pdf(0)=0, logPdf(0)=-Inf
        val g2 = GammaDistribution(2.0, 1.0)
        assertEquals(0.0, g2.pdf(0.0), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, g2.logPdf(0.0))
        // shape<1: pdf(0)=+Inf, logPdf(0)=+Inf
        val g3 = GammaDistribution(0.5, 1.0)
        assertEquals(Double.POSITIVE_INFINITY, g3.pdf(0.0))
        assertEquals(Double.POSITIVE_INFINITY, g3.logPdf(0.0))
    }

    @Test
    fun chiSquaredLogPdfBoundaryConsistency() {
        // df=2: pdf(0)=0.5, logPdf(0)=-ln(2)
        val c1 = ChiSquaredDistribution(2.0)
        assertEquals(c1.pdf(0.0), kotlin.math.exp(c1.logPdf(0.0)), 1e-12, "ChiSq(2): exp(logPdf(0)) ≈ pdf(0)")
        // df<2: both +Inf
        val c2 = ChiSquaredDistribution(1.0)
        assertEquals(Double.POSITIVE_INFINITY, c2.pdf(0.0))
        assertEquals(Double.POSITIVE_INFINITY, c2.logPdf(0.0))
    }

    @Test
    fun fDistributionLogPdfBoundaryConsistency() {
        // d1=2: pdf(0)=1.0, logPdf(0)=0.0
        val f1 = FDistribution(2.0, 10.0)
        assertEquals(f1.pdf(0.0), kotlin.math.exp(f1.logPdf(0.0)), 1e-12, "F(2,10): exp(logPdf(0)) ≈ pdf(0)")
        // d1>2: both 0 / -Inf
        val f2 = FDistribution(5.0, 10.0)
        assertEquals(0.0, f2.pdf(0.0), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, f2.logPdf(0.0))
    }

    @Test
    fun nakagamiLogPdfBoundaryConsistency() {
        // mu=0.5: pdf(0) = sqrt(2/(pi*omega))
        val n1 = NakagamiDistribution(0.5, 1.0)
        assertEquals(n1.pdf(0.0), kotlin.math.exp(n1.logPdf(0.0)), 1e-12, "Nakagami(0.5,1): exp(logPdf(0)) ≈ pdf(0)")
        // mu>0.5: both 0 / -Inf
        val n2 = NakagamiDistribution(2.0, 1.0)
        assertEquals(0.0, n2.pdf(0.0), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, n2.logPdf(0.0))
    }

    // --- Sampling finiteness for distributions with guards ---

    @Test
    fun laplaceSampleProducesFiniteValues() {
        val d = LaplaceDistribution.STANDARD
        val samples = d.sample(100_000, Random(42))
        assertTrue(samples.all { it.isFinite() }, "all Laplace samples should be finite")
    }

    @Test
    fun cauchySampleProducesFiniteValues() {
        val d = CauchyDistribution.STANDARD
        val samples = d.sample(100_000, Random(42))
        assertTrue(samples.all { it.isFinite() }, "all Cauchy samples should be finite")
    }

    @Test
    fun levySampleProducesFiniteValues() {
        val d = LevyDistribution.STANDARD
        val samples = d.sample(100_000, Random(42))
        assertTrue(samples.all { it.isFinite() }, "all Levy samples should be finite")
    }

    @Test
    fun gumbelSampleProducesFiniteValues() {
        val d = GumbelDistribution.STANDARD
        val samples = d.sample(100_000, Random(42))
        assertTrue(samples.all { it.isFinite() }, "all Gumbel samples should be finite")
    }

    @Test
    fun logisticSampleProducesFiniteValues() {
        val d = LogisticDistribution.STANDARD
        val samples = d.sample(100_000, Random(42))
        assertTrue(samples.all { it.isFinite() }, "all Logistic samples should be finite")
    }

    @Test
    fun paretoSampleProducesFiniteValues() {
        val d = ParetoDistribution.STANDARD
        val samples = d.sample(100_000, Random(42))
        assertTrue(samples.all { it.isFinite() }, "all Pareto samples should be finite")
    }

    // --- ExponentialDistribution NaN propagation ---

    @Test
    fun exponentialPdfNaNReturnsNaN() {
        val d = ExponentialDistribution.STANDARD
        assertTrue(d.pdf(Double.NaN).isNaN(), "pdf(NaN) should return NaN")
    }

    @Test
    fun exponentialLogPdfNaNReturnsNaN() {
        val d = ExponentialDistribution.STANDARD
        assertTrue(d.logPdf(Double.NaN).isNaN(), "logPdf(NaN) should return NaN")
    }

    @Test
    fun exponentialCdfNaNReturnsNaN() {
        val d = ExponentialDistribution.STANDARD
        assertTrue(d.cdf(Double.NaN).isNaN(), "cdf(NaN) should return NaN")
    }

    @Test
    fun exponentialSfNaNReturnsNaN() {
        val d = ExponentialDistribution.STANDARD
        assertTrue(d.sf(Double.NaN).isNaN(), "sf(NaN) should return NaN")
    }

    // --- Parameter validation: NaN and Infinity rejection ---

    @Test
    fun normalRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { NormalDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { NormalDistribution(0.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { NormalDistribution(0.0, Double.POSITIVE_INFINITY) }
        assertFailsWith<InvalidParameterException> { NormalDistribution(Double.POSITIVE_INFINITY, 1.0) }
    }

    @Test
    fun gammaRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { GammaDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { GammaDistribution(1.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { GammaDistribution(Double.POSITIVE_INFINITY, 1.0) }
        assertFailsWith<InvalidParameterException> { GammaDistribution(1.0, Double.POSITIVE_INFINITY) }
    }

    @Test
    fun exponentialRejectsNaNAndInfinityRate() {
        assertFailsWith<InvalidParameterException> { ExponentialDistribution(Double.NaN) }
        assertFailsWith<InvalidParameterException> { ExponentialDistribution(Double.POSITIVE_INFINITY) }
    }

    @Test
    fun betaRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { BetaDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { BetaDistribution(1.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { BetaDistribution(Double.POSITIVE_INFINITY, 1.0) }
        assertFailsWith<InvalidParameterException> { BetaDistribution(1.0, Double.POSITIVE_INFINITY) }
    }

    @Test
    fun studentTRejectsNaNAndInfinityDf() {
        assertFailsWith<InvalidParameterException> { StudentTDistribution(Double.NaN) }
        assertFailsWith<InvalidParameterException> { StudentTDistribution(Double.POSITIVE_INFINITY) }
    }

    @Test
    fun chiSquaredRejectsNaNAndInfinityDf() {
        assertFailsWith<InvalidParameterException> { ChiSquaredDistribution(Double.NaN) }
        assertFailsWith<InvalidParameterException> { ChiSquaredDistribution(Double.POSITIVE_INFINITY) }
    }

    @Test
    fun fDistributionRejectsNaNAndInfinityDf() {
        assertFailsWith<InvalidParameterException> { FDistribution(Double.NaN, 10.0) }
        assertFailsWith<InvalidParameterException> { FDistribution(10.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { FDistribution(Double.POSITIVE_INFINITY, 10.0) }
        assertFailsWith<InvalidParameterException> { FDistribution(10.0, Double.POSITIVE_INFINITY) }
    }

    @Test
    fun cauchyRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { CauchyDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { CauchyDistribution(0.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { CauchyDistribution(Double.POSITIVE_INFINITY, 1.0) }
        assertFailsWith<InvalidParameterException> { CauchyDistribution(0.0, Double.POSITIVE_INFINITY) }
    }

    @Test
    fun gumbelRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { GumbelDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { GumbelDistribution(0.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { GumbelDistribution(Double.POSITIVE_INFINITY, 1.0) }
        assertFailsWith<InvalidParameterException> { GumbelDistribution(0.0, Double.POSITIVE_INFINITY) }
    }

    @Test
    fun weibullRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { WeibullDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { WeibullDistribution(1.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { WeibullDistribution(Double.POSITIVE_INFINITY, 1.0) }
        assertFailsWith<InvalidParameterException> { WeibullDistribution(1.0, Double.POSITIVE_INFINITY) }
    }

    @Test
    fun paretoRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { ParetoDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { ParetoDistribution(1.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { ParetoDistribution(Double.POSITIVE_INFINITY, 1.0) }
        assertFailsWith<InvalidParameterException> { ParetoDistribution(1.0, Double.POSITIVE_INFINITY) }
    }

    @Test
    fun logNormalRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { LogNormalDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { LogNormalDistribution(0.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { LogNormalDistribution(0.0, Double.POSITIVE_INFINITY) }
    }

    @Test
    fun logisticRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { LogisticDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { LogisticDistribution(0.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { LogisticDistribution(0.0, Double.POSITIVE_INFINITY) }
    }

    @Test
    fun nakagamiRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { NakagamiDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { NakagamiDistribution(1.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { NakagamiDistribution(Double.POSITIVE_INFINITY, 1.0) }
        assertFailsWith<InvalidParameterException> { NakagamiDistribution(1.0, Double.POSITIVE_INFINITY) }
    }

    @Test
    fun levyRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { LevyDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { LevyDistribution(0.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { LevyDistribution(0.0, Double.POSITIVE_INFINITY) }
    }

    @Test
    fun laplaceRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { LaplaceDistribution(Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { LaplaceDistribution(0.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { LaplaceDistribution(0.0, Double.POSITIVE_INFINITY) }
    }

    @Test
    fun triangularRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { TriangularDistribution(Double.NaN, 1.0, 0.5) }
        assertFailsWith<InvalidParameterException> { TriangularDistribution(0.0, Double.NaN, 0.5) }
        assertFailsWith<InvalidParameterException> { TriangularDistribution(0.0, 1.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { TriangularDistribution(Double.POSITIVE_INFINITY, 1.0, 0.5) }
    }

    @Test
    fun geometricRejectsNaN() {
        assertFailsWith<InvalidParameterException> { GeometricDistribution(Double.NaN) }
    }

    @Test
    fun negativeBinomialRejectsNaN() {
        assertFailsWith<InvalidParameterException> { NegativeBinomialDistribution(5, Double.NaN) }
    }

    @Test
    fun betaBinomialRejectsNaNAndInfinityParameters() {
        assertFailsWith<InvalidParameterException> { BetaBinomialDistribution(10, Double.NaN, 1.0) }
        assertFailsWith<InvalidParameterException> { BetaBinomialDistribution(10, 1.0, Double.NaN) }
        assertFailsWith<InvalidParameterException> { BetaBinomialDistribution(10, Double.POSITIVE_INFINITY, 1.0) }
        assertFailsWith<InvalidParameterException> { BetaBinomialDistribution(10, 1.0, Double.POSITIVE_INFINITY) }
    }

    // ── DiscreteDistribution.cdf(Double) integer overflow fix ────────────

    @Test
    fun discreteCdfHandlesExtremeDoubleValues() {
        val dist = BinomialDistribution(100, 0.5)
        // Values beyond Int range should not overflow
        assertEquals(1.0, dist.cdf(3e9), tol)
        assertEquals(1.0, dist.cdf(Double.POSITIVE_INFINITY), tol)
        assertEquals(0.0, dist.cdf(-3e9), tol)
        assertEquals(0.0, dist.cdf(Double.NEGATIVE_INFINITY), tol)
        assertTrue(dist.cdf(Double.NaN).isNaN())
    }

    @Test
    fun discreteSfHandlesExtremeDoubleValues() {
        val dist = BinomialDistribution(100, 0.5)
        assertEquals(0.0, dist.sf(3e9), tol)
        assertEquals(0.0, dist.sf(Double.POSITIVE_INFINITY), tol)
        assertEquals(1.0, dist.sf(-3e9), tol)
        assertEquals(1.0, dist.sf(Double.NEGATIVE_INFINITY), tol)
        assertTrue(dist.sf(Double.NaN).isNaN())
    }

    @Test
    fun zipfRejectsLargeNumberOfElements() {
        assertFailsWith<InvalidParameterException> { ZipfDistribution(20_000_000, 1.0) }
    }
}
