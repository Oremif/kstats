package org.oremif.kstats.benchmark.distributions

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.State as JmhState
import org.oremif.kstats.distributions.*
import org.apache.commons.math3.distribution.NormalDistribution as CommonsNormal
import org.apache.commons.math3.distribution.BetaDistribution as CommonsBeta
import org.apache.commons.math3.distribution.GammaDistribution as CommonsGamma
import org.apache.commons.math3.distribution.ChiSquaredDistribution as CommonsChiSquared
import org.apache.commons.math3.distribution.TDistribution as CommonsStudentT
import org.apache.commons.math3.distribution.ExponentialDistribution as CommonsExponential

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
open class ContinuousDistributionBenchmark {

    // kstats distributions
    private val kNormal = NormalDistribution(0.0, 1.0)
    private val kBeta = BetaDistribution(2.0, 5.0)
    private val kGamma = GammaDistribution(shape = 2.0, rate = 1.0)
    private val kChiSq = ChiSquaredDistribution(5.0)
    private val kStudentT = StudentTDistribution(10.0)
    private val kExponential = ExponentialDistribution(rate = 2.0)

    // Commons Math distributions
    // Note: Commons Gamma uses (shape, scale) where scale = 1/rate
    // Note: Commons Exponential uses mean = 1/rate
    private val cNormal = CommonsNormal(0.0, 1.0)
    private val cBeta = CommonsBeta(2.0, 5.0)
    private val cGamma = CommonsGamma(2.0, 1.0) // shape=2, scale=1/1=1
    private val cChiSq = CommonsChiSquared(5.0)
    private val cStudentT = CommonsStudentT(10.0)
    private val cExponential = CommonsExponential(0.5) // mean = 1/rate = 1/2 = 0.5

    private val xValue = 0.5
    private val pValue = 0.75

    // ===== Normal =====

    @Benchmark fun kstatsNormalPdf(): Double = kNormal.pdf(xValue)
    @Benchmark fun commonsNormalPdf(): Double = cNormal.density(xValue)

    @Benchmark fun kstatsNormalCdf(): Double = kNormal.cdf(xValue)
    @Benchmark fun commonsNormalCdf(): Double = cNormal.cumulativeProbability(xValue)

    @Benchmark fun kstatsNormalQuantile(): Double = kNormal.quantile(pValue)
    @Benchmark fun commonsNormalQuantile(): Double = cNormal.inverseCumulativeProbability(pValue)

    // ===== Beta =====

    @Benchmark fun kstatsBetaPdf(): Double = kBeta.pdf(xValue)
    @Benchmark fun commonsBetaPdf(): Double = cBeta.density(xValue)

    @Benchmark fun kstatsBetaCdf(): Double = kBeta.cdf(xValue)
    @Benchmark fun commonsBetaCdf(): Double = cBeta.cumulativeProbability(xValue)

    @Benchmark fun kstatsBetaQuantile(): Double = kBeta.quantile(pValue)
    @Benchmark fun commonsBetaQuantile(): Double = cBeta.inverseCumulativeProbability(pValue)

    // ===== Gamma =====

    @Benchmark fun kstatsGammaPdf(): Double = kGamma.pdf(xValue)
    @Benchmark fun commonsGammaPdf(): Double = cGamma.density(xValue)

    @Benchmark fun kstatsGammaCdf(): Double = kGamma.cdf(xValue)
    @Benchmark fun commonsGammaCdf(): Double = cGamma.cumulativeProbability(xValue)

    @Benchmark fun kstatsGammaQuantile(): Double = kGamma.quantile(pValue)
    @Benchmark fun commonsGammaQuantile(): Double = cGamma.inverseCumulativeProbability(pValue)

    // ===== Chi-Squared =====

    @Benchmark fun kstatsChiSqPdf(): Double = kChiSq.pdf(xValue)
    @Benchmark fun commonsChiSqPdf(): Double = cChiSq.density(xValue)

    @Benchmark fun kstatsChiSqCdf(): Double = kChiSq.cdf(xValue)
    @Benchmark fun commonsChiSqCdf(): Double = cChiSq.cumulativeProbability(xValue)

    @Benchmark fun kstatsChiSqQuantile(): Double = kChiSq.quantile(pValue)
    @Benchmark fun commonsChiSqQuantile(): Double = cChiSq.inverseCumulativeProbability(pValue)

    // ===== Student's t =====

    @Benchmark fun kstatsStudentTPdf(): Double = kStudentT.pdf(xValue)
    @Benchmark fun commonsStudentTPdf(): Double = cStudentT.density(xValue)

    @Benchmark fun kstatsStudentTCdf(): Double = kStudentT.cdf(xValue)
    @Benchmark fun commonsStudentTCdf(): Double = cStudentT.cumulativeProbability(xValue)

    @Benchmark fun kstatsStudentTQuantile(): Double = kStudentT.quantile(pValue)
    @Benchmark fun commonsStudentTQuantile(): Double = cStudentT.inverseCumulativeProbability(pValue)

    // ===== Exponential =====

    @Benchmark fun kstatsExponentialPdf(): Double = kExponential.pdf(xValue)
    @Benchmark fun commonsExponentialPdf(): Double = cExponential.density(xValue)

    @Benchmark fun kstatsExponentialCdf(): Double = kExponential.cdf(xValue)
    @Benchmark fun commonsExponentialCdf(): Double = cExponential.cumulativeProbability(xValue)

    @Benchmark fun kstatsExponentialQuantile(): Double = kExponential.quantile(pValue)
    @Benchmark fun commonsExponentialQuantile(): Double = cExponential.inverseCumulativeProbability(pValue)
}
