package org.oremif.kstats.benchmark.distributions

import kotlinx.benchmark.*
import org.oremif.kstats.distributions.LogNormalDistribution
import org.apache.commons.math3.distribution.LogNormalDistribution as CommonsLogNormal
import org.openjdk.jmh.annotations.State as JmhState

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class LogNormalBenchmark {

    private val kDist = LogNormalDistribution(mu = 0.0, sigma = 1.0)
    private val cDist = CommonsLogNormal(0.0, 1.0)

    private val xValue = 1.0
    private val pValue = 0.5

    // ===== PDF =====

    @Benchmark
    fun kstatsLogNormalPdf(): Double = kDist.pdf(xValue)
    @Benchmark
    fun commonsLogNormalPdf(): Double = cDist.density(xValue)

    // ===== CDF =====

    @Benchmark
    fun kstatsLogNormalCdf(): Double = kDist.cdf(xValue)
    @Benchmark
    fun commonsLogNormalCdf(): Double = cDist.cumulativeProbability(xValue)

    // ===== Quantile =====

    @Benchmark
    fun kstatsLogNormalQuantile(): Double = kDist.quantile(pValue)
    @Benchmark
    fun commonsLogNormalQuantile(): Double = cDist.inverseCumulativeProbability(pValue)
}
