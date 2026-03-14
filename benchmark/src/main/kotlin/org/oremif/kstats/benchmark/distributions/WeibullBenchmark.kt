package org.oremif.kstats.benchmark.distributions

import kotlinx.benchmark.*
import org.oremif.kstats.distributions.WeibullDistribution
import org.apache.commons.math3.distribution.WeibullDistribution as CommonsWeibull
import org.openjdk.jmh.annotations.State as JmhState

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class WeibullBenchmark {

    private val kDist = WeibullDistribution(shape = 1.5, scale = 1.0)
    private val cDist = CommonsWeibull(1.5, 1.0)

    private val xValue = 0.7
    private val pValue = 0.5

    // ===== PDF =====

    @Benchmark
    fun kstatsWeibullPdf(): Double = kDist.pdf(xValue)
    @Benchmark
    fun commonsWeibullPdf(): Double = cDist.density(xValue)

    // ===== CDF =====

    @Benchmark
    fun kstatsWeibullCdf(): Double = kDist.cdf(xValue)
    @Benchmark
    fun commonsWeibullCdf(): Double = cDist.cumulativeProbability(xValue)

    // ===== Quantile =====

    @Benchmark
    fun kstatsWeibullQuantile(): Double = kDist.quantile(pValue)
    @Benchmark
    fun commonsWeibullQuantile(): Double = cDist.inverseCumulativeProbability(pValue)
}
