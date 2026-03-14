package org.oremif.kstats.benchmark.distributions

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.State as JmhState
import org.oremif.kstats.distributions.UniformDistribution
import org.apache.commons.math3.distribution.UniformRealDistribution as CommonsUniform

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
open class UniformContinuousBenchmark {

    private val kDist = UniformDistribution(0.0, 1.0)
    private val cDist = CommonsUniform(0.0, 1.0)

    private val xValue = 0.5
    private val pValue = 0.5

    // ===== PDF =====

    @Benchmark fun kstatsUniformPdf(): Double = kDist.pdf(xValue)
    @Benchmark fun commonsUniformPdf(): Double = cDist.density(xValue)

    // ===== CDF =====

    @Benchmark fun kstatsUniformCdf(): Double = kDist.cdf(xValue)
    @Benchmark fun commonsUniformCdf(): Double = cDist.cumulativeProbability(xValue)

    // ===== Quantile =====

    @Benchmark fun kstatsUniformQuantile(): Double = kDist.quantile(pValue)
    @Benchmark fun commonsUniformQuantile(): Double = cDist.inverseCumulativeProbability(pValue)
}
