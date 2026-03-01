package org.oremif.kstats.benchmark.distributions

import kotlinx.benchmark.*
import org.oremif.kstats.distributions.HypergeometricDistribution
import org.apache.commons.math3.distribution.HypergeometricDistribution as CommonsHypergeometric
import org.openjdk.jmh.annotations.State as JmhState

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class HypergeometricBenchmark {

    private val kDist = HypergeometricDistribution(population = 50, successes = 15, draws = 10)
    private val cDist = CommonsHypergeometric(50, 15, 10)

    private val kValue = 3
    private val pValue = 0.5

    // ===== PMF =====

    @Benchmark
    fun kstatsHypergeometricPmf(): Double = kDist.pmf(kValue)
    @Benchmark
    fun commonsHypergeometricPmf(): Double = cDist.probability(kValue)

    // ===== CDF =====

    @Benchmark
    fun kstatsHypergeometricCdf(): Double = kDist.cdf(kValue)
    @Benchmark
    fun commonsHypergeometricCdf(): Double = cDist.cumulativeProbability(kValue)

    // ===== Quantile =====

    @Benchmark
    fun kstatsHypergeometricQuantile(): Int = kDist.quantileInt(pValue)
    @Benchmark
    fun commonsHypergeometricQuantile(): Int = cDist.inverseCumulativeProbability(pValue)
}
