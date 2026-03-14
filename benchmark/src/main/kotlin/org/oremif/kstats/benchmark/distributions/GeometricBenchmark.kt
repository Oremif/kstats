package org.oremif.kstats.benchmark.distributions

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.State as JmhState
import org.oremif.kstats.distributions.GeometricDistribution
import org.apache.commons.math3.distribution.GeometricDistribution as CommonsGeometric

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
open class GeometricBenchmark {

    private val kDist = GeometricDistribution(probability = 0.3)
    private val cDist = CommonsGeometric(0.3)

    private val kValue = 2
    private val pValue = 0.5

    // ===== PMF =====

    @Benchmark fun kstatsGeometricPmf(): Double = kDist.pmf(kValue)
    @Benchmark fun commonsGeometricPmf(): Double = cDist.probability(kValue)

    // ===== CDF =====

    @Benchmark fun kstatsGeometricCdf(): Double = kDist.cdf(kValue)
    @Benchmark fun commonsGeometricCdf(): Double = cDist.cumulativeProbability(kValue)

    // ===== Quantile =====

    @Benchmark fun kstatsGeometricQuantile(): Int = kDist.quantileInt(pValue)
    @Benchmark fun commonsGeometricQuantile(): Int = cDist.inverseCumulativeProbability(pValue)
}
