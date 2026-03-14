package org.oremif.kstats.benchmark.distributions

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.State as JmhState
import org.oremif.kstats.distributions.CauchyDistribution
import org.apache.commons.math3.distribution.CauchyDistribution as CommonsCauchy

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
open class CauchyBenchmark {

    private val kDist = CauchyDistribution(location = 0.0, scale = 1.0)
    private val cDist = CommonsCauchy(0.0, 1.0)

    private val xValue = 0.0
    private val pValue = 0.75

    // ===== PDF =====

    @Benchmark fun kstatsCauchyPdf(): Double = kDist.pdf(xValue)
    @Benchmark fun commonsCauchyPdf(): Double = cDist.density(xValue)

    // ===== CDF =====

    @Benchmark fun kstatsCauchyCdf(): Double = kDist.cdf(xValue)
    @Benchmark fun commonsCauchyCdf(): Double = cDist.cumulativeProbability(xValue)

    // ===== Quantile =====

    @Benchmark fun kstatsCauchyQuantile(): Double = kDist.quantile(pValue)
    @Benchmark fun commonsCauchyQuantile(): Double = cDist.inverseCumulativeProbability(pValue)
}
