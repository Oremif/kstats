package org.oremif.kstats.benchmark.distributions

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.State as JmhState
import org.oremif.kstats.distributions.FDistribution
import org.apache.commons.math3.distribution.FDistribution as CommonsF

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
open class FDistributionBenchmark {

    private val kDist = FDistribution(dfNumerator = 5.0, dfDenominator = 10.0)
    private val cDist = CommonsF(5.0, 10.0)

    private val xValue = 1.0
    private val pValue = 0.5

    // ===== PDF =====

    @Benchmark fun kstatsFPdf(): Double = kDist.pdf(xValue)
    @Benchmark fun commonsFPdf(): Double = cDist.density(xValue)

    // ===== CDF =====

    @Benchmark fun kstatsFCdf(): Double = kDist.cdf(xValue)
    @Benchmark fun commonsFCdf(): Double = cDist.cumulativeProbability(xValue)

    // ===== Quantile =====

    @Benchmark fun kstatsFQuantile(): Double = kDist.quantile(pValue)
    @Benchmark fun commonsFQuantile(): Double = cDist.inverseCumulativeProbability(pValue)
}
