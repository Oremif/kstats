package org.oremif.kstats.benchmark.distributions

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.State as JmhState
import org.oremif.kstats.distributions.ZipfDistribution
import org.apache.commons.math3.distribution.ZipfDistribution as CommonsZipf

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class ZipfDistributionBenchmark {

    // kstats distribution
    private val kZipf = ZipfDistribution(numberOfElements = 1000, exponent = 1.5)

    // Commons Math distribution
    private val cZipf = CommonsZipf(1000, 1.5)

    private val kValue = 1
    private val pValue = 0.5

    // ===== Zipf PMF =====

    @Benchmark fun kstatsZipfPmf(): Double = kZipf.pmf(kValue)
    @Benchmark fun commonsZipfPmf(): Double = cZipf.probability(kValue)

    // ===== Zipf CDF =====

    @Benchmark fun kstatsZipfCdf(): Double = kZipf.cdf(kValue)
    @Benchmark fun commonsZipfCdf(): Double = cZipf.cumulativeProbability(kValue)

    // ===== Zipf Quantile =====

    @Benchmark fun kstatsZipfQuantile(): Int = kZipf.quantileInt(pValue)
    @Benchmark fun commonsZipfQuantile(): Int = cZipf.inverseCumulativeProbability(pValue)
}
