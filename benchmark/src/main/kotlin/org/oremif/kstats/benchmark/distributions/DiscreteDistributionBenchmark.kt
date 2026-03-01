package org.oremif.kstats.benchmark.distributions

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.State as JmhState
import org.oremif.kstats.distributions.BinomialDistribution
import org.oremif.kstats.distributions.PoissonDistribution
import org.apache.commons.math3.distribution.BinomialDistribution as CommonsBinomial
import org.apache.commons.math3.distribution.PoissonDistribution as CommonsPoisson

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.NANOSECONDS)
class DiscreteDistributionBenchmark {

    // kstats distributions
    private val kBinomial = BinomialDistribution(trials = 100, probability = 0.3)
    private val kPoisson = PoissonDistribution(rate = 5.0)

    // Commons Math distributions
    private val cBinomial = CommonsBinomial(100, 0.3)
    private val cPoisson = CommonsPoisson(5.0)

    private val kValue = 30
    private val poissonK = 5

    // ===== Binomial =====

    @Benchmark fun kstatsBinomialPmf(): Double = kBinomial.pmf(kValue)
    @Benchmark fun commonsBinomialPmf(): Double = cBinomial.probability(kValue)

    @Benchmark fun kstatsBinomialCdf(): Double = kBinomial.cdf(kValue)
    @Benchmark fun commonsBinomialCdf(): Double = cBinomial.cumulativeProbability(kValue)

    // ===== Poisson =====

    @Benchmark fun kstatsPoissonPmf(): Double = kPoisson.pmf(poissonK)
    @Benchmark fun commonsPoissonPmf(): Double = cPoisson.probability(poissonK)

    @Benchmark fun kstatsPoissonCdf(): Double = kPoisson.cdf(poissonK)
    @Benchmark fun commonsPoissonCdf(): Double = cPoisson.cumulativeProbability(poissonK)
}
