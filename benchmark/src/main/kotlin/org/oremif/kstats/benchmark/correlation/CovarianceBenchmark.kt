package org.oremif.kstats.benchmark.correlation

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.Param
import org.oremif.kstats.benchmark.util.DataGenerators
import org.oremif.kstats.correlation.covariance
import org.apache.commons.math3.stat.correlation.Covariance as CommonsCovariance
import org.openjdk.jmh.annotations.State as JmhState

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class CovarianceBenchmark {

    @Param("100", "1000", "10000")
    var size: Int = 0

    private lateinit var x: DoubleArray
    private lateinit var y: DoubleArray

    private val commonsCovariance = CommonsCovariance()

    @Setup
    fun setup() {
        val pair = DataGenerators.generateCorrelatedPair(size)
        x = pair.first
        y = pair.second
    }

    // --- Covariance ---

    @Benchmark
    fun kstatsCovariance(): Double = covariance(x, y)

    @Benchmark
    fun commonsCovariance(): Double = commonsCovariance.covariance(x, y)
}
