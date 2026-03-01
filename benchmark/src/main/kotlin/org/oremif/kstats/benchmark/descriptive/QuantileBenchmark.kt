package org.oremif.kstats.benchmark.descriptive

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.State as JmhState
import org.oremif.kstats.benchmark.util.DataGenerators
import org.oremif.kstats.descriptive.interquartileRange
import org.oremif.kstats.descriptive.percentile
import org.apache.commons.math3.stat.descriptive.rank.Percentile as CommonsPercentile

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class QuantileBenchmark {

    @Param("1000", "10000", "100000")
    var size: Int = 0

    private lateinit var data: DoubleArray

    private val commonsPercentile = CommonsPercentile()

    @Setup
    fun setup() {
        data = DataGenerators.generateDoubleArray(size)
    }

    // --- Median (percentile 50) ---

    @Benchmark
    fun kstatsPercentile50(): Double = data.percentile(50.0)

    @Benchmark
    fun commonsPercentile50(): Double = commonsPercentile.evaluate(data, 50.0)

    // --- 95th percentile ---

    @Benchmark
    fun kstatsPercentile95(): Double = data.percentile(95.0)

    @Benchmark
    fun commonsPercentile95(): Double = commonsPercentile.evaluate(data, 95.0)

    // --- IQR ---

    @Benchmark
    fun kstatsIqr(): Double = data.interquartileRange()

    @Benchmark
    fun commonsIqr(): Double =
        commonsPercentile.evaluate(data, 75.0) - commonsPercentile.evaluate(data, 25.0)
}
