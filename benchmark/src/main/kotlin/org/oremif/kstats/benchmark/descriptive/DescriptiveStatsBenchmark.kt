package org.oremif.kstats.benchmark.descriptive

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.State as JmhState
import org.oremif.kstats.benchmark.util.DataGenerators
import org.oremif.kstats.descriptive.*
import org.apache.commons.math3.stat.StatUtils
import org.apache.commons.math3.stat.descriptive.moment.Kurtosis as CommonsKurtosis
import org.apache.commons.math3.stat.descriptive.moment.Skewness as CommonsSkewness
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation as CommonsStdDev
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics as CommonsDescriptiveStats

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class DescriptiveStatsBenchmark {

    @Param("100", "1000", "10000", "100000")
    var size: Int = 0

    private lateinit var data: DoubleArray
    private lateinit var positiveData: DoubleArray

    @Setup
    fun setup() {
        data = DataGenerators.generateDoubleArray(size)
        positiveData = DataGenerators.generatePositiveArray(size)
    }

    // --- Mean ---

    @Benchmark
    fun kstatsMean(): Double = data.mean()

    @Benchmark
    fun commonsMean(): Double = StatUtils.mean(data)

    // --- Variance ---

    @Benchmark
    fun kstatsVariance(): Double = data.variance()

    @Benchmark
    fun commonsVariance(): Double = StatUtils.variance(data)

    // --- Standard Deviation ---

    @Benchmark
    fun kstatsStdDev(): Double = data.standardDeviation()

    @Benchmark
    fun commonsStdDev(): Double = CommonsStdDev().evaluate(data)

    // --- Geometric Mean ---

    @Benchmark
    fun kstatsGeometricMean(): Double = positiveData.geometricMean()

    @Benchmark
    fun commonsGeometricMean(): Double = StatUtils.geometricMean(positiveData)

    // --- Median ---

    @Benchmark
    fun kstatsMedian(): Double = data.median()

    @Benchmark
    fun commonsMedian(): Double {
        val ds = CommonsDescriptiveStats(data.size)
        for (v in data) ds.addValue(v)
        return ds.getPercentile(50.0)
    }

    // --- Skewness ---

    @Benchmark
    fun kstatsSkewness(): Double = data.skewness()

    @Benchmark
    fun commonsSkewness(): Double = CommonsSkewness().evaluate(data)

    // --- Kurtosis ---

    @Benchmark
    fun kstatsKurtosis(): Double = data.kurtosis()

    @Benchmark
    fun commonsKurtosis(): Double = CommonsKurtosis().evaluate(data)

    // --- Full Descriptive Statistics ---

    @Benchmark
    fun kstatsDescribe(): DescriptiveStatistics = data.describe()

    @Benchmark
    fun commonsDescribe(): CommonsDescriptiveStats {
        val ds = CommonsDescriptiveStats(data.size)
        for (v in data) ds.addValue(v)
        return ds
    }
}
