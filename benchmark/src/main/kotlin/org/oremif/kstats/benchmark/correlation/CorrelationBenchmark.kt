package org.oremif.kstats.benchmark.correlation

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.State as JmhState
import org.oremif.kstats.benchmark.util.DataGenerators
import org.oremif.kstats.correlation.*
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation as CommonsPearson
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation as CommonsSpearman
import org.apache.commons.math3.stat.correlation.KendallsCorrelation as CommonsKendall
import org.apache.commons.math3.stat.regression.SimpleRegression as CommonsRegression

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
open class CorrelationBenchmark {

    @Param("100", "1000", "10000")
    var size: Int = 0

    private lateinit var x: DoubleArray
    private lateinit var y: DoubleArray

    private val commonsPearson = CommonsPearson()
    private val commonsSpearman = CommonsSpearman()
    private val commonsKendall = CommonsKendall()

    @Setup
    fun setup() {
        val pair = DataGenerators.generateCorrelatedPair(size)
        x = pair.first
        y = pair.second
    }

    // --- Pearson ---

    @Benchmark
    fun kstatsPearson(): CorrelationResult = pearsonCorrelation(x, y)

    @Benchmark
    fun commonsPearson(): Double = commonsPearson.correlation(x, y)

    // --- Spearman ---

    @Benchmark
    fun kstatsSpearman(): CorrelationResult = spearmanCorrelation(x, y)

    @Benchmark
    fun commonsSpearman(): Double = commonsSpearman.correlation(x, y)

    // --- Kendall Tau ---

    @Benchmark
    fun kstatsKendallTau(): CorrelationResult = kendallTau(x, y)

    @Benchmark
    fun commonsKendallTau(): Double = commonsKendall.correlation(x, y)

    // --- Simple Linear Regression ---

    @Benchmark
    fun kstatsLinearRegression(): SimpleLinearRegressionResult = simpleLinearRegression(x, y)

    @Benchmark
    fun commonsLinearRegression(): CommonsRegression {
        val reg = CommonsRegression()
        for (i in x.indices) reg.addData(x[i], y[i])
        return reg
    }
}
