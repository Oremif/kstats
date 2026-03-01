package org.oremif.kstats.benchmark.hypothesis

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.State as JmhState
import org.oremif.kstats.benchmark.util.DataGenerators
import org.oremif.kstats.hypothesis.*
import org.apache.commons.math3.stat.inference.TTest as CommonsTTest
import org.apache.commons.math3.stat.inference.OneWayAnova as CommonsAnova
import org.apache.commons.math3.stat.inference.ChiSquareTest as CommonsChiSq
import org.apache.commons.math3.stat.inference.MannWhitneyUTest as CommonsMannWhitney
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest as CommonsKS

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class HypothesisTestBenchmark {

    @Param("100", "1000", "10000")
    var size: Int = 0

    private lateinit var sample1: DoubleArray
    private lateinit var sample2: DoubleArray
    private lateinit var sample3: DoubleArray
    private lateinit var chiObserved: IntArray
    private lateinit var chiExpected: DoubleArray

    private val commonsTTest = CommonsTTest()
    private val commonsAnova = CommonsAnova()
    private val commonsChiSq = CommonsChiSq()
    private val commonsMannWhitney = CommonsMannWhitney()
    private val commonsKS = CommonsKS()

    @Setup
    fun setup() {
        sample1 = DataGenerators.generateDoubleArray(size, seed = 42L)
        sample2 = DataGenerators.generateDoubleArray(size, seed = 99L)
        sample3 = DataGenerators.generateDoubleArray(size, seed = 7L)

        // Chi-squared test data: 10 bins
        val random = kotlin.random.Random(42)
        chiObserved = IntArray(10) { random.nextInt(10, 100) }
        val total = chiObserved.sum().toDouble()
        chiExpected = DoubleArray(10) { total / 10.0 }
    }

    // --- One-sample t-test ---

    @Benchmark
    fun kstatsOneSampleTTest(): TestResult = tTest(sample1, mu = 0.0)

    @Benchmark
    fun commonsOneSampleTTest(): Double = commonsTTest.tTest(0.0, sample1)

    // --- Two-sample t-test (Welch) ---

    @Benchmark
    fun kstatsTwoSampleTTest(): TestResult = tTest(sample1, sample2)

    @Benchmark
    fun commonsTwoSampleTTest(): Double = commonsTTest.tTest(sample1, sample2)

    // --- One-way ANOVA ---

    @Benchmark
    fun kstatsOneWayAnova(): AnovaResult = oneWayAnova(sample1, sample2, sample3)

    @Benchmark
    fun commonsOneWayAnova(): Double =
        commonsAnova.anovaPValue(listOf(sample1, sample2, sample3))

    // --- Chi-squared test ---

    @Benchmark
    fun kstatsChiSquared(): TestResult = chiSquaredTest(chiObserved, chiExpected)

    @Benchmark
    fun commonsChiSquared(): Double {
        val longObserved = LongArray(chiObserved.size) { chiObserved[it].toLong() }
        return commonsChiSq.chiSquareTest(chiExpected, longObserved)
    }

    // --- Mann-Whitney U ---

    @Benchmark
    fun kstatsMannWhitney(): TestResult = mannWhitneyUTest(sample1, sample2)

    @Benchmark
    fun commonsMannWhitney(): Double = commonsMannWhitney.mannWhitneyUTest(sample1, sample2)

    // --- Kolmogorov-Smirnov (two-sample) ---

    @Benchmark
    fun kstatsKolmogorovSmirnov(): TestResult = kolmogorovSmirnovTest(sample1, sample2)

    @Benchmark
    fun commonsKolmogorovSmirnov(): Double = commonsKS.kolmogorovSmirnovTest(sample1, sample2)
}
