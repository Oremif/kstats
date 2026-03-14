package org.oremif.kstats.benchmark.hypothesis

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.Param
import org.openjdk.jmh.annotations.State as JmhState
import org.oremif.kstats.benchmark.util.DataGenerators
import org.oremif.kstats.hypothesis.*
import org.apache.commons.math3.stat.inference.TTest as CommonsTTest

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
open class PairedTTestBenchmark {

    @Param("100", "1000", "10000")
    var size: Int = 0

    private lateinit var sample1: DoubleArray
    private lateinit var sample2: DoubleArray

    private val commonsTTest = CommonsTTest()

    @Setup
    fun setup() {
        sample1 = DataGenerators.generateDoubleArray(size, seed = 42L)
        sample2 = DataGenerators.generateDoubleArray(size, seed = 99L)
    }

    // --- Paired t-test ---

    @Benchmark
    fun kstatsPairedTTest(): TestResult = pairedTTest(sample1, sample2)

    @Benchmark
    fun commonsPairedTTest(): Double = commonsTTest.pairedTTest(sample1, sample2)
}
