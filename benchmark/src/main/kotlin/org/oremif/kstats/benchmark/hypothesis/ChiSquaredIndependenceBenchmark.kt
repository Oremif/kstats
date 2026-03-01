package org.oremif.kstats.benchmark.hypothesis

import kotlinx.benchmark.*
import org.openjdk.jmh.annotations.Param
import org.oremif.kstats.hypothesis.TestResult
import org.oremif.kstats.hypothesis.chiSquaredIndependenceTest
import org.apache.commons.math3.stat.inference.ChiSquareTest as CommonsChiSq
import org.openjdk.jmh.annotations.State as JmhState

@JmhState(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(BenchmarkTimeUnit.MICROSECONDS)
class ChiSquaredIndependenceBenchmark {

    @Param("3", "10", "50")
    var tableSize: Int = 0

    private lateinit var kstatsTable: Array<IntArray>
    private lateinit var commonsTable: Array<LongArray>

    private val commonsChiSq = CommonsChiSq()

    @Setup
    fun setup() {
        val random = kotlin.random.Random(42)
        kstatsTable = Array(tableSize) { IntArray(tableSize) { random.nextInt(10, 100) } }
        commonsTable = Array(tableSize) { r -> LongArray(tableSize) { c -> kstatsTable[r][c].toLong() } }
    }

    // --- Chi-squared independence test ---

    @Benchmark
    fun kstatsChiSquaredIndependence(): TestResult = chiSquaredIndependenceTest(kstatsTable)

    @Benchmark
    fun commonsChiSquaredIndependence(): Double = commonsChiSq.chiSquareTest(commonsTable)
}
