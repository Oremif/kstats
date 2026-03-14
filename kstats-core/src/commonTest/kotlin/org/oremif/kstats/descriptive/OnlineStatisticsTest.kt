package org.oremif.kstats.descriptive

import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class OnlineStatisticsTest {

    // ── Basic correctness ──────────────────────────────────────────────────

    @Test
    fun testBasicMeanAndVariance() {
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val stats = OnlineStatistics()
        stats.addAll(data)

        assertEquals(8L, stats.count)
        assertEquals(5.0, stats.mean, 1e-10)
        assertEquals(4.571428571428571, stats.variance(), 1e-10) // sample variance
        assertEquals(4.0, stats.variance(PopulationKind.POPULATION), 1e-10)
    }

    @Test
    fun testSumMinMax() {
        val data = doubleArrayOf(3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0)
        val stats = OnlineStatistics()
        stats.addAll(data)

        assertEquals(31.0, stats.sum, 1e-10)
        assertEquals(1.0, stats.min, 1e-10)
        assertEquals(9.0, stats.max, 1e-10)
    }

    @Test
    fun testStandardDeviation() {
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val stats = OnlineStatistics()
        stats.addAll(data)

        assertEquals(sqrt(stats.variance()), stats.standardDeviation(), 1e-15)
        assertEquals(
            sqrt(stats.variance(PopulationKind.POPULATION)),
            stats.standardDeviation(PopulationKind.POPULATION),
            1e-15
        )
    }

    @Test
    fun testCrossValidationWithBatchFunctions() {
        // scipy.stats.describe([1,2,3,5,8,13,21])
        // mean=7.571..., variance=52.952..., skewness=1.0897..., kurtosis=-0.3044...
        val data = doubleArrayOf(1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0)
        val stats = OnlineStatistics()
        stats.addAll(data)

        // Cross-validate with batch extension functions
        assertEquals(data.toList().mean(), stats.mean, 1e-10)
        assertEquals(data.variance(), stats.variance(), 1e-10)
        assertEquals(data.standardDeviation(), stats.standardDeviation(), 1e-10)
    }

    @Test
    fun testSkewnessAgainstBatch() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0)
        val stats = OnlineStatistics()
        stats.addAll(data)

        val batchSkewness = data.skewness()
        assertEquals(batchSkewness, stats.skewness, 1e-10)
    }

    @Test
    fun testKurtosisAgainstBatch() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0)
        val stats = OnlineStatistics()
        stats.addAll(data)

        val batchKurtosis = data.kurtosis()
        assertEquals(batchKurtosis, stats.kurtosis, 1e-10)
    }

    @Test
    fun testScipyReferenceValues() {
        // Data: [1..10], symmetric uniform
        // scipy.stats.describe: mean=5.5, variance=9.1667 (sample), skewness=0.0
        // Sample-adjusted excess kurtosis = -6/5 = -1.2 (exact)
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val stats = OnlineStatistics()
        stats.addAll(data)

        assertEquals(10L, stats.count)
        assertEquals(5.5, stats.mean, 1e-10)
        assertEquals(9.166666666666666, stats.variance(), 1e-10)
        assertEquals(0.0, stats.skewness, 1e-10)
        assertEquals(-1.2, stats.kurtosis, 1e-10)
    }

    @Test
    fun testAddAllIterable() {
        val list = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val stats = OnlineStatistics()
        stats.addAll(list)

        assertEquals(5L, stats.count)
        assertEquals(3.0, stats.mean, 1e-10)
    }

    @Test
    fun testIncrementalAdd() {
        val stats = OnlineStatistics()
        stats.add(1.0)
        stats.add(2.0)
        stats.add(3.0)
        stats.add(4.0)
        stats.add(5.0)

        val batch = OnlineStatistics()
        batch.addAll(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0))

        assertEquals(batch.mean, stats.mean, 1e-15)
        assertEquals(batch.variance(), stats.variance(), 1e-15)
        assertEquals(batch.skewness, stats.skewness, 1e-15)
    }

    // ── Edge cases ─────────────────────────────────────────────────────────

    @Test
    fun testEmptyStatistics() {
        val stats = OnlineStatistics()

        assertEquals(0L, stats.count)
        assertTrue(stats.mean.isNaN())
        assertTrue(stats.sum.isNaN())
        assertTrue(stats.min.isNaN())
        assertTrue(stats.max.isNaN())
        assertTrue(stats.variance().isNaN())
        assertTrue(stats.standardDeviation().isNaN())
        assertTrue(stats.skewness.isNaN())
        assertTrue(stats.kurtosis.isNaN())
    }

    @Test
    fun testSingleElement() {
        val stats = OnlineStatistics()
        stats.add(42.0)

        assertEquals(1L, stats.count)
        assertEquals(42.0, stats.mean, 1e-15)
        assertEquals(42.0, stats.sum, 1e-15)
        assertEquals(42.0, stats.min, 1e-15)
        assertEquals(42.0, stats.max, 1e-15)
        assertTrue(stats.variance().isNaN(), "Sample variance with n=1 should be NaN")
        assertEquals(0.0, stats.variance(PopulationKind.POPULATION), 1e-15)
        assertTrue(stats.skewness.isNaN())
        assertTrue(stats.kurtosis.isNaN())
    }

    @Test
    fun testTwoElements() {
        val stats = OnlineStatistics()
        stats.add(1.0)
        stats.add(3.0)

        assertEquals(2L, stats.count)
        assertEquals(2.0, stats.mean, 1e-15)
        assertEquals(2.0, stats.variance(), 1e-10) // sample variance
        assertTrue(stats.skewness.isNaN(), "Skewness with n=2 should be NaN")
        assertTrue(stats.kurtosis.isNaN())
    }

    @Test
    fun testThreeElements() {
        val stats = OnlineStatistics()
        stats.addAll(doubleArrayOf(1.0, 2.0, 3.0))

        assertEquals(3L, stats.count)
        assertTrue(stats.skewness.isFinite(), "Skewness with n=3 should be finite")
        assertTrue(stats.kurtosis.isNaN(), "Kurtosis with n=3 should be NaN")
    }

    @Test
    fun testFourElements() {
        val stats = OnlineStatistics()
        stats.addAll(doubleArrayOf(1.0, 2.0, 3.0, 4.0))

        assertEquals(4L, stats.count)
        assertTrue(stats.skewness.isFinite())
        assertTrue(stats.kurtosis.isFinite(), "Kurtosis with n=4 should be finite")
    }

    @Test
    fun testClearAndReuse() {
        val stats = OnlineStatistics()
        stats.addAll(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0))
        stats.clear()

        assertEquals(0L, stats.count)
        assertTrue(stats.mean.isNaN())
        assertTrue(stats.min.isNaN())
        assertTrue(stats.max.isNaN())

        // Reuse after clear
        stats.addAll(doubleArrayOf(10.0, 20.0, 30.0))
        assertEquals(3L, stats.count)
        assertEquals(20.0, stats.mean, 1e-10)
        assertEquals(10.0, stats.min, 1e-10)
        assertEquals(30.0, stats.max, 1e-10)
    }

    // ── Degenerate data ────────────────────────────────────────────────────

    @Test
    fun testConstantData() {
        val stats = OnlineStatistics()
        stats.addAll(doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0))

        assertEquals(5.0, stats.mean, 1e-15)
        assertEquals(0.0, stats.variance(), 1e-15)
        assertEquals(0.0, stats.variance(PopulationKind.POPULATION), 1e-15)
        assertEquals(0.0, stats.standardDeviation(), 1e-15)
        assertEquals(0.0, stats.skewness, 1e-15)
        assertEquals(-3.0, stats.kurtosis, 1e-15)
    }

    @Test
    fun testTwoIdenticalValues() {
        val stats = OnlineStatistics()
        stats.addAll(doubleArrayOf(7.0, 7.0))

        assertEquals(7.0, stats.mean, 1e-15)
        assertEquals(0.0, stats.variance(), 1e-15)
        assertEquals(0.0, stats.variance(PopulationKind.POPULATION), 1e-15)
    }

    // ── Numerical stability / extreme values ───────────────────────────────

    @Test
    fun testLargeOffset() {
        // Welford/Terriberry handles large offsets — classic two-pass would lose precision
        val offset = 1e15
        val data = doubleArrayOf(offset + 1, offset + 2, offset + 3, offset + 4, offset + 5)
        val stats = OnlineStatistics()
        stats.addAll(data)

        assertEquals(offset + 3.0, stats.mean, 1e-5)
        assertEquals(2.5, stats.variance(), 1e-5) // sample variance of {1,2,3,4,5}
        assertEquals(0.0, stats.skewness, 1e-5)
    }

    @Test
    fun testLargeMagnitude() {
        // M4 accumulates ~x^4, so magnitudes above ~1e75 will overflow for kurtosis.
        // Use 1e50 to stay within Double range for 4th-power terms.
        val data = doubleArrayOf(1e50, 2e50, 3e50, 4e50, 5e50, 6e50, 7e50)
        val stats = OnlineStatistics()
        stats.addAll(data)

        assertTrue(stats.mean.isFinite(), "Mean should be finite for large-magnitude data")
        assertTrue(stats.variance().isFinite(), "Variance should be finite for large-magnitude data")
        assertTrue(stats.skewness.isFinite(), "Skewness should be finite for large-magnitude data")
        assertTrue(stats.kurtosis.isFinite(), "Kurtosis should be finite for large-magnitude data")
    }

    @Test
    fun testLargeSampleNormalDistribution() {
        // Generate 100K samples from approximate standard normal using Box-Muller
        val n = 100_000
        val rng = kotlin.random.Random(42)
        val data = DoubleArray(n) {
            // Box-Muller transform
            val u1 = rng.nextDouble()
            val u2 = rng.nextDouble()
            sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
        }

        val stats = OnlineStatistics()
        stats.addAll(data)

        assertEquals(n.toLong(), stats.count)
        // For standard normal: mean ≈ 0, variance ≈ 1, skewness ≈ 0, kurtosis ≈ 0
        assertEquals(0.0, stats.mean, 0.02)
        assertEquals(1.0, stats.variance(), 0.02)
        assertEquals(0.0, stats.skewness, 0.1)
        assertEquals(0.0, stats.kurtosis, 0.1)
    }

    // ── Non-finite values ──────────────────────────────────────────────────

    @Test
    fun testNaNPropagation() {
        val stats = OnlineStatistics()
        stats.add(1.0)
        stats.add(Double.NaN)
        stats.add(3.0)

        assertTrue(stats.mean.isNaN(), "Mean should be NaN after NaN input")
        assertTrue(stats.variance().isNaN(), "Variance should be NaN after NaN input")
    }

    @Test
    fun testInfinityHandling() {
        val stats = OnlineStatistics()
        stats.add(1.0)
        stats.add(Double.POSITIVE_INFINITY)

        assertEquals(Double.POSITIVE_INFINITY, stats.max)
        assertEquals(1.0, stats.min, 1e-15)
    }

    @Test
    fun testMinMaxWithNaN() {
        // kotlin.math.min/max propagate NaN per IEEE 754
        val stats = OnlineStatistics()
        stats.add(1.0)
        stats.add(Double.NaN)
        stats.add(3.0)

        assertTrue(stats.min.isNaN(), "min should be NaN when NaN is in input")
        assertTrue(stats.max.isNaN(), "max should be NaN when NaN is in input")
    }

    @Test
    fun testNegativeValues() {
        val stats = OnlineStatistics()
        stats.addAll(doubleArrayOf(-5.0, -3.0, -1.0, 1.0, 3.0, 5.0))

        assertEquals(0.0, stats.mean, 1e-10)
        assertEquals(-5.0, stats.min, 1e-15)
        assertEquals(5.0, stats.max, 1e-15)
        assertEquals(0.0, stats.skewness, 1e-10)
    }

    // ── Skewed distribution reference ──────────────────────────────────────

    @Test
    fun testRightSkewedData() {
        val data = doubleArrayOf(1.0, 1.0, 1.0, 2.0, 5.0, 10.0)
        val stats = OnlineStatistics()
        stats.addAll(data)

        assertTrue(stats.skewness > 0, "Right-skewed data should have positive skewness")
    }

    @Test
    fun testLeftSkewedData() {
        val data = doubleArrayOf(1.0, 6.0, 10.0, 10.0, 10.0, 10.0)
        val stats = OnlineStatistics()
        stats.addAll(data)

        assertTrue(stats.skewness < 0, "Left-skewed data should have negative skewness")
    }
}
