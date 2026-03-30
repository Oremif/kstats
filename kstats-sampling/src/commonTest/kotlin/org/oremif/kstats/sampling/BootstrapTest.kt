package org.oremif.kstats.sampling

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BootstrapTest {

    private val tol = 1e-10

    // ── Basic correctness: known properties of bootstrap CIs ──────────────

    @Test
    fun testBootstrapCIMeanKnownDataset() {
        // Use a well-known dataset and verify structural properties of the result
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        // Observed statistic should be the mean of the data
        assertEquals(5.0, result.observedStatistic, tol)
        assertEquals(10_000, result.nResamples)
        assertEquals(0.95, result.confidenceLevel, tol)

        // All CI lower bounds should be < observed < upper bounds (for this data)
        assertTrue(result.percentile.lower < result.observedStatistic, "percentile lower < observed")
        assertTrue(result.percentile.upper > result.observedStatistic, "percentile upper > observed")
        assertTrue(result.basic.lower < result.observedStatistic, "basic lower < observed")
        assertTrue(result.basic.upper > result.observedStatistic, "basic upper > observed")
        assertTrue(result.bca.lower < result.observedStatistic, "BCa lower < observed")
        assertTrue(result.bca.upper > result.observedStatistic, "BCa upper > observed")
    }

    @Test
    fun testBootstrapCIMedianKnownDataset() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { arr ->
                val sorted = arr.sorted()
                val n = sorted.size
                if (n % 2 == 0) (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 else sorted[n / 2]
            },
        )

        // Observed statistic should be the median = 5.5
        assertEquals(5.5, result.observedStatistic, tol)
    }

    @Test
    fun testBootstrapCIListOverloadMean() {
        val data = listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertEquals(5.0, result.observedStatistic, tol)
        assertEquals(10_000, result.nResamples)
        assertEquals(0.95, result.confidenceLevel, tol)

        assertTrue(result.percentile.lower < result.observedStatistic, "percentile lower < observed")
        assertTrue(result.percentile.upper > result.observedStatistic, "percentile upper > observed")
    }

    @Test
    fun testBootstrapCIConstantData() {
        // For constant data, all CIs should collapse to [c, c]
        val data = doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 1_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertEquals(5.0, result.observedStatistic, tol)
        assertEquals(5.0, result.percentile.lower, tol)
        assertEquals(5.0, result.percentile.upper, tol)
        assertEquals(5.0, result.basic.lower, tol)
        assertEquals(5.0, result.basic.upper, tol)
        assertEquals(5.0, result.bca.lower, tol)
        assertEquals(5.0, result.bca.upper, tol)
    }

    @Test
    fun testBootstrapCIListOverloadConstantData() {
        val data = listOf(5.0, 5.0, 5.0, 5.0, 5.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 1_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertEquals(5.0, result.observedStatistic, tol)
        assertEquals(5.0, result.percentile.lower, tol)
        assertEquals(5.0, result.percentile.upper, tol)
    }

    // ── Edge cases ────────────────────────────────────────────────────────

    @Test
    fun testBootstrapCISingleElement() {
        // Single element: all resamples are the same value
        val data = doubleArrayOf(7.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 1_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertEquals(7.0, result.observedStatistic, tol)
        assertEquals(7.0, result.percentile.lower, tol)
        assertEquals(7.0, result.percentile.upper, tol)
        assertEquals(7.0, result.basic.lower, tol)
        assertEquals(7.0, result.basic.upper, tol)
    }

    @Test
    fun testBootstrapCITwoElements() {
        // Two elements [a, b]: bootstrap means can only be a, (a+b)/2, b
        val data = doubleArrayOf(1.0, 3.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertEquals(2.0, result.observedStatistic, tol)
        // The CI must be within the range of possible bootstrap means [1, 3]
        assertTrue(result.percentile.lower >= 1.0 - tol, "lower >= 1.0")
        assertTrue(result.percentile.upper <= 3.0 + tol, "upper <= 3.0")
    }

    @Test
    fun testBootstrapCIMinimumResamples() {
        // nResamples = 1: degenerate but valid
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 1,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertEquals(1, result.nResamples)
        assertTrue(result.observedStatistic.isFinite())
        // With only 1 resample, percentile lower = upper = that single bootstrap stat
        assertEquals(result.percentile.lower, result.percentile.upper, tol)
    }

    @Test
    fun testBootstrapCIConfidenceLevelZeroThrows() {
        // confidenceLevel = 0.0 is excluded: 0% CI is degenerate
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertFailsWith<InvalidParameterException> {
            bootstrapCI(
                data = data,
                nResamples = 1_000,
                confidenceLevel = 0.0,
                random = Random(42),
                statistic = { it.average() },
            )
        }
    }

    @Test
    fun testBootstrapCIConfidenceLevelOneThrows() {
        // confidenceLevel = 1.0 is excluded: 100% CI is degenerate
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertFailsWith<InvalidParameterException> {
            bootstrapCI(
                data = data,
                nResamples = 1_000,
                confidenceLevel = 1.0,
                random = Random(42),
                statistic = { it.average() },
            )
        }
    }

    @Test
    fun testBootstrapCIListSingleElement() {
        val data = listOf(7.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 1_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertEquals(7.0, result.observedStatistic, tol)
    }

    // ── Degenerate input ──────────────────────────────────────────────────

    @Test
    fun testBootstrapCIEmptyArrayThrows() {
        assertFailsWith<InsufficientDataException> {
            bootstrapCI(
                data = doubleArrayOf(),
                statistic = { it.average() },
            )
        }
    }

    @Test
    fun testBootstrapCIEmptyListThrows() {
        assertFailsWith<InsufficientDataException> {
            bootstrapCI(
                data = emptyList<Double>(),
                statistic = { it.average() },
            )
        }
    }

    @Test
    fun testBootstrapCINResamplesZeroThrows() {
        assertFailsWith<InvalidParameterException> {
            bootstrapCI(
                data = doubleArrayOf(1.0, 2.0, 3.0),
                nResamples = 0,
                statistic = { it.average() },
            )
        }
    }

    @Test
    fun testBootstrapCINResamplesNegativeThrows() {
        assertFailsWith<InvalidParameterException> {
            bootstrapCI(
                data = doubleArrayOf(1.0, 2.0, 3.0),
                nResamples = -5,
                statistic = { it.average() },
            )
        }
    }

    @Test
    fun testBootstrapCIConfidenceLevelBelowRangeThrows() {
        assertFailsWith<InvalidParameterException> {
            bootstrapCI(
                data = doubleArrayOf(1.0, 2.0, 3.0),
                confidenceLevel = -0.01,
                statistic = { it.average() },
            )
        }
    }

    @Test
    fun testBootstrapCIConfidenceLevelAboveRangeThrows() {
        assertFailsWith<InvalidParameterException> {
            bootstrapCI(
                data = doubleArrayOf(1.0, 2.0, 3.0),
                confidenceLevel = 1.01,
                statistic = { it.average() },
            )
        }
    }

    @Test
    fun testBootstrapCIListNResamplesZeroThrows() {
        assertFailsWith<InvalidParameterException> {
            bootstrapCI(
                data = listOf(1.0, 2.0, 3.0),
                nResamples = 0,
                statistic = { it.average() },
            )
        }
    }

    @Test
    fun testBootstrapCIListConfidenceLevelInvalidThrows() {
        assertFailsWith<InvalidParameterException> {
            bootstrapCI(
                data = listOf(1.0, 2.0, 3.0),
                confidenceLevel = 1.5,
                statistic = { it.average() },
            )
        }
    }

    // ── Extreme parameters ────────────────────────────────────────────────

    @Test
    fun testBootstrapCILargeValues() {
        // Numerical stability with large-offset data
        val data = DoubleArray(50) { 1e12 + it.toDouble() }
        val result = bootstrapCI(
            data = data,
            nResamples = 1_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        val expectedMean = 1e12 + 24.5
        assertEquals(expectedMean, result.observedStatistic, 1e-4)
        assertTrue(result.percentile.lower.isFinite(), "lower should be finite for large values")
        assertTrue(result.percentile.upper.isFinite(), "upper should be finite for large values")
        assertTrue(result.bca.lower.isFinite(), "BCa lower should be finite for large values")
        assertTrue(result.bca.upper.isFinite(), "BCa upper should be finite for large values")
    }

    @Test
    fun testBootstrapCISmallValues() {
        // Numerical stability with very small values
        val data = doubleArrayOf(1e-15, 2e-15, 3e-15, 4e-15, 5e-15)
        val result = bootstrapCI(
            data = data,
            nResamples = 1_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertEquals(3e-15, result.observedStatistic, 1e-25)
        assertTrue(result.percentile.lower.isFinite(), "lower should be finite for small values")
        assertTrue(result.percentile.upper.isFinite(), "upper should be finite for small values")
    }

    @Test
    fun testBootstrapCILargeNResamples() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 50_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertEquals(3.0, result.observedStatistic, tol)
        assertEquals(50_000, result.nResamples)
        assertTrue(result.percentile.lower.isFinite())
        assertTrue(result.percentile.upper.isFinite())
    }

    @Test
    fun testBootstrapCILargeDataset() {
        // Large dataset should produce narrow CI
        val data = DoubleArray(1000) { it.toDouble() }
        val result = bootstrapCI(
            data = data,
            nResamples = 5_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        val expectedMean = 499.5
        assertEquals(expectedMean, result.observedStatistic, tol)
        // CI width should be much narrower than the range of data (0..999)
        val ciWidth = result.percentile.upper - result.percentile.lower
        assertTrue(ciWidth < 100.0, "CI width $ciWidth should be much less than data range 999")
        assertTrue(ciWidth > 0.0, "CI width should be positive")
    }

    @Test
    fun testBootstrapCIHighlySkewedData() {
        // Highly skewed data: BCa should adjust for skewness
        val data = doubleArrayOf(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertTrue(result.observedStatistic.isFinite())
        assertTrue(result.percentile.lower.isFinite())
        assertTrue(result.percentile.upper.isFinite())
        assertTrue(result.bca.lower.isFinite())
        assertTrue(result.bca.upper.isFinite())
    }

    // ── Non-finite input ──────────────────────────────────────────────────

    @Test
    fun testBootstrapCINaNInData() {
        // NaN in data should propagate through the statistic
        val data = doubleArrayOf(1.0, Double.NaN, 3.0, 4.0, 5.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 100,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        // The observed statistic is NaN since average of data with NaN is NaN
        assertTrue(result.observedStatistic.isNaN(), "observed should be NaN when data contains NaN")
    }

    @Test
    fun testBootstrapCIInfinityInData() {
        val data = doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0, 4.0, 5.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 100,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        // The observed statistic should be Infinity since average includes +Inf
        assertEquals(Double.POSITIVE_INFINITY, result.observedStatistic)
    }

    @Test
    fun testBootstrapCINaNConfidenceLevelThrows() {
        // NaN for confidenceLevel: NaN !in 0.0..1.0 is true (NaN comparisons are false)
        // so it should throw InvalidParameterException
        assertFailsWith<InvalidParameterException> {
            bootstrapCI(
                data = doubleArrayOf(1.0, 2.0, 3.0),
                confidenceLevel = Double.NaN,
                statistic = { it.average() },
            )
        }
    }

    @Test
    fun testBootstrapCIListNaNInData() {
        val data = listOf(1.0, Double.NaN, 3.0, 4.0, 5.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 100,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertTrue(result.observedStatistic.isNaN(), "observed should be NaN when data contains NaN")
    }

    // ── Property-based ────────────────────────────────────────────────────

    @Test
    fun testPercentileCIContainsObserved() {
        // For typical data with sufficient resamples, the percentile CI should
        // contain the observed statistic (at 95% confidence level)
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertTrue(
            result.observedStatistic in result.percentile.lower..result.percentile.upper,
            "Observed ${result.observedStatistic} should be in " +
                "[${result.percentile.lower}, ${result.percentile.upper}]"
        )
    }

    @Test
    fun testBasicCIContainsObserved() {
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertTrue(
            result.observedStatistic in result.basic.lower..result.basic.upper,
            "Observed ${result.observedStatistic} should be in " +
                "[${result.basic.lower}, ${result.basic.upper}]"
        )
    }

    @Test
    fun testBcaCIContainsObserved() {
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertTrue(
            result.observedStatistic in result.bca.lower..result.bca.upper,
            "Observed ${result.observedStatistic} should be in " +
                "[${result.bca.lower}, ${result.bca.upper}]"
        )
    }

    @Test
    fun testCILowerLessThanOrEqualUpper() {
        // For all three CI methods, lower <= upper
        val data = doubleArrayOf(1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 5_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertTrue(result.percentile.lower <= result.percentile.upper,
            "percentile: lower <= upper")
        assertTrue(result.basic.lower <= result.basic.upper,
            "basic: lower <= upper")
        assertTrue(result.bca.lower <= result.bca.upper,
            "BCa: lower <= upper")
    }

    @Test
    fun testHigherConfidenceWiderCI() {
        // A higher confidence level should produce a wider (or equal) CI
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)

        val result90 = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.90,
            random = Random(42),
            statistic = { it.average() },
        )
        val result95 = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )
        val result99 = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.99,
            random = Random(42),
            statistic = { it.average() },
        )

        val width90 = result90.percentile.upper - result90.percentile.lower
        val width95 = result95.percentile.upper - result95.percentile.lower
        val width99 = result99.percentile.upper - result99.percentile.lower

        assertTrue(width90 <= width95 + tol, "90% width ($width90) <= 95% width ($width95)")
        assertTrue(width95 <= width99 + tol, "95% width ($width95) <= 99% width ($width99)")
    }

    @Test
    fun testDeterministicWithSeed() {
        // Same seed produces same results
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)

        val result1 = bootstrapCI(
            data = data,
            nResamples = 5_000,
            confidenceLevel = 0.95,
            random = Random(123),
            statistic = { it.average() },
        )
        val result2 = bootstrapCI(
            data = data,
            nResamples = 5_000,
            confidenceLevel = 0.95,
            random = Random(123),
            statistic = { it.average() },
        )

        assertEquals(result1.observedStatistic, result2.observedStatistic, tol)
        assertEquals(result1.percentile.lower, result2.percentile.lower, tol)
        assertEquals(result1.percentile.upper, result2.percentile.upper, tol)
        assertEquals(result1.basic.lower, result2.basic.lower, tol)
        assertEquals(result1.basic.upper, result2.basic.upper, tol)
        assertEquals(result1.bca.lower, result2.bca.lower, tol)
        assertEquals(result1.bca.upper, result2.bca.upper, tol)
    }

    @Test
    fun testListAndDoubleArrayConsistency() {
        // Both overloads should produce the same result when given the same seed
        val dataArray = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val dataList = dataArray.toList()

        val resultArray = bootstrapCI(
            data = dataArray,
            nResamples = 5_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )
        val resultList = bootstrapCI(
            data = dataList,
            nResamples = 5_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertEquals(resultArray.observedStatistic, resultList.observedStatistic, tol)
        assertEquals(resultArray.percentile.lower, resultList.percentile.lower, tol)
        assertEquals(resultArray.percentile.upper, resultList.percentile.upper, tol)
        assertEquals(resultArray.basic.lower, resultList.basic.lower, tol)
        assertEquals(resultArray.basic.upper, resultList.basic.upper, tol)
        assertEquals(resultArray.bca.lower, resultList.bca.lower, tol)
        assertEquals(resultArray.bca.upper, resultList.bca.upper, tol)
    }

    @Test
    fun testSymmetricDataBcaApproxPercentile() {
        // For perfectly symmetric data, BCa acceleration factor a_hat = 0
        // and if bias is also ~0, BCa should approximate the percentile CI
        // scipy: symmetric data [1,2,3,4,5] has a_hat = 0
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 50_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        // BCa and percentile should be close for symmetric data
        val diff = abs(result.bca.lower - result.percentile.lower) +
            abs(result.bca.upper - result.percentile.upper)
        assertTrue(diff < 1.0, "BCa and percentile should be close for symmetric data, diff=$diff")
    }

    @Test
    fun testBasicCIFormula() {
        // Verify basic (pivotal) CI formula: 2*observed - percentile(1-alpha/2), 2*observed - percentile(alpha/2)
        // i.e., basic.lower = 2*observed - percentile.upper
        //       basic.upper = 2*observed - percentile.lower
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val result = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        assertEquals(
            2.0 * result.observedStatistic - result.percentile.upper,
            result.basic.lower,
            tol,
            "basic.lower = 2*observed - percentile.upper"
        )
        assertEquals(
            2.0 * result.observedStatistic - result.percentile.lower,
            result.basic.upper,
            tol,
            "basic.upper = 2*observed - percentile.lower"
        )
    }

    @Test
    fun testInputArrayNotMutated() {
        val data = doubleArrayOf(5.0, 3.0, 1.0, 4.0, 2.0)
        val copy = data.copyOf()
        bootstrapCI(
            data = data,
            nResamples = 1_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { it.average() },
        )

        for (i in data.indices) {
            assertEquals(copy[i], data[i], 0.0, "input data[$i] should not be mutated")
        }
    }

    @Test
    fun testCICoversPopulationMean() {
        // For a known population, the bootstrap CI should cover the true mean
        // Generate sample from a pseudo-normal distribution using fixed data
        val data = doubleArrayOf(
            3.1, 4.7, 5.2, 4.8, 5.5, 6.1, 4.3, 5.0, 5.8, 4.9,
            5.3, 4.6, 5.7, 5.1, 4.4, 5.9, 4.2, 5.4, 4.5, 5.6,
        )
        val trueMean = data.average()

        val result = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.99,
            random = Random(42),
            statistic = { it.average() },
        )

        // The 99% CI should contain the observed mean (which is the true sample mean here)
        assertTrue(
            trueMean in result.percentile.lower..result.percentile.upper,
            "99% percentile CI should contain the sample mean"
        )
    }

    @Test
    fun testCustomStatisticStandardDeviation() {
        // Verify bootstrapCI works with a custom statistic (standard deviation)
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val observedStd = sqrt(
            data.map { (it - data.average()) * (it - data.average()) }.sum() / (data.size - 1)
        )

        val result = bootstrapCI(
            data = data,
            nResamples = 10_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { arr ->
                val mean = arr.average()
                sqrt(arr.map { (it - mean) * (it - mean) }.sum() / (arr.size - 1))
            },
        )

        assertEquals(observedStd, result.observedStatistic, 1e-8)
        assertTrue(result.percentile.lower > 0.0, "std CI lower > 0")
        assertTrue(result.percentile.upper > result.percentile.lower, "std CI upper > lower")
    }

    @Test
    fun testBootstrapCIWithNonDoubleList() {
        // List<T> overload works with non-Double types
        data class Measurement(val value: Double, val weight: Double)

        val data = listOf(
            Measurement(1.0, 0.5),
            Measurement(2.0, 1.0),
            Measurement(3.0, 1.5),
            Measurement(4.0, 2.0),
            Measurement(5.0, 2.5),
        )

        val result = bootstrapCI(
            data = data,
            nResamples = 1_000,
            confidenceLevel = 0.95,
            random = Random(42),
            statistic = { sample ->
                val totalWeight = sample.sumOf { it.weight }
                sample.sumOf { it.value * it.weight } / totalWeight
            },
        )

        assertTrue(result.observedStatistic.isFinite())
        assertTrue(result.percentile.lower.isFinite())
        assertTrue(result.percentile.upper.isFinite())
    }
}
