package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class CentralTendencyTest {

    @Test
    fun testMean() {
        assertEquals(3.0, listOf(1.0, 2.0, 3.0, 4.0, 5.0).mean(), 1e-10)
        assertEquals(3.0, doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0).mean(), 1e-10)
    }

    @Test
    fun testMeanEmpty() {
        assertFailsWith<InsufficientDataException> { emptyList<Double>().mean() }
        assertFailsWith<InsufficientDataException> { doubleArrayOf().mean() }
    }

    @Test
    fun testGeometricMean() {
        // geometric mean of 1,2,4,8 = (1*2*4*8)^(1/4) = 64^0.25 = 2√2 ≈ 2.8284
        val expected = 2.82842712474619
        assertEquals(expected, listOf(1.0, 2.0, 4.0, 8.0).geometricMean(), 1e-10)
    }

    @Test
    fun testHarmonicMean() {
        // harmonic mean of 1,2,4 = 3 / (1 + 0.5 + 0.25) = 3/1.75 ≈ 1.7143
        val expected = 3.0 / 1.75
        assertEquals(expected, listOf(1.0, 2.0, 4.0).harmonicMean(), 1e-10)
    }

    @Test
    fun testWeightedMean() {
        val values = listOf(1.0, 2.0, 3.0)
        val weights = listOf(3.0, 1.0, 1.0)
        // (1*3 + 2*1 + 3*1) / (3+1+1) = 8/5 = 1.6
        assertEquals(1.6, values.weightedMean(weights), 1e-10)
    }

    @Test
    fun testMedianOdd() {
        assertEquals(3.0, listOf(5.0, 1.0, 3.0, 2.0, 4.0).median(), 1e-10)
    }

    @Test
    fun testMedianEven() {
        assertEquals(2.5, listOf(1.0, 2.0, 3.0, 4.0).median(), 1e-10)
    }

    @Test
    fun testMode() {
        assertEquals(setOf(3.0), listOf(1.0, 2.0, 3.0, 3.0, 4.0).mode())
        // multiple modes
        assertEquals(setOf(1.0, 2.0), listOf(1.0, 1.0, 2.0, 2.0, 3.0).mode())
    }

    @Test
    fun testModeIntegers() {
        assertEquals(setOf(5), listOf(1, 5, 5, 3).mode())
    }

    @Test
    fun testMeanLargeOffset() {
        val data = DoubleArray(1000) { 1e15 + it.toDouble() }
        val expected = 1e15 + 499.5
        assertEquals(expected, data.mean(), 1e-6)
        assertEquals(expected, data.toList().mean(), 1e-6)
        assertEquals(expected, data.asSequence().mean(), 1e-6)
    }

    @Test
    fun testHarmonicMeanPrecision() {
        // Spread values enough so harmonic < arithmetic is clear,
        // but use large magnitudes to stress compensated summation
        val data = DoubleArray(100) { 1e8 + it.toDouble() * 1e6 }
        val listResult = data.toList().harmonicMean()
        val arrayResult = data.harmonicMean()
        // Both paths should agree to high relative precision
        assertEquals(listResult, arrayResult, arrayResult * 1e-10)
        // Harmonic mean must be less than arithmetic mean
        assertTrue(arrayResult < data.mean())
    }

    @Test
    fun testWeightedMeanPrecision() {
        val values = DoubleArray(100) { 1e14 + it.toDouble() }
        val weights = DoubleArray(100) { 1.0 }
        val expected = 1e14 + 49.5 // uniform weights → arithmetic mean
        assertEquals(expected, values.weightedMean(weights), 1e-4)
        assertEquals(expected, values.toList().weightedMean(weights.toList()), 1e-4)
    }
}

class DispersionTest {

    @Test
    fun testVarianceSample() {
        // R: var(c(2,4,4,4,5,5,7,9)) = 4.571429
        val data = listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        assertEquals(4.571428571428571, data.variance(), 1e-10)
    }

    @Test
    fun testVariancePopulation() {
        val data = listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        assertEquals(4.0, data.variance(PopulationKind.POPULATION), 1e-10)
    }

    @Test
    fun testStandardDeviation() {
        val data = listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        assertEquals(sqrt(4.571428571428571), data.standardDeviation(), 1e-10)
    }

    @Test
    fun testRange() {
        assertEquals(8.0, listOf(1.0, 3.0, 5.0, 9.0).range(), 1e-10)
    }

    @Test
    fun testInterquartileRange() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val iqr = data.interquartileRange()
        assertTrue(iqr > 0)
    }

    @Test
    fun testMeanAbsoluteDeviation() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        // mean = 3, deviations = |−2|+|−1|+0+1+2 = 6, MAD = 6/5 = 1.2
        assertEquals(1.2, data.meanAbsoluteDeviation(), 1e-10)
    }

    @Test
    fun testStandardError() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val se = data.standardError()
        assertEquals(data.standardDeviation() / sqrt(5.0), se, 1e-10)
    }

    @Test
    fun testCoefficientOfVariation() {
        val data = listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val cv = data.coefficientOfVariation()
        assertEquals(data.standardDeviation() / data.mean(), cv, 1e-10)
    }
}

class ShapeTest {

    @Test
    fun testSkewnessSymmetric() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(0.0, data.skewness(), 1e-10)
    }

    @Test
    fun testSkewnessRightSkewed() {
        val data = listOf(1.0, 1.0, 1.0, 2.0, 5.0, 10.0)
        assertTrue(data.skewness() > 0)
    }

    @Test
    fun testKurtosisNormal() {
        // For a normal-like symmetric distribution, excess kurtosis should be near 0
        // But small samples won't be exactly 0
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        // Just verify it returns a finite number
        assertTrue(data.kurtosis().isFinite())
    }

    @Test
    fun testSkewnessNoOverflow() {
        val data = doubleArrayOf(1e154, 1.5e154, 2e154, 2.5e154, 3e154, 3.5e154, 4e154)
        val result = data.skewness()
        assertTrue(result.isFinite(), "Skewness must be finite for large-magnitude data, got $result")
    }

    @Test
    fun testKurtosisNoOverflow() {
        val data = doubleArrayOf(1e100, 2e100, 3e100, 4e100, 5e100, 6e100, 7e100)
        val result = data.kurtosis()
        assertTrue(result.isFinite(), "Kurtosis must be finite for large-magnitude data, got $result")
    }

    @Test
    fun testSkewnessLargeMagnitude() {
        // Symmetric data at large scale — skewness must be 0
        val data = doubleArrayOf(1e154, 2e154, 3e154, 4e154, 5e154)
        assertEquals(0.0, data.skewness(), 1e-10)
    }

    @Test
    fun testKurtosisConstant() {
        val data = doubleArrayOf(5.0, 5.0, 5.0, 5.0)
        assertEquals(-3.0, data.kurtosis(excess = true), 1e-10)
        assertEquals(0.0, data.kurtosis(excess = false), 1e-10)
    }

    @Test
    fun testSkewnessPopulation() {
        // Symmetric data — population skewness must be 0
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(0.0, data.skewness(PopulationKind.POPULATION), 1e-10)
    }

    @Test
    fun testKurtosisPopulationExcessRelation() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        val excess = data.kurtosis(PopulationKind.POPULATION, excess = true)
        val nonExcess = data.kurtosis(PopulationKind.POPULATION, excess = false)
        assertEquals(nonExcess, excess + 3.0, 1e-10)
    }

    @Test
    fun testSkewnessDoubleArrayConsistency() {
        val array = doubleArrayOf(1.0, 3.0, 5.0, 2.0, 8.0, 4.0)
        val list = array.toList()
        assertEquals(list.skewness(), array.skewness(), 1e-15)
        assertEquals(list.skewness(PopulationKind.POPULATION), array.skewness(PopulationKind.POPULATION), 1e-15)
    }

    @Test
    fun testKurtosisDoubleArrayConsistency() {
        val array = doubleArrayOf(1.0, 3.0, 5.0, 2.0, 8.0, 4.0)
        val list = array.toList()
        assertEquals(list.kurtosis(), array.kurtosis(), 1e-15)
        assertEquals(list.kurtosis(PopulationKind.POPULATION), array.kurtosis(PopulationKind.POPULATION), 1e-15)
    }
}

class QuantilesTest {

    @Test
    fun testPercentileMedian() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(3.0, data.percentile(50.0), 1e-10)
    }

    @Test
    fun testPercentileInterpolation() {
        val data = listOf(1.0, 2.0, 3.0, 4.0)
        // 25th percentile with linear interpolation
        assertEquals(1.75, data.percentile(25.0), 1e-10)
    }

    @Test
    fun testQuantileBounds() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(1.0, data.quantile(0.0), 1e-10)
        assertEquals(5.0, data.quantile(1.0), 1e-10)
    }

    @Test
    fun testQuartiles() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val (q1, q2, q3) = data.quartiles()
        assertEquals(q2, data.median(), 1e-10)
        assertTrue(q1 < q2)
        assertTrue(q2 < q3)
    }

    @Test
    fun testPercentileLower() {
        val data = listOf(1.0, 2.0, 3.0, 4.0)
        assertEquals(1.0, data.percentile(25.0, QuantileInterpolation.LOWER), 1e-10)
    }

    @Test
    fun testPercentileHigher() {
        val data = listOf(1.0, 2.0, 3.0, 4.0)
        assertEquals(2.0, data.percentile(25.0, QuantileInterpolation.HIGHER), 1e-10)
    }
}

class SummaryStatisticsTest {

    @Test
    fun testDescribe() {
        val data = listOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val stats = data.describe()
        assertEquals(10, stats.count)
        assertEquals(5.5, stats.mean, 1e-10)
        assertEquals(55.0, stats.sum, 1e-10)
        assertEquals(1.0, stats.min, 1e-10)
        assertEquals(10.0, stats.max, 1e-10)
        assertEquals(9.0, stats.range, 1e-10)
    }

    @Test
    fun testDescribeEmpty() {
        assertFailsWith<InsufficientDataException> { emptyList<Double>().describe() }
        assertFailsWith<InsufficientDataException> { doubleArrayOf().describe() }
    }

    @Test
    fun testDescribeN1() {
        val stats = listOf(42.0).describe()
        assertEquals(1, stats.count)
        assertEquals(42.0, stats.mean, 1e-10)
        assertEquals(42.0, stats.sum, 1e-10)
        assertEquals(42.0, stats.min, 1e-10)
        assertEquals(42.0, stats.max, 1e-10)
        assertEquals(0.0, stats.range, 1e-10)
        assertEquals(42.0, stats.q1, 1e-10)
        assertEquals(42.0, stats.median, 1e-10)
        assertEquals(42.0, stats.q3, 1e-10)
        assertEquals(0.0, stats.interquartileRange, 1e-10)
        assertTrue(stats.variance.isNaN())
        assertTrue(stats.standardDeviation.isNaN())
        assertTrue(stats.standardError.isNaN())
        assertTrue(stats.skewness.isNaN())
        assertTrue(stats.kurtosis.isNaN())
    }

    @Test
    fun testDescribeN2() {
        val stats = listOf(1.0, 3.0).describe()
        assertEquals(2, stats.count)
        assertEquals(2.0, stats.mean, 1e-10)
        assertTrue(stats.variance.isFinite())
        assertTrue(stats.standardDeviation.isFinite())
        assertTrue(stats.standardError.isFinite())
        assertTrue(stats.skewness.isNaN())
        assertTrue(stats.kurtosis.isNaN())
    }

    @Test
    fun testDescribeN3() {
        val stats = listOf(1.0, 2.0, 3.0).describe()
        assertEquals(3, stats.count)
        assertTrue(stats.variance.isFinite())
        assertTrue(stats.standardDeviation.isFinite())
        assertTrue(stats.skewness.isFinite())
        assertTrue(stats.kurtosis.isNaN())
    }

    @Test
    fun testDescribeN4() {
        val stats = listOf(1.0, 2.0, 3.0, 4.0).describe()
        assertEquals(4, stats.count)
        assertTrue(stats.variance.isFinite())
        assertTrue(stats.standardDeviation.isFinite())
        assertTrue(stats.skewness.isFinite())
        assertTrue(stats.kurtosis.isFinite())
    }

    @Test
    fun testDescribeDoubleArraySmallN() {
        val stats = doubleArrayOf(7.0).describe()
        assertEquals(1, stats.count)
        assertEquals(7.0, stats.mean, 1e-10)
        assertTrue(stats.variance.isNaN())
        assertTrue(stats.skewness.isNaN())
        assertTrue(stats.kurtosis.isNaN())
    }
}

class NumberExtensionsTest {

    @Test
    fun testIntMean() {
        assertEquals(3.0, listOf(1, 2, 3, 4, 5).mean(), 1e-10)
    }

    @Test
    fun testIntMedian() {
        assertEquals(3.0, listOf(1, 2, 3, 4, 5).median(), 1e-10)
    }

    @Test
    fun testLongMean() {
        assertEquals(3.0, listOf(1L, 2L, 3L, 4L, 5L).mean(), 1e-10)
    }
}

class TrimmedStatisticsTest {

    private val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)

    // ── trimmedMean ─────────────────────────────────────────────────────────

    @Test
    fun testTrimmedMeanBasic() {
        // scipy: trim_mean([1..10], 0.1) = 5.5 (trim 1 from each end → [2..9])
        assertEquals(5.5, data.trimmedMean(0.1), 1e-10)
        // scipy: trim_mean([1..10], 0.2) = 5.5 (trim 2 from each end → [3..8])
        assertEquals(5.5, data.trimmedMean(0.2), 1e-10)
        // scipy: trim_mean([1..10], 0.3) = 5.5 (trim 3 from each end → [4..7])
        assertEquals(5.5, data.trimmedMean(0.3), 1e-10)
    }

    @Test
    fun testTrimmedMeanProportionZero() {
        // proportion=0.0 should equal regular mean
        assertEquals(data.mean(), data.trimmedMean(0.0), 1e-10)
    }

    @Test
    fun testTrimmedMeanSingleElement() {
        assertEquals(42.0, doubleArrayOf(42.0).trimmedMean(0.0), 1e-10)
    }

    @Test
    fun testTrimmedMeanLeavesOneElement() {
        // 3 elements, proportion=0.4 → k=floor(3*0.4)=1, trimmed=[middle], mean=middle
        val arr = doubleArrayOf(1.0, 5.0, 9.0)
        assertEquals(5.0, arr.trimmedMean(0.4), 1e-10)
    }

    @Test
    fun testTrimmedMeanEmptyThrows() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf().trimmedMean(0.1) }
    }

    @Test
    fun testTrimmedMeanInvalidProportionThrows() {
        assertFailsWith<InvalidParameterException> { data.trimmedMean(-0.1) }
        assertFailsWith<InvalidParameterException> { data.trimmedMean(0.5) }
        assertFailsWith<InvalidParameterException> { data.trimmedMean(1.0) }
        assertFailsWith<InvalidParameterException> { data.trimmedMean(Double.NaN) }
    }

    @Test
    fun testTrimmedMeanLargeOffset() {
        // Compensated sum precision with large offsets
        val largeData = DoubleArray(1000) { 1e15 + it.toDouble() }
        val expected = 1e15 + 499.5 // mean of [100..899]
        val result = largeData.trimmedMean(0.1)
        assertTrue(abs(result - expected) < 1e-2, "Expected $expected but got $result")
    }

    @Test
    fun testTrimmedMeanNaN() {
        val withNaN = doubleArrayOf(1.0, Double.NaN, 3.0, 4.0, 5.0)
        assertTrue(withNaN.trimmedMean(0.0).isNaN())
    }

    @Test
    fun testTrimmedMeanConstantArray() {
        val constant = DoubleArray(10) { 7.0 }
        assertEquals(7.0, constant.trimmedMean(0.2), 1e-10)
    }

    @Test
    fun testTrimmedMeanIterableAndSequence() {
        val list = data.toList()
        assertEquals(data.trimmedMean(0.2), list.trimmedMean(0.2), 1e-15)
        assertEquals(data.trimmedMean(0.2), data.asSequence().trimmedMean(0.2), 1e-15)
    }

    // ── trimmedVariance ─────────────────────────────────────────────────────

    @Test
    fun testTrimmedVarianceBasic() {
        // scipy reference: sorted [1..10], p=0.1 → [2..9], sample var = 6.0
        assertEquals(6.0, data.trimmedVariance(0.1), 1e-10)
        // p=0.2 → [3..8], sample var = 3.5
        assertEquals(3.5, data.trimmedVariance(0.2), 1e-10)
        // p=0.3 → [4..7], sample var = 5/3
        assertEquals(5.0 / 3.0, data.trimmedVariance(0.3), 1e-10)
    }

    @Test
    fun testTrimmedVariancePopulation() {
        // p=0.1 → [2..9], population var = 5.25
        assertEquals(5.25, data.trimmedVariance(0.1, PopulationKind.POPULATION), 1e-10)
        // p=0.2 → [3..8], population var = 35/12
        assertEquals(35.0 / 12.0, data.trimmedVariance(0.2, PopulationKind.POPULATION), 1e-10)
        // p=0.3 → [4..7], population var = 1.25
        assertEquals(1.25, data.trimmedVariance(0.3, PopulationKind.POPULATION), 1e-10)
    }

    @Test
    fun testTrimmedVarianceProportionZero() {
        // proportion=0.0 should equal regular variance
        assertEquals(data.toList().variance(), data.trimmedVariance(0.0), 1e-10)
        assertEquals(
            data.toList().variance(PopulationKind.POPULATION),
            data.trimmedVariance(0.0, PopulationKind.POPULATION),
            1e-10
        )
    }

    @Test
    fun testTrimmedVarianceEmptyThrows() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf().trimmedVariance(0.1) }
    }

    @Test
    fun testTrimmedVarianceInvalidProportionThrows() {
        assertFailsWith<InvalidParameterException> { data.trimmedVariance(-0.1) }
        assertFailsWith<InvalidParameterException> { data.trimmedVariance(0.5) }
        assertFailsWith<InvalidParameterException> { data.trimmedVariance(1.0) }
        assertFailsWith<InvalidParameterException> { data.trimmedVariance(Double.NaN) }
    }

    @Test
    fun testTrimmedVarianceSingleRemainingElementSampleThrows() {
        // 3 elements, proportion=0.4 → k=1, m=1: sample variance requires ≥2
        val arr = doubleArrayOf(1.0, 5.0, 9.0)
        assertFailsWith<InsufficientDataException> { arr.trimmedVariance(0.4, PopulationKind.SAMPLE) }
    }

    @Test
    fun testTrimmedVarianceSingleRemainingElementPopulation() {
        // Population variance of single element is 0
        val arr = doubleArrayOf(1.0, 5.0, 9.0)
        assertEquals(0.0, arr.trimmedVariance(0.4, PopulationKind.POPULATION), 1e-10)
    }

    @Test
    fun testTrimmedVarianceConstantArray() {
        val constant = DoubleArray(10) { 7.0 }
        assertEquals(0.0, constant.trimmedVariance(0.2), 1e-10)
    }

    @Test
    fun testTrimmedVarianceNaN() {
        val withNaN = doubleArrayOf(1.0, Double.NaN, 3.0, 4.0, 5.0)
        assertTrue(withNaN.trimmedVariance(0.0).isNaN())
    }

    @Test
    fun testTrimmedVarianceIterableAndSequence() {
        val list = data.toList()
        assertEquals(data.trimmedVariance(0.2), list.trimmedVariance(0.2), 1e-15)
        assertEquals(data.trimmedVariance(0.2), data.asSequence().trimmedVariance(0.2), 1e-15)
    }

    // ── trimmedStd ──────────────────────────────────────────────────────────

    @Test
    fun testTrimmedStdBasic() {
        // sqrt of trimmedVariance
        assertEquals(sqrt(6.0), data.trimmedStd(0.1), 1e-10)
        assertEquals(sqrt(3.5), data.trimmedStd(0.2), 1e-10)
        assertEquals(sqrt(5.0 / 3.0), data.trimmedStd(0.3), 1e-10)
    }

    @Test
    fun testTrimmedStdPopulation() {
        assertEquals(sqrt(5.25), data.trimmedStd(0.1, PopulationKind.POPULATION), 1e-10)
        assertEquals(sqrt(35.0 / 12.0), data.trimmedStd(0.2, PopulationKind.POPULATION), 1e-10)
        assertEquals(sqrt(1.25), data.trimmedStd(0.3, PopulationKind.POPULATION), 1e-10)
    }

    @Test
    fun testTrimmedStdProportionZero() {
        assertEquals(data.toList().standardDeviation(), data.trimmedStd(0.0), 1e-10)
    }

    @Test
    fun testTrimmedStdEmptyThrows() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf().trimmedStd(0.1) }
    }

    @Test
    fun testTrimmedStdInvalidProportionThrows() {
        assertFailsWith<InvalidParameterException> { data.trimmedStd(-0.1) }
        assertFailsWith<InvalidParameterException> { data.trimmedStd(0.5) }
        assertFailsWith<InvalidParameterException> { data.trimmedStd(Double.NaN) }
    }

    @Test
    fun testTrimmedStdIterableAndSequence() {
        val list = data.toList()
        assertEquals(data.trimmedStd(0.2), list.trimmedStd(0.2), 1e-15)
        assertEquals(data.trimmedStd(0.2), data.asSequence().trimmedStd(0.2), 1e-15)
    }

    @Test
    fun testTrimmedStdConsistentWithVariance() {
        // Verify sqrt(trimmedVariance) == trimmedStd for various proportions
        for (p in listOf(0.0, 0.1, 0.2, 0.3, 0.4)) {
            val v = data.trimmedVariance(p)
            val s = data.trimmedStd(p)
            assertEquals(sqrt(v), s, 1e-15, "trimmedStd($p) should equal sqrt(trimmedVariance($p))")
        }
    }
}

class SemiVarianceTest {

    private val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
    // mean = 5.0
    // Below mean: diffs = -3,-1,-1,-1 → sum of squares = 12
    // Above mean: diffs = +2,+4 → sum of squares = 20

    // ── basic correctness ───────────────────────────────────────────────────

    @Test
    fun testDownsideSampleVariance() {
        assertEquals(12.0 / 7.0, data.semiVariance(direction = SemiVarianceDirection.DOWNSIDE), 1e-10)
    }

    @Test
    fun testUpsideSampleVariance() {
        assertEquals(20.0 / 7.0, data.semiVariance(direction = SemiVarianceDirection.UPSIDE), 1e-10)
    }

    @Test
    fun testDownsidePopulationVariance() {
        assertEquals(
            12.0 / 8.0,
            data.semiVariance(direction = SemiVarianceDirection.DOWNSIDE, kind = PopulationKind.POPULATION),
            1e-10
        )
    }

    @Test
    fun testUpsidePopulationVariance() {
        assertEquals(
            20.0 / 8.0,
            data.semiVariance(direction = SemiVarianceDirection.UPSIDE, kind = PopulationKind.POPULATION),
            1e-10
        )
    }

    // ── sum property: downside + upside = variance when threshold = mean ────

    @Test
    fun testSumPropertySample() {
        val down = data.semiVariance(direction = SemiVarianceDirection.DOWNSIDE)
        val up = data.semiVariance(direction = SemiVarianceDirection.UPSIDE)
        assertEquals(data.toList().variance(), down + up, 1e-10)
    }

    @Test
    fun testSumPropertyPopulation() {
        val down = data.semiVariance(direction = SemiVarianceDirection.DOWNSIDE, kind = PopulationKind.POPULATION)
        val up = data.semiVariance(direction = SemiVarianceDirection.UPSIDE, kind = PopulationKind.POPULATION)
        assertEquals(data.toList().variance(PopulationKind.POPULATION), down + up, 1e-10)
    }

    // ── custom threshold ────────────────────────────────────────────────────

    @Test
    fun testCustomThresholdDownside() {
        // [1,2,3,4,5], threshold=3.0: below diffs = -2,-1 → sum of squares = 5
        val arr = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(5.0 / 4.0, arr.semiVariance(threshold = 3.0, direction = SemiVarianceDirection.DOWNSIDE), 1e-10)
    }

    @Test
    fun testCustomThresholdUpside() {
        // [1,2,3,4,5], threshold=3.0: above diffs = +1,+2 → sum of squares = 5
        val arr = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(5.0 / 4.0, arr.semiVariance(threshold = 3.0, direction = SemiVarianceDirection.UPSIDE), 1e-10)
    }

    // ── edge cases ──────────────────────────────────────────────────────────

    @Test
    fun testEmptyThrows() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf().semiVariance() }
    }

    @Test
    fun testSingleElementSampleThrows() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf(5.0).semiVariance() }
    }

    @Test
    fun testSingleElementPopulation() {
        val result = doubleArrayOf(5.0).semiVariance(kind = PopulationKind.POPULATION)
        assertEquals(0.0, result, 1e-10)
    }

    @Test
    fun testConstantArrayReturnsZero() {
        val constant = doubleArrayOf(3.0, 3.0, 3.0, 3.0)
        assertEquals(0.0, constant.semiVariance(direction = SemiVarianceDirection.DOWNSIDE), 1e-10)
        assertEquals(0.0, constant.semiVariance(direction = SemiVarianceDirection.UPSIDE), 1e-10)
    }

    @Test
    fun testAllOnOneSideDownside() {
        // All values below threshold → upside is 0
        val arr = doubleArrayOf(1.0, 2.0, 3.0)
        assertEquals(0.0, arr.semiVariance(threshold = 10.0, direction = SemiVarianceDirection.UPSIDE), 1e-10)
    }

    @Test
    fun testAllOnOneSideUpside() {
        // All values above threshold → downside is 0
        val arr = doubleArrayOf(10.0, 20.0, 30.0)
        assertEquals(0.0, arr.semiVariance(threshold = 0.0, direction = SemiVarianceDirection.DOWNSIDE), 1e-10)
    }

    // ── default parameters ──────────────────────────────────────────────────

    @Test
    fun testDefaultParameters() {
        val defaultResult = data.semiVariance()
        val explicitResult = data.semiVariance(
            threshold = data.mean(),
            direction = SemiVarianceDirection.DOWNSIDE
        )
        assertEquals(explicitResult, defaultResult, 1e-15)
    }

    // ── NaN ─────────────────────────────────────────────────────────────────

    @Test
    fun testNaNInData() {
        val withNaN = doubleArrayOf(1.0, Double.NaN, 3.0, 4.0, 5.0)
        assertTrue(withNaN.semiVariance(threshold = 3.0).isNaN())
    }

    // ── large offset (compensated sum precision) ────────────────────────────

    @Test
    fun testLargeOffsetPrecision() {
        val n = 1000
        val base = 1e15
        val arr = DoubleArray(n) { base + it.toDouble() }
        val mean = arr.mean()
        val down = arr.semiVariance(direction = SemiVarianceDirection.DOWNSIDE)
        val up = arr.semiVariance(direction = SemiVarianceDirection.UPSIDE)
        // downside + upside must equal sample variance
        val variance = arr.toList().variance()
        assertEquals(variance, down + up, variance * 1e-10)
    }

    // ── Iterable and Sequence overloads ─────────────────────────────────────

    @Test
    fun testIterableMatchesDoubleArray() {
        val list = data.toList()
        assertEquals(
            data.semiVariance(direction = SemiVarianceDirection.DOWNSIDE),
            list.semiVariance(direction = SemiVarianceDirection.DOWNSIDE),
            1e-15
        )
        assertEquals(
            data.semiVariance(direction = SemiVarianceDirection.UPSIDE),
            list.semiVariance(direction = SemiVarianceDirection.UPSIDE),
            1e-15
        )
    }

    @Test
    fun testSequenceMatchesDoubleArray() {
        assertEquals(
            data.semiVariance(direction = SemiVarianceDirection.DOWNSIDE),
            data.asSequence().semiVariance(direction = SemiVarianceDirection.DOWNSIDE),
            1e-15
        )
        assertEquals(
            data.semiVariance(direction = SemiVarianceDirection.UPSIDE),
            data.asSequence().semiVariance(direction = SemiVarianceDirection.UPSIDE),
            1e-15
        )
    }
}

class CentralMomentTest {

    private val data1 = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
    private val data2 = doubleArrayOf(1.0, 1.0, 1.0, 2.0, 5.0, 10.0)

    // ── basic correctness (scipy reference) ─────────────────────────────

    @Test
    fun testOrder0() {
        assertEquals(1.0, data1.centralMoment(0), 1e-10)
    }

    @Test
    fun testOrder1() {
        assertEquals(0.0, data1.centralMoment(1), 1e-10)
    }

    @Test
    fun testOrder2Dataset1() {
        assertEquals(4.0, data1.centralMoment(2), 1e-10)
    }

    @Test
    fun testOrder3Dataset1() {
        assertEquals(5.25, data1.centralMoment(3), 1e-10)
    }

    @Test
    fun testOrder4Dataset1() {
        assertEquals(44.5, data1.centralMoment(4), 1e-10)
    }

    @Test
    fun testOrder5Dataset1() {
        assertEquals(101.25, data1.centralMoment(5), 1e-10)
    }

    @Test
    fun testOrder2Dataset2() {
        assertEquals(10.888888888888889, data2.centralMoment(2), 1e-10)
    }

    @Test
    fun testOrder3Dataset2() {
        assertEquals(43.407407407407405, data2.centralMoment(3), 1e-10)
    }

    @Test
    fun testOrder4Dataset2() {
        assertEquals(345.85185185185185, data2.centralMoment(4), 1e-8)
    }

    @Test
    fun testOrder5Dataset2() {
        assertEquals(2161.6460905349794, data2.centralMoment(5), 1e-6)
    }

    // ── cross-validation ────────────────────────────────────────────────

    @Test
    fun testOrder2EqualsPopulationVariance() {
        assertEquals(
            data1.toList().variance(PopulationKind.POPULATION),
            data1.centralMoment(2),
            1e-15
        )
    }

    // ── symmetric data ──────────────────────────────────────────────────

    @Test
    fun testSymmetricOddMomentsAreZero() {
        val symmetric = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(0.0, symmetric.centralMoment(1), 1e-10)
        assertEquals(0.0, symmetric.centralMoment(3), 1e-10)
        assertEquals(0.0, symmetric.centralMoment(5), 1e-10)
    }

    // ── constant data ───────────────────────────────────────────────────

    @Test
    fun testConstantDataReturnsZero() {
        val constant = doubleArrayOf(5.0, 5.0, 5.0, 5.0)
        for (order in 1..5) {
            assertEquals(0.0, constant.centralMoment(order), 1e-15,
                "centralMoment($order) of constant data should be 0.0")
        }
    }

    // ── edge cases ──────────────────────────────────────────────────────

    @Test
    fun testSingleElement() {
        val single = doubleArrayOf(42.0)
        assertEquals(1.0, single.centralMoment(0), 1e-15)
        for (order in 1..5) {
            assertEquals(0.0, single.centralMoment(order), 1e-15)
        }
    }

    @Test
    fun testEmptyThrows() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf().centralMoment(2) }
    }

    @Test
    fun testNegativeOrderThrows() {
        assertFailsWith<InvalidParameterException> { data1.centralMoment(-1) }
    }

    // ── numerical stability ─────────────────────────────────────────────

    @Test
    fun testLargeOffsetAllOrdersFinite() {
        // Large offset but small spread — all central moments should be finite
        val data = doubleArrayOf(1e15 + 1.0, 1e15 + 2.0, 1e15 + 3.0, 1e15 + 4.0, 1e15 + 5.0)
        for (order in 2..5) {
            val result = data.centralMoment(order)
            assertTrue(result.isFinite(), "centralMoment($order) must be finite for large-offset data, got $result")
        }
    }

    @Test
    fun testLargeMagnitudeSymmetricOddMomentsZero() {
        // Symmetric data: odd moments should be 0 even at large scale
        // (z-normalization ensures no overflow in the accumulation)
        val data = doubleArrayOf(1e154, 2e154, 3e154, 4e154, 5e154)
        assertEquals(0.0, data.centralMoment(3), 1e-10)
        assertEquals(0.0, data.centralMoment(5), 1e-10)
    }

    // ── NaN ─────────────────────────────────────────────────────────────

    @Test
    fun testNaNInData() {
        val withNaN = doubleArrayOf(1.0, Double.NaN, 3.0, 4.0, 5.0)
        assertTrue(withNaN.centralMoment(2).isNaN())
    }

    // ── overloads ───────────────────────────────────────────────────────

    @Test
    fun testOverloadsMatch() {
        val list = data1.toList()
        val seq = data1.asSequence()
        for (order in 0..5) {
            val expected = data1.centralMoment(order)
            assertEquals(expected, list.centralMoment(order), 1e-15,
                "Iterable overload differs at order $order")
            assertEquals(expected, seq.centralMoment(order), 1e-15,
                "Sequence overload differs at order $order")
        }
    }
}

class KStatisticTest {

    private val data1 = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
    private val data2 = doubleArrayOf(1.0, 1.0, 1.0, 2.0, 5.0, 10.0)

    // ── basic correctness (scipy reference) ─────────────────────────────

    @Test
    fun testOrder1Dataset1() {
        assertEquals(5.0, data1.kStatistic(1), 1e-10)
    }

    @Test
    fun testOrder2Dataset1() {
        assertEquals(4.571428571428571, data1.kStatistic(2), 1e-10)
    }

    @Test
    fun testOrder3Dataset1() {
        assertEquals(8.0, data1.kStatistic(3), 1e-10)
    }

    @Test
    fun testOrder4Dataset1() {
        assertEquals(19.657142857142857, data1.kStatistic(4), 1e-8)
    }

    @Test
    fun testOrder1Dataset2() {
        assertEquals(3.333333333333333, data2.kStatistic(1), 1e-10)
    }

    @Test
    fun testOrder2Dataset2() {
        assertEquals(13.066666666666666, data2.kStatistic(2), 1e-10)
    }

    @Test
    fun testOrder3Dataset2() {
        assertEquals(78.133333333333333, data2.kStatistic(3), 1e-8)
    }

    @Test
    fun testOrder4Dataset2() {
        assertEquals(385.46666666666667, data2.kStatistic(4), 1e-6)
    }

    // ── cross-validation ────────────────────────────────────────────────

    @Test
    fun testOrder1EqualsMean() {
        assertEquals(data1.toList().mean(), data1.kStatistic(1), 1e-15)
    }

    @Test
    fun testOrder2EqualsSampleVariance() {
        assertEquals(data1.toList().variance(), data1.kStatistic(2), 1e-15)
    }

    // ── constant data ───────────────────────────────────────────────────

    @Test
    fun testConstantData() {
        val constant = doubleArrayOf(5.0, 5.0, 5.0, 5.0)
        assertEquals(5.0, constant.kStatistic(1), 1e-15)
        assertEquals(0.0, constant.kStatistic(2), 1e-15)
        assertEquals(0.0, constant.kStatistic(3), 1e-15)
        assertEquals(0.0, constant.kStatistic(4), 1e-15)
    }

    // ── minimum n ───────────────────────────────────────────────────────

    @Test
    fun testOrder1WithN1() {
        assertEquals(42.0, doubleArrayOf(42.0).kStatistic(1), 1e-15)
    }

    @Test
    fun testOrder2WithN1Throws() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf(42.0).kStatistic(2) }
    }

    @Test
    fun testOrder2WithN2() {
        val result = doubleArrayOf(1.0, 3.0).kStatistic(2)
        assertTrue(result.isFinite())
        assertEquals(doubleArrayOf(1.0, 3.0).toList().variance(), result, 1e-15)
    }

    @Test
    fun testOrder3WithN2Throws() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf(1.0, 3.0).kStatistic(3) }
    }

    @Test
    fun testOrder3WithN3() {
        val result = doubleArrayOf(1.0, 2.0, 3.0).kStatistic(3)
        assertTrue(result.isFinite())
    }

    @Test
    fun testOrder4WithN3Throws() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf(1.0, 2.0, 3.0).kStatistic(4) }
    }

    @Test
    fun testOrder4WithN4() {
        val result = doubleArrayOf(1.0, 2.0, 3.0, 4.0).kStatistic(4)
        assertTrue(result.isFinite())
    }

    // ── validation ──────────────────────────────────────────────────────

    @Test
    fun testOrder0Throws() {
        assertFailsWith<InvalidParameterException> { data1.kStatistic(0) }
    }

    @Test
    fun testOrder5Throws() {
        assertFailsWith<InvalidParameterException> { data1.kStatistic(5) }
    }

    @Test
    fun testNegativeOrderThrows() {
        assertFailsWith<InvalidParameterException> { data1.kStatistic(-1) }
    }

    @Test
    fun testEmptyThrows() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf().kStatistic(1) }
    }

    // ── numerical stability ─────────────────────────────────────────────

    @Test
    fun testLargeMagnitudeK1Finite() {
        val data = doubleArrayOf(1e154, 2e154, 3e154, 4e154, 5e154)
        val result = data.kStatistic(1)
        assertTrue(result.isFinite(), "kStatistic(1) must be finite for large data, got $result")
        assertEquals(3e154, result, 1e144)
    }

    @Test
    fun testLargeOffsetAllOrdersFinite() {
        // Large offset but small spread — all k-statistics should be finite
        val data = doubleArrayOf(1e15 + 1.0, 1e15 + 2.0, 1e15 + 3.0, 1e15 + 4.0, 1e15 + 5.0)
        for (order in 1..4) {
            val result = data.kStatistic(order)
            assertTrue(result.isFinite(), "kStatistic($order) must be finite for large-offset data, got $result")
        }
    }

    // ── NaN ─────────────────────────────────────────────────────────────

    @Test
    fun testNaNInData() {
        val withNaN = doubleArrayOf(1.0, Double.NaN, 3.0, 4.0, 5.0)
        assertTrue(withNaN.kStatistic(2).isNaN())
    }

    // ── overloads ───────────────────────────────────────────────────────

    @Test
    fun testOverloadsMatch() {
        val list = data1.toList()
        val seq = data1.asSequence()
        for (order in 1..4) {
            val expected = data1.kStatistic(order)
            assertEquals(expected, list.kStatistic(order), 1e-15,
                "Iterable overload differs at order $order")
            assertEquals(expected, seq.kStatistic(order), 1e-15,
                "Sequence overload differs at order $order")
        }
    }
}
