package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
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
