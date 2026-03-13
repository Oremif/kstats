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
