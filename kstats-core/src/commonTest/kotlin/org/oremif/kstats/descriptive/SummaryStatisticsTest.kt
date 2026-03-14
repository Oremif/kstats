package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

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
