package org.oremif.kstats.descriptive

import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

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
