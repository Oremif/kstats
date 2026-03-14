package org.oremif.kstats.descriptive

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

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
