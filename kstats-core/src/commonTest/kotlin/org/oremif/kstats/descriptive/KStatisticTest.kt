package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

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
