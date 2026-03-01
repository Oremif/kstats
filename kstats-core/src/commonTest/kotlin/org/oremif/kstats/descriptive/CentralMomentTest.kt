package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

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
            assertEquals(
                0.0, constant.centralMoment(order), 1e-15,
                "centralMoment($order) of constant data should be 0.0"
            )
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
            assertEquals(
                expected, list.centralMoment(order), 1e-15,
                "Iterable overload differs at order $order"
            )
            assertEquals(
                expected, seq.centralMoment(order), 1e-15,
                "Sequence overload differs at order $order"
            )
        }
    }
}
