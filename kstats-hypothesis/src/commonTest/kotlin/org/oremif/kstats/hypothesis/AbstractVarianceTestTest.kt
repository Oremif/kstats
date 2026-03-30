package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Base class for variance homogeneity tests (Bartlett, Levene, Fligner-Killeen).
 *
 * Provides shared tests for input validation, NaN/Infinity propagation,
 * degenerate cases, and property-based checks. Subclasses supply the
 * [runTest] function and their own scipy golden-value tests.
 */
abstract class AbstractVarianceTestTest {

    protected val tolStat = 1e-10
    protected val tolP = 1e-3

    /** Execute the test under examination. */
    protected abstract fun runTest(vararg groups: DoubleArray): TestResult

    // ── Input validation ───────────────────────────────────────────────────

    @Test
    fun fewerThan2Groups() {
        assertFailsWith<InsufficientDataException> {
            runTest(doubleArrayOf(1.0, 2.0))
        }
    }

    @Test
    fun zeroGroups() {
        assertFailsWith<InsufficientDataException> {
            @Suppress("SpreadOperator")
            runTest()
        }
    }

    @Test
    fun groupSizeLessThan2() {
        assertFailsWith<InsufficientDataException> {
            runTest(doubleArrayOf(1.0), doubleArrayOf(2.0, 3.0))
        }
    }

    @Test
    fun emptyGroup() {
        assertFailsWith<InsufficientDataException> {
            runTest(doubleArrayOf(), doubleArrayOf(2.0, 3.0))
        }
    }

    // ── Non-finite input ───────────────────────────────────────────────────

    @Test
    fun nanInGroup() {
        val result = runTest(TestData.WITH_NAN, TestData.SEQUENTIAL_6_10)
        assertTrue(result.pValue.isNaN(), "NaN in input should produce NaN p-value")
    }

    @Test
    fun infinityInGroup() {
        val result = runTest(TestData.WITH_POS_INF, TestData.SEQUENTIAL_6_10)
        assertTrue(result.pValue.isNaN(), "Infinity in input should produce NaN p-value")
    }

    // ── Degenerate cases ───────────────────────────────────────────────────

    @Test
    fun allConstantSameValue() {
        val result = runTest(TestData.CONSTANT_5x3, TestData.CONSTANT_5x3)
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    @Test
    fun allConstantDifferentValues() {
        val result = runTest(TestData.CONSTANT_3x3, TestData.CONSTANT_7x3)
        assertEquals(0.0, result.statistic)
        assertEquals(1.0, result.pValue)
    }

    // ── Property-based ─────────────────────────────────────────────────────

    @Test
    fun identicalGroupsStatisticNearZero() {
        val data = doubleArrayOf(1.0, 3.0, 5.0, 7.0, 9.0)
        val result = runTest(data, data.copyOf(), data.copyOf())
        assertEquals(0.0, result.statistic, 1e-10)
        assertTrue(result.pValue > 0.99, "Identical groups should have p-value near 1")
    }

    @Test
    fun isSignificantConsistency() {
        val result = runTest(TestData.LOW_VARIANCE, TestData.HIGH_VARIANCE)
        TestAssertions.assertIsSignificantConsistency(result, 0.05)
        TestAssertions.assertIsSignificantConsistency(result, 0.01)
    }

    // ── Common golden values (equal variance → stat=0, p=1) ────────────────

    @Test
    fun twoGroupsEqualVariance() {
        val result = runTest(TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10)
        assertEquals(0.0, result.statistic, 1e-6)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun threeGroupsEqualVariance() {
        val result = runTest(TestData.SEQUENTIAL_1_5, TestData.SEQUENTIAL_6_10, TestData.SEQUENTIAL_11_15)
        assertEquals(0.0, result.statistic, 1e-6)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    open fun unequalGroupSizes() {
        // Each test has different expected values, but the setup is the same.
        // Subclasses override if needed; default just checks p is in [0,1].
        val result = runTest(TestData.SHORT_3, TestData.LONG_5)
        assertTrue(result.pValue in 0.0..1.0)
    }

    @Test
    fun largeGroupsEqualVariance() {
        val g1 = TestData.largeSequential(100)
        val g2 = TestData.largeSequential(100, offset = 1000.0)
        val result = runTest(g1, g2)
        assertEquals(0.0, result.statistic, 1e-6)
        assertTrue(result.pValue > 0.99, "Equal variances should have p-value near 1")
    }
}
