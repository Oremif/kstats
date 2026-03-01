package org.oremif.kstats.correlation

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.descriptive.PopulationKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class CovarianceTest {

    // ── Basic correctness ──────────────────────────────────────────────

    @Test
    fun testSampleCovariancePositive() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        assertEquals(5.0, covariance(x, y), 1e-10)
    }

    @Test
    fun testPopulationCovariancePositive() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        assertEquals(4.0, covariance(x, y, PopulationKind.POPULATION), 1e-10)
    }

    @Test
    fun testSampleCovarianceNegative() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(10.0, 8.0, 6.0, 4.0, 2.0)
        assertEquals(-5.0, covariance(x, y), 1e-10)
    }

    @Test
    fun testSampleCovarianceMixedValues() {
        // scipy: np.cov([2.1,2.5,3.6,4.0,5.2,6.1,7.8,8.3,9.0,10.5],
        //              [8.0,7.2,6.5,5.8,5.0,4.3,3.2,2.5,1.8,0.5], ddof=1)[0,1]
        val x = doubleArrayOf(2.1, 2.5, 3.6, 4.0, 5.2, 6.1, 7.8, 8.3, 9.0, 10.5)
        val y = doubleArrayOf(8.0, 7.2, 6.5, 5.8, 5.0, 4.3, 3.2, 2.5, 1.8, 0.5)
        assertEquals(-7.108666666666666, covariance(x, y), 1e-10)
    }

    @Test
    fun testPopulationCovarianceMixedValues() {
        val x = doubleArrayOf(2.1, 2.5, 3.6, 4.0, 5.2, 6.1, 7.8, 8.3, 9.0, 10.5)
        val y = doubleArrayOf(8.0, 7.2, 6.5, 5.8, 5.0, 4.3, 3.2, 2.5, 1.8, 0.5)
        assertEquals(-6.3978, covariance(x, y, PopulationKind.POPULATION), 1e-10)
    }

    // ── Edge cases ─────────────────────────────────────────────────────

    @Test
    fun testCovarianceMinimumSize() {
        // n=2
        val x = doubleArrayOf(1.0, 3.0)
        val y = doubleArrayOf(2.0, 6.0)
        assertEquals(4.0, covariance(x, y), 1e-10)
        assertEquals(2.0, covariance(x, y, PopulationKind.POPULATION), 1e-10)
    }

    @Test
    fun testCovarianceIdenticalArraysEqualsVariance() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        // cov(x, x) == var(x) = 2.5
        assertEquals(2.5, covariance(x, x), 1e-10)
    }

    @Test
    fun testCovarianceOneConstantArray() {
        val constant = doubleArrayOf(3.0, 3.0, 3.0, 3.0, 3.0)
        val varying = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(0.0, covariance(constant, varying), 1e-10)
        assertEquals(0.0, covariance(varying, constant), 1e-10)
    }

    // ── Degenerate input ───────────────────────────────────────────────

    @Test
    fun testCovarianceBothConstant() {
        val x = doubleArrayOf(5.0, 5.0, 5.0)
        val y = doubleArrayOf(3.0, 3.0, 3.0)
        assertEquals(0.0, covariance(x, y), 1e-10)
    }

    @Test
    fun testCovarianceNearlyConstant() {
        // Values with tiny differences around a large base
        val x = doubleArrayOf(1e15, 1e15 + 1.0, 1e15 + 2.0)
        val y = doubleArrayOf(1e15, 1e15 + 2.0, 1e15 + 4.0)
        // scipy: 2.0
        assertEquals(2.0, covariance(x, y), 1e-6)
    }

    // ── Extreme parameters / numerical stability ───────────────────────

    @Test
    fun testCovarianceLargeOffset() {
        // Key numerical stability test: large constant offset where naive
        // two-pass can lose precision but Welford maintains accuracy.
        val offset = 1e12
        val x = DoubleArray(10) { offset + (it + 1).toDouble() }
        val y = DoubleArray(10) { offset + 2.0 * (it + 1).toDouble() }
        // scipy reference: 18.333333333333332
        assertEquals(18.333333333333332, covariance(x, y), 1e-6)
        assertEquals(16.5, covariance(x, y, PopulationKind.POPULATION), 1e-6)
    }

    @Test
    fun testCovarianceLargeOffsetMatchesNoOffset() {
        // With Welford, offset should not affect result
        val offset = 1e12
        val xOffset = DoubleArray(10) { offset + (it + 1).toDouble() }
        val yOffset = DoubleArray(10) { offset + 2.0 * (it + 1).toDouble() }
        val xPlain = DoubleArray(10) { (it + 1).toDouble() }
        val yPlain = DoubleArray(10) { 2.0 * (it + 1).toDouble() }
        assertEquals(covariance(xPlain, yPlain), covariance(xOffset, yOffset), 1e-6)
    }

    @Test
    fun testCovarianceLargeN() {
        // 10000 points with known linear relationship + noise (seeded RNG for reproducibility)
        val n = 10000
        val rng = kotlin.random.Random(42)
        val x = DoubleArray(n) { rng.nextDouble() * 100.0 }
        val y = DoubleArray(n) { 0.5 * x[it] + rng.nextDouble() * 10.0 }

        val covSample = covariance(x, y)
        val covPop = covariance(x, y, PopulationKind.POPULATION)

        // Sample and population should converge for large n
        assertEquals(covSample, covPop, covSample * 0.001)
        // Covariance should be positive (positive linear relationship)
        assertTrue(covSample > 0.0, "Covariance should be positive for positively correlated data")
    }

    // ── Non-finite input ───────────────────────────────────────────────

    @Test
    fun testCovarianceNaNPropagation() {
        val x = doubleArrayOf(1.0, Double.NaN, 3.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0)
        assertTrue(covariance(x, y).isNaN(), "NaN in x should propagate")
    }

    @Test
    fun testCovarianceNaNInY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(2.0, Double.NaN, 6.0)
        assertTrue(covariance(x, y).isNaN(), "NaN in y should propagate")
    }

    @Test
    fun testCovarianceInfHandling() {
        val x = doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0)
        val result = covariance(x, y)
        assertTrue(result.isNaN() || result.isInfinite(), "Inf input should produce non-finite result")
    }

    // ── Validation ─────────────────────────────────────────────────────

    @Test
    fun testCovarianceDifferentSizesThrows() {
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(1.0, 2.0)
        assertFailsWith<InvalidParameterException> { covariance(x, y) }
    }

    @Test
    fun testCovarianceTooFewObservationsThrows() {
        val x = doubleArrayOf(1.0)
        val y = doubleArrayOf(2.0)
        assertFailsWith<InsufficientDataException> { covariance(x, y) }
    }
}
