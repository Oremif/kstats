package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

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
