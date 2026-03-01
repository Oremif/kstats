package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TrimmedStatisticsTest {

    private val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)

    // ── trimmedMean ─────────────────────────────────────────────────────────

    @Test
    fun testTrimmedMeanBasic() {
        // scipy: trim_mean([1..10], 0.1) = 5.5 (trim 1 from each end → [2..9])
        assertEquals(5.5, data.trimmedMean(0.1), 1e-10)
        // scipy: trim_mean([1..10], 0.2) = 5.5 (trim 2 from each end → [3..8])
        assertEquals(5.5, data.trimmedMean(0.2), 1e-10)
        // scipy: trim_mean([1..10], 0.3) = 5.5 (trim 3 from each end → [4..7])
        assertEquals(5.5, data.trimmedMean(0.3), 1e-10)
    }

    @Test
    fun testTrimmedMeanProportionZero() {
        // proportion=0.0 should equal regular mean
        assertEquals(data.mean(), data.trimmedMean(0.0), 1e-10)
    }

    @Test
    fun testTrimmedMeanSingleElement() {
        assertEquals(42.0, doubleArrayOf(42.0).trimmedMean(0.0), 1e-10)
    }

    @Test
    fun testTrimmedMeanLeavesOneElement() {
        // 3 elements, proportion=0.4 → k=floor(3*0.4)=1, trimmed=[middle], mean=middle
        val arr = doubleArrayOf(1.0, 5.0, 9.0)
        assertEquals(5.0, arr.trimmedMean(0.4), 1e-10)
    }

    @Test
    fun testTrimmedMeanEmptyThrows() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf().trimmedMean(0.1) }
    }

    @Test
    fun testTrimmedMeanInvalidProportionThrows() {
        assertFailsWith<InvalidParameterException> { data.trimmedMean(-0.1) }
        assertFailsWith<InvalidParameterException> { data.trimmedMean(0.5) }
        assertFailsWith<InvalidParameterException> { data.trimmedMean(1.0) }
        assertFailsWith<InvalidParameterException> { data.trimmedMean(Double.NaN) }
    }

    @Test
    fun testTrimmedMeanLargeOffset() {
        // Compensated sum precision with large offsets
        val largeData = DoubleArray(1000) { 1e15 + it.toDouble() }
        val expected = 1e15 + 499.5 // mean of [100..899]
        val result = largeData.trimmedMean(0.1)
        assertTrue(abs(result - expected) < 1e-2, "Expected $expected but got $result")
    }

    @Test
    fun testTrimmedMeanNaN() {
        val withNaN = doubleArrayOf(1.0, Double.NaN, 3.0, 4.0, 5.0)
        assertTrue(withNaN.trimmedMean(0.0).isNaN())
    }

    @Test
    fun testTrimmedMeanConstantArray() {
        val constant = DoubleArray(10) { 7.0 }
        assertEquals(7.0, constant.trimmedMean(0.2), 1e-10)
    }

    @Test
    fun testTrimmedMeanIterableAndSequence() {
        val list = data.toList()
        assertEquals(data.trimmedMean(0.2), list.trimmedMean(0.2), 1e-15)
        assertEquals(data.trimmedMean(0.2), data.asSequence().trimmedMean(0.2), 1e-15)
    }

    // ── trimmedVariance ─────────────────────────────────────────────────────

    @Test
    fun testTrimmedVarianceBasic() {
        // scipy reference: sorted [1..10], p=0.1 → [2..9], sample var = 6.0
        assertEquals(6.0, data.trimmedVariance(0.1), 1e-10)
        // p=0.2 → [3..8], sample var = 3.5
        assertEquals(3.5, data.trimmedVariance(0.2), 1e-10)
        // p=0.3 → [4..7], sample var = 5/3
        assertEquals(5.0 / 3.0, data.trimmedVariance(0.3), 1e-10)
    }

    @Test
    fun testTrimmedVariancePopulation() {
        // p=0.1 → [2..9], population var = 5.25
        assertEquals(5.25, data.trimmedVariance(0.1, PopulationKind.POPULATION), 1e-10)
        // p=0.2 → [3..8], population var = 35/12
        assertEquals(35.0 / 12.0, data.trimmedVariance(0.2, PopulationKind.POPULATION), 1e-10)
        // p=0.3 → [4..7], population var = 1.25
        assertEquals(1.25, data.trimmedVariance(0.3, PopulationKind.POPULATION), 1e-10)
    }

    @Test
    fun testTrimmedVarianceProportionZero() {
        // proportion=0.0 should equal regular variance
        assertEquals(data.toList().variance(), data.trimmedVariance(0.0), 1e-10)
        assertEquals(
            data.toList().variance(PopulationKind.POPULATION),
            data.trimmedVariance(0.0, PopulationKind.POPULATION),
            1e-10
        )
    }

    @Test
    fun testTrimmedVarianceEmptyThrows() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf().trimmedVariance(0.1) }
    }

    @Test
    fun testTrimmedVarianceInvalidProportionThrows() {
        assertFailsWith<InvalidParameterException> { data.trimmedVariance(-0.1) }
        assertFailsWith<InvalidParameterException> { data.trimmedVariance(0.5) }
        assertFailsWith<InvalidParameterException> { data.trimmedVariance(1.0) }
        assertFailsWith<InvalidParameterException> { data.trimmedVariance(Double.NaN) }
    }

    @Test
    fun testTrimmedVarianceSingleRemainingElementSampleThrows() {
        // 3 elements, proportion=0.4 → k=1, m=1: sample variance requires ≥2
        val arr = doubleArrayOf(1.0, 5.0, 9.0)
        assertFailsWith<InsufficientDataException> { arr.trimmedVariance(0.4, PopulationKind.SAMPLE) }
    }

    @Test
    fun testTrimmedVarianceSingleRemainingElementPopulation() {
        // Population variance of single element is 0
        val arr = doubleArrayOf(1.0, 5.0, 9.0)
        assertEquals(0.0, arr.trimmedVariance(0.4, PopulationKind.POPULATION), 1e-10)
    }

    @Test
    fun testTrimmedVarianceConstantArray() {
        val constant = DoubleArray(10) { 7.0 }
        assertEquals(0.0, constant.trimmedVariance(0.2), 1e-10)
    }

    @Test
    fun testTrimmedVarianceNaN() {
        val withNaN = doubleArrayOf(1.0, Double.NaN, 3.0, 4.0, 5.0)
        assertTrue(withNaN.trimmedVariance(0.0).isNaN())
    }

    @Test
    fun testTrimmedVarianceIterableAndSequence() {
        val list = data.toList()
        assertEquals(data.trimmedVariance(0.2), list.trimmedVariance(0.2), 1e-15)
        assertEquals(data.trimmedVariance(0.2), data.asSequence().trimmedVariance(0.2), 1e-15)
    }

    // ── trimmedStd ──────────────────────────────────────────────────────────

    @Test
    fun testTrimmedStdBasic() {
        // sqrt of trimmedVariance
        assertEquals(sqrt(6.0), data.trimmedStd(0.1), 1e-10)
        assertEquals(sqrt(3.5), data.trimmedStd(0.2), 1e-10)
        assertEquals(sqrt(5.0 / 3.0), data.trimmedStd(0.3), 1e-10)
    }

    @Test
    fun testTrimmedStdPopulation() {
        assertEquals(sqrt(5.25), data.trimmedStd(0.1, PopulationKind.POPULATION), 1e-10)
        assertEquals(sqrt(35.0 / 12.0), data.trimmedStd(0.2, PopulationKind.POPULATION), 1e-10)
        assertEquals(sqrt(1.25), data.trimmedStd(0.3, PopulationKind.POPULATION), 1e-10)
    }

    @Test
    fun testTrimmedStdProportionZero() {
        assertEquals(data.toList().standardDeviation(), data.trimmedStd(0.0), 1e-10)
    }

    @Test
    fun testTrimmedStdEmptyThrows() {
        assertFailsWith<InsufficientDataException> { doubleArrayOf().trimmedStd(0.1) }
    }

    @Test
    fun testTrimmedStdInvalidProportionThrows() {
        assertFailsWith<InvalidParameterException> { data.trimmedStd(-0.1) }
        assertFailsWith<InvalidParameterException> { data.trimmedStd(0.5) }
        assertFailsWith<InvalidParameterException> { data.trimmedStd(Double.NaN) }
    }

    @Test
    fun testTrimmedStdIterableAndSequence() {
        val list = data.toList()
        assertEquals(data.trimmedStd(0.2), list.trimmedStd(0.2), 1e-15)
        assertEquals(data.trimmedStd(0.2), data.asSequence().trimmedStd(0.2), 1e-15)
    }

    @Test
    fun testTrimmedStdConsistentWithVariance() {
        // Verify sqrt(trimmedVariance) == trimmedStd for various proportions
        for (p in listOf(0.0, 0.1, 0.2, 0.3, 0.4)) {
            val v = data.trimmedVariance(p)
            val s = data.trimmedStd(p)
            assertEquals(sqrt(v), s, 1e-15, "trimmedStd($p) should equal sqrt(trimmedVariance($p))")
        }
    }
}
