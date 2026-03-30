package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class EffectSizeTest {

    private val tol = 1e-10

    // ===== Basic correctness: pooled =====

    @Test
    fun testCohensDPooledKnownValues() {
        // scipy (manual): cohens_d_pooled([1,2,3,4,5], [2,4,6,8,10]) = -1.2
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        assertEquals(-1.2, cohensD(x, y, pooled = true), tol, "pooled d for simple arrays")
    }

    @Test
    fun testCohensDPooledUnequalSizes() {
        // scipy (manual): cohens_d_pooled([10,12,11,13,14,15], [8,9,7,10]) = 2.38513917599978
        val x = doubleArrayOf(10.0, 12.0, 11.0, 13.0, 14.0, 15.0)
        val y = doubleArrayOf(8.0, 9.0, 7.0, 10.0)
        assertEquals(2.38513917599978, cohensD(x, y, pooled = true), tol, "pooled d with unequal sizes")
    }

    @Test
    fun testCohensDPooledLargeEffect() {
        // scipy (manual): cohens_d_pooled([100,101,102,103,104], [0,1,2,3,4]) = 63.2455532033676
        val x = doubleArrayOf(100.0, 101.0, 102.0, 103.0, 104.0)
        val y = doubleArrayOf(0.0, 1.0, 2.0, 3.0, 4.0)
        assertEquals(63.2455532033676, cohensD(x, y, pooled = true), tol, "pooled d for large effect")
    }

    @Test
    fun testCohensDPooledNegativeEffect() {
        // scipy (manual): cohens_d_pooled([1,2,3], [10,11,12]) = -9.0
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(10.0, 11.0, 12.0)
        assertEquals(-9.0, cohensD(x, y, pooled = true), tol, "pooled d for negative effect")
    }

    @Test
    fun testCohensDPooledDifferentVariances() {
        // scipy (manual): cohens_d_pooled([1,2,3,4,5], [10,20,30,40,50]) = -2.40296846124592
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0)
        assertEquals(-2.40296846124592, cohensD(x, y, pooled = true), tol, "pooled d with different variances")
    }

    // ===== Basic correctness: unpooled =====

    @Test
    fun testCohensDUnpooledKnownValues() {
        // scipy (manual): cohens_d_unpooled([1,2,3,4,5], [2,4,6,8,10]) = -1.2
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        assertEquals(-1.2, cohensD(x, y, pooled = false), tol, "unpooled d for simple arrays")
    }

    @Test
    fun testCohensDUnpooledUnequalSizes() {
        // scipy (manual): cohens_d_unpooled([10,12,11,13,14,15], [8,9,7,10]) = 2.48868406735302
        val x = doubleArrayOf(10.0, 12.0, 11.0, 13.0, 14.0, 15.0)
        val y = doubleArrayOf(8.0, 9.0, 7.0, 10.0)
        assertEquals(2.48868406735302, cohensD(x, y, pooled = false), tol, "unpooled d with unequal sizes")
    }

    @Test
    fun testCohensDUnpooledDifferentVariances() {
        // scipy (manual): cohens_d_unpooled([1,2,3,4,5], [10,20,30,40,50]) = -2.40296846124592
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0)
        assertEquals(-2.40296846124592, cohensD(x, y, pooled = false), tol, "unpooled d with different variances")
    }

    // ===== Basic: pooled default parameter =====

    @Test
    fun testCohensDDefaultIsPooled() {
        val x = doubleArrayOf(10.0, 12.0, 11.0, 13.0, 14.0, 15.0)
        val y = doubleArrayOf(8.0, 9.0, 7.0, 10.0)
        assertEquals(
            cohensD(x, y, pooled = true),
            cohensD(x, y),
            0.0,
            "default should be pooled"
        )
    }

    // ===== Edge cases =====

    @Test
    fun testCohensDMinimumSampleSize() {
        // Minimum valid: 2 elements per array
        // scipy (manual): cohens_d_pooled([1,3], [5,7]) = -2.82842712474619
        val x = doubleArrayOf(1.0, 3.0)
        val y = doubleArrayOf(5.0, 7.0)
        assertEquals(-2.82842712474619, cohensD(x, y, pooled = true), tol, "min sample pooled")
        assertEquals(-2.82842712474619, cohensD(x, y, pooled = false), tol, "min sample unpooled")
    }

    @Test
    fun testCohensDIdenticalMeans() {
        // When means are identical, d should be exactly 0
        // scipy (manual): cohens_d_pooled([1,2,3,4,5], [1,2,3,4,5]) = 0.0
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(0.0, cohensD(x, y, pooled = true), tol, "pooled d for identical arrays")
        assertEquals(0.0, cohensD(x, y, pooled = false), tol, "unpooled d for identical arrays")
    }

    @Test
    fun testCohensDIdenticalMeansDifferentVariances() {
        // Same mean but different spreads: d should be 0
        val x = doubleArrayOf(4.0, 5.0, 6.0)
        val y = doubleArrayOf(1.0, 5.0, 9.0)
        assertEquals(0.0, cohensD(x, y, pooled = true), tol, "d = 0 when means equal")
        assertEquals(0.0, cohensD(x, y, pooled = false), tol, "d = 0 when means equal (unpooled)")
    }

    @Test
    fun testCohensDLargeSizeDifference() {
        // scipy (manual): cohens_d_pooled([1,2], [10,11,12,13,14,15,16,17,18,19]) = -4.51236587425405
        val x = doubleArrayOf(1.0, 2.0)
        val y = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0)
        assertEquals(-4.51236587425405, cohensD(x, y, pooled = true), tol, "pooled d with large size diff")
        // scipy (manual): cohens_d_unpooled = -5.91316473991545
        assertEquals(-5.91316473991545, cohensD(x, y, pooled = false), tol, "unpooled d with large size diff")
    }

    // ===== Degenerate input =====

    @Test
    fun testCohensDEmptyArrayX() {
        assertFailsWith<InsufficientDataException> {
            cohensD(doubleArrayOf(), doubleArrayOf(1.0, 2.0))
        }
    }

    @Test
    fun testCohensDEmptyArrayY() {
        assertFailsWith<InsufficientDataException> {
            cohensD(doubleArrayOf(1.0, 2.0), doubleArrayOf())
        }
    }

    @Test
    fun testCohensDSingleElementX() {
        assertFailsWith<InsufficientDataException> {
            cohensD(doubleArrayOf(1.0), doubleArrayOf(2.0, 3.0))
        }
    }

    @Test
    fun testCohensDSingleElementY() {
        assertFailsWith<InsufficientDataException> {
            cohensD(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0))
        }
    }

    @Test
    fun testCohensDSingleElementBoth() {
        assertFailsWith<InsufficientDataException> {
            cohensD(doubleArrayOf(1.0), doubleArrayOf(2.0))
        }
    }

    @Test
    fun testCohensDConstantArraysSameMean() {
        // Both arrays constant with same value: var=0, diff=0 -> 0/0 = NaN
        val x = doubleArrayOf(5.0, 5.0, 5.0)
        val y = doubleArrayOf(5.0, 5.0, 5.0)
        assertTrue(cohensD(x, y, pooled = true).isNaN(), "constant same mean -> NaN (pooled)")
        assertTrue(cohensD(x, y, pooled = false).isNaN(), "constant same mean -> NaN (unpooled)")
    }

    @Test
    fun testCohensDConstantArraysDifferentMeans() {
        // Both constant, different values: var=0, diff!=0 -> d/0 = +/-Inf
        val x = doubleArrayOf(5.0, 5.0, 5.0)
        val y = doubleArrayOf(3.0, 3.0, 3.0)
        val d = cohensD(x, y, pooled = true)
        assertEquals(Double.POSITIVE_INFINITY, d, "constant different means -> +Inf (pooled)")
    }

    @Test
    fun testCohensDOneConstantArray() {
        // scipy (manual): cohens_d_pooled([5,5,5], [1,2,3]) = 4.24264068711928
        val x = doubleArrayOf(5.0, 5.0, 5.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0)
        assertEquals(4.24264068711928, cohensD(x, y, pooled = true), tol, "one constant array pooled")
        assertEquals(4.24264068711928, cohensD(x, y, pooled = false), tol, "one constant array unpooled")
    }

    // ===== Extreme parameters =====

    @Test
    fun testCohensDLargeValues() {
        // scipy (manual): cohens_d_pooled([1e10, 1e10+1, 1e10+2], [1e10+100, 1e10+101, 1e10+102]) = -100.0
        val x = doubleArrayOf(1e10, 1e10 + 1.0, 1e10 + 2.0)
        val y = doubleArrayOf(1e10 + 100.0, 1e10 + 101.0, 1e10 + 102.0)
        assertEquals(-100.0, cohensD(x, y, pooled = true), tol, "large offset values pooled")
    }

    @Test
    fun testCohensDSmallValues() {
        // scipy (manual): cohens_d_pooled([1e-10, 2e-10, 3e-10], [4e-10, 5e-10, 6e-10]) = -3.0
        val x = doubleArrayOf(1e-10, 2e-10, 3e-10)
        val y = doubleArrayOf(4e-10, 5e-10, 6e-10)
        assertEquals(-3.0, cohensD(x, y, pooled = true), tol, "very small values pooled")
        assertEquals(-3.0, cohensD(x, y, pooled = false), tol, "very small values unpooled")
    }

    @Test
    fun testCohensDVeryLargeSample() {
        // Large arrays should not overflow or produce NaN
        val x = DoubleArray(10_000) { it.toDouble() }
        val y = DoubleArray(10_000) { it.toDouble() + 100.0 }
        val d = cohensD(x, y, pooled = true)
        assertTrue(d.isFinite(), "d should be finite for large arrays")
        assertTrue(d < 0.0, "d should be negative when x < y")
    }

    // ===== Non-finite input =====

    @Test
    fun testCohensDNaNInX() {
        // NaN in data should propagate to result
        val x = doubleArrayOf(1.0, Double.NaN, 3.0)
        val y = doubleArrayOf(4.0, 5.0, 6.0)
        assertTrue(cohensD(x, y).isNaN(), "NaN in x should propagate")
    }

    @Test
    fun testCohensDNaNInY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(4.0, Double.NaN, 6.0)
        assertTrue(cohensD(x, y).isNaN(), "NaN in y should propagate")
    }

    @Test
    fun testCohensDNaNInBoth() {
        val x = doubleArrayOf(Double.NaN, 2.0, 3.0)
        val y = doubleArrayOf(4.0, 5.0, Double.NaN)
        assertTrue(cohensD(x, y).isNaN(), "NaN in both arrays should propagate")
    }

    @Test
    fun testCohensDPositiveInfinityInX() {
        val x = doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0)
        val y = doubleArrayOf(4.0, 5.0, 6.0)
        val d = cohensD(x, y)
        // Infinity in mean -> Infinity - finite = Infinity, Infinity in variance -> NaN in sd
        assertTrue(d.isNaN() || d.isInfinite(), "Infinity in x should produce non-finite result")
    }

    @Test
    fun testCohensDNegativeInfinityInY() {
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(4.0, Double.NEGATIVE_INFINITY, 6.0)
        val d = cohensD(x, y)
        assertTrue(d.isNaN() || d.isInfinite(), "Negative Infinity in y should produce non-finite result")
    }

    // ===== Property-based tests =====

    @Test
    fun testCohensDSignReversal() {
        // Property: cohensD(x, y) = -cohensD(y, x)
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        assertEquals(
            cohensD(x, y, pooled = true),
            -cohensD(y, x, pooled = true),
            1e-14,
            "pooled: d(x,y) = -d(y,x)"
        )
        assertEquals(
            cohensD(x, y, pooled = false),
            -cohensD(y, x, pooled = false),
            1e-14,
            "unpooled: d(x,y) = -d(y,x)"
        )
    }

    @Test
    fun testCohensDSignReversalUnequalSizes() {
        // Sign reversal holds even with unequal sizes
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0)
        assertEquals(
            cohensD(x, y, pooled = true),
            -cohensD(y, x, pooled = true),
            1e-14,
            "sign reversal pooled, unequal sizes"
        )
        assertEquals(
            cohensD(x, y, pooled = false),
            -cohensD(y, x, pooled = false),
            1e-14,
            "sign reversal unpooled, unequal sizes"
        )
    }

    @Test
    fun testCohensDPooledEqualsUnpooledForEqualSizes() {
        // When n1 = n2, pooled and unpooled should give identical results
        // because ((n-1)*v1 + (n-1)*v2) / (2n-2) = (v1+v2)/2
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        assertEquals(
            cohensD(x, y, pooled = true),
            cohensD(x, y, pooled = false),
            1e-14,
            "pooled = unpooled when n1 = n2"
        )
    }

    @Test
    fun testCohensDPooledDiffersFromUnpooledForUnequalSizes() {
        // When n1 != n2 and variances differ, pooled != unpooled
        val x = doubleArrayOf(10.0, 12.0, 11.0, 13.0, 14.0, 15.0)
        val y = doubleArrayOf(8.0, 9.0, 7.0, 10.0)
        val pooled = cohensD(x, y, pooled = true)
        val unpooled = cohensD(x, y, pooled = false)
        assertTrue(
            abs(pooled - unpooled) > 0.01,
            "pooled ($pooled) and unpooled ($unpooled) should differ for unequal sizes with different variances"
        )
    }

    @Test
    fun testCohensDSignMatchesMeanDifference() {
        // Property: sign of d matches sign of (mean(x) - mean(y))
        val pairs = listOf(
            doubleArrayOf(10.0, 20.0, 30.0) to doubleArrayOf(1.0, 2.0, 3.0),   // positive d
            doubleArrayOf(1.0, 2.0, 3.0) to doubleArrayOf(10.0, 20.0, 30.0),   // negative d
        )
        for ((x, y) in pairs) {
            val d = cohensD(x, y)
            val meanDiff = x.average() - y.average()
            assertTrue(
                d * meanDiff >= 0.0,
                "sign of d ($d) should match sign of mean diff ($meanDiff)"
            )
        }
    }

    @Test
    fun testCohensDZeroWhenMeansEqual() {
        // Property: d = 0 whenever means are equal (regardless of variances)
        val x1 = doubleArrayOf(0.0, 10.0) // mean = 5
        val y1 = doubleArrayOf(4.0, 6.0)  // mean = 5
        assertEquals(0.0, cohensD(x1, y1, pooled = true), tol, "d = 0 for equal means (pooled)")
        assertEquals(0.0, cohensD(x1, y1, pooled = false), tol, "d = 0 for equal means (unpooled)")

        val x2 = doubleArrayOf(2.0, 4.0, 6.0)   // mean = 4
        val y2 = doubleArrayOf(1.0, 4.0, 7.0)    // mean = 4
        assertEquals(0.0, cohensD(x2, y2, pooled = true), tol, "d = 0 for equal means, different spread (pooled)")
    }

    @Test
    fun testCohensDScaleInvariance() {
        // Property: d(c*x, c*y) = d(x, y) for any positive constant c
        // Because scaling both arrays by c scales mean diff and sd equally
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val d = cohensD(x, y, pooled = true)
        for (c in listOf(0.001, 0.1, 10.0, 1000.0)) {
            val xScaled = DoubleArray(x.size) { x[it] * c }
            val yScaled = DoubleArray(y.size) { y[it] * c }
            assertEquals(d, cohensD(xScaled, yScaled, pooled = true), 1e-8, "scale invariance for c=$c")
        }
    }

    @Test
    fun testCohensDTranslationInvariance() {
        // Property: d(x + c, y + c) = d(x, y) for any constant c
        // Because shifting both arrays by c shifts both means equally, sd unchanged
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val d = cohensD(x, y, pooled = true)
        for (c in listOf(-1000.0, -1.0, 0.0, 1.0, 1000.0)) {
            val xShifted = DoubleArray(x.size) { x[it] + c }
            val yShifted = DoubleArray(y.size) { y[it] + c }
            assertEquals(d, cohensD(xShifted, yShifted, pooled = true), 1e-8, "translation invariance for c=$c")
        }
    }
}
