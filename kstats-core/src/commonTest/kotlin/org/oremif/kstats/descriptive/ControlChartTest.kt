package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class ControlChartTest {

    private val tol = 1e-10

    // ===== spcConstants: Basic correctness =====

    @Test
    fun testSpcConstantsN2() {
        // Montgomery "Introduction to Statistical Quality Control" (7th ed.), Appendix VI
        val c = spcConstants(2)
        assertEquals(1.880, c.a2, 0.0, "A2 for n=2")
        assertEquals(2.659, c.a3, 0.0, "A3 for n=2")
        assertEquals(0.000, c.d3, 0.0, "D3 for n=2")
        assertEquals(3.267, c.d4, 0.0, "D4 for n=2")
        assertEquals(0.000, c.b3, 0.0, "B3 for n=2")
        assertEquals(3.267, c.b4, 0.0, "B4 for n=2")
        assertEquals(0.7979, c.c4, 0.0, "c4 for n=2")
    }

    @Test
    fun testSpcConstantsN5() {
        // Montgomery Appendix VI, n=5
        val c = spcConstants(5)
        assertEquals(0.577, c.a2, 0.0, "A2 for n=5")
        assertEquals(1.427, c.a3, 0.0, "A3 for n=5")
        assertEquals(0.000, c.d3, 0.0, "D3 for n=5")
        assertEquals(2.114, c.d4, 0.0, "D4 for n=5")
        assertEquals(0.000, c.b3, 0.0, "B3 for n=5")
        assertEquals(2.089, c.b4, 0.0, "B4 for n=5")
        assertEquals(0.9400, c.c4, 0.0, "c4 for n=5")
    }

    @Test
    fun testSpcConstantsN10() {
        // Montgomery Appendix VI, n=10
        val c = spcConstants(10)
        assertEquals(0.308, c.a2, 0.0, "A2 for n=10")
        assertEquals(0.975, c.a3, 0.0, "A3 for n=10")
        assertEquals(0.223, c.d3, 0.0, "D3 for n=10")
        assertEquals(1.777, c.d4, 0.0, "D4 for n=10")
        assertEquals(0.284, c.b3, 0.0, "B3 for n=10")
        assertEquals(1.716, c.b4, 0.0, "B4 for n=10")
        assertEquals(0.9727, c.c4, 0.0, "c4 for n=10")
    }

    @Test
    fun testSpcConstantsN25() {
        // Montgomery Appendix VI, n=25 (maximum)
        val c = spcConstants(25)
        assertEquals(0.153, c.a2, 0.0, "A2 for n=25")
        assertEquals(0.606, c.a3, 0.0, "A3 for n=25")
        assertEquals(0.459, c.d3, 0.0, "D3 for n=25")
        assertEquals(1.541, c.d4, 0.0, "D4 for n=25")
        assertEquals(0.565, c.b3, 0.0, "B3 for n=25")
        assertEquals(1.435, c.b4, 0.0, "B4 for n=25")
        assertEquals(0.9896, c.c4, 0.0, "c4 for n=25")
    }

    // ===== spcConstants: Edge cases =====

    @Test
    fun testSpcConstantsBoundaryN2() {
        // n=2 is the minimum valid subgroup size
        val c = spcConstants(2)
        assertTrue(c.a2 > 0.0, "A2 should be positive for n=2")
        assertTrue(c.d4 > 0.0, "D4 should be positive for n=2")
    }

    @Test
    fun testSpcConstantsBoundaryN25() {
        // n=25 is the maximum valid subgroup size
        val c = spcConstants(25)
        assertTrue(c.a2 > 0.0, "A2 should be positive for n=25")
        assertTrue(c.c4 < 1.0, "c4 should be < 1 for n=25")
    }

    // ===== spcConstants: Degenerate input =====

    @Test
    fun testSpcConstantsN0() {
        assertFailsWith<InvalidParameterException> {
            spcConstants(0)
        }
    }

    @Test
    fun testSpcConstantsN1() {
        assertFailsWith<InvalidParameterException> {
            spcConstants(1)
        }
    }

    @Test
    fun testSpcConstantsN26() {
        assertFailsWith<InvalidParameterException> {
            spcConstants(26)
        }
    }

    @Test
    fun testSpcConstantsNegative() {
        assertFailsWith<InvalidParameterException> {
            spcConstants(-1)
        }
    }

    @Test
    fun testSpcConstantsLargeN() {
        assertFailsWith<InvalidParameterException> {
            spcConstants(100)
        }
    }

    // ===== spcConstants: Property-based =====

    @Test
    fun testSpcConstantsA2Decreasing() {
        // A2 should decrease as n increases (more data = tighter control limits)
        for (n in 2..24) {
            val current = spcConstants(n)
            val next = spcConstants(n + 1)
            assertTrue(
                current.a2 >= next.a2,
                "A2 should decrease: A2($n)=${current.a2} >= A2(${n + 1})=${next.a2}"
            )
        }
    }

    @Test
    fun testSpcConstantsC4Increasing() {
        // c4 should increase toward 1 as n increases
        for (n in 2..24) {
            val current = spcConstants(n)
            val next = spcConstants(n + 1)
            assertTrue(
                current.c4 <= next.c4,
                "c4 should increase: c4($n)=${current.c4} <= c4(${n + 1})=${next.c4}"
            )
        }
    }

    @Test
    fun testSpcConstantsD4Decreasing() {
        // D4 should decrease as n increases
        for (n in 2..24) {
            val current = spcConstants(n)
            val next = spcConstants(n + 1)
            assertTrue(
                current.d4 >= next.d4,
                "D4 should decrease: D4($n)=${current.d4} >= D4(${n + 1})=${next.d4}"
            )
        }
    }

    @Test
    fun testSpcConstantsD3NonNegative() {
        // D3 should be non-negative for all n
        for (n in 2..25) {
            val c = spcConstants(n)
            assertTrue(c.d3 >= 0.0, "D3 should be non-negative for n=$n")
        }
    }

    @Test
    fun testSpcConstantsB3NonNegative() {
        // B3 should be non-negative for all n
        for (n in 2..25) {
            val c = spcConstants(n)
            assertTrue(c.b3 >= 0.0, "B3 should be non-negative for n=$n")
        }
    }

    @Test
    fun testSpcConstantsC4LessThanOne() {
        // c4 should be strictly less than 1 for all finite n
        for (n in 2..25) {
            val c = spcConstants(n)
            assertTrue(c.c4 < 1.0, "c4 should be < 1 for n=$n, got ${c.c4}")
        }
    }

    @Test
    fun testSpcConstantsDataClassEquality() {
        val c1 = spcConstants(5)
        val c2 = spcConstants(5)
        assertEquals(c1, c2, "Same subgroup size should produce equal SpcConstants")
    }

    // ===== xBarRChart: Basic correctness =====

    @Test
    fun testXBarRChartKnownValues() {
        // numpy: 5 subgroups of size 4
        val subgroups = listOf(
            doubleArrayOf(72.0, 84.0, 79.0, 49.0),
            doubleArrayOf(56.0, 87.0, 33.0, 42.0),
            doubleArrayOf(55.0, 73.0, 22.0, 60.0),
            doubleArrayOf(44.0, 80.0, 54.0, 74.0),
            doubleArrayOf(97.0, 26.0, 48.0, 58.0),
        )
        val result = xBarRChart(subgroups)

        // numpy: xbar_bar = mean([71, 54.5, 52.5, 63, 57.25]) = 59.65
        assertEquals(59.65, result.centerLine, tol, "centerLine")
        // numpy: ucl = 59.65 + 0.729 * 49.4 = 95.6626
        assertEquals(95.6626, result.ucl, tol, "ucl")
        // numpy: lcl = 59.65 - 0.729 * 49.4 = 23.6374
        assertEquals(23.6374, result.lcl, tol, "lcl")

        // R-chart
        // numpy: r_bar = mean([35, 54, 51, 36, 71]) = 49.4
        assertEquals(49.4, result.rChart.centerLine, tol, "R-chart centerLine")
        // numpy: D4 * r_bar = 2.282 * 49.4 = 112.7308
        assertEquals(112.7308, result.rChart.ucl, tol, "R-chart ucl")
        // numpy: D3 * r_bar = 0.0 * 49.4 = 0
        assertEquals(0.0, result.rChart.lcl, tol, "R-chart lcl")
    }

    @Test
    fun testXBarRChartSimpleValues() {
        // numpy: 3 subgroups of size 5
        val subgroups = listOf(
            doubleArrayOf(10.0, 12.0, 11.0, 13.0, 9.0),
            doubleArrayOf(11.0, 10.0, 12.0, 11.0, 14.0),
            doubleArrayOf(9.0, 13.0, 10.0, 12.0, 11.0),
        )
        val result = xBarRChart(subgroups)

        // numpy: xbar_bar = mean([11.0, 11.6, 11.0]) = 11.2
        assertEquals(11.2, result.centerLine, tol, "centerLine")
        // numpy: ucl = 11.2 + 0.577 * 4.0 = 13.508
        assertEquals(13.508, result.ucl, tol, "ucl")
        // numpy: lcl = 11.2 - 0.577 * 4.0 = 8.892
        assertEquals(8.892, result.lcl, tol, "lcl")

        // R-chart: r_bar = mean([4, 4, 4]) = 4
        assertEquals(4.0, result.rChart.centerLine, tol, "R-chart centerLine")
        // numpy: D4 * 4 = 2.114 * 4 = 8.456
        assertEquals(8.456, result.rChart.ucl, tol, "R-chart ucl")
        assertEquals(0.0, result.rChart.lcl, tol, "R-chart lcl")
    }

    // ===== xBarRChart: Edge cases =====

    @Test
    fun testXBarRChartMinimumSubgroups() {
        // Minimum: 2 subgroups
        val subgroups = listOf(
            doubleArrayOf(10.0, 20.0),
            doubleArrayOf(15.0, 25.0),
        )
        val result = xBarRChart(subgroups)
        assertTrue(result.centerLine.isFinite(), "centerLine should be finite")
        assertTrue(result.ucl.isFinite(), "ucl should be finite")
        assertTrue(result.lcl.isFinite(), "lcl should be finite")
    }

    @Test
    fun testXBarRChartMinimumSubgroupSize() {
        // Minimum subgroup size: n=2
        val subgroups = listOf(
            doubleArrayOf(10.0, 20.0),
            doubleArrayOf(15.0, 25.0),
            doubleArrayOf(12.0, 18.0),
        )
        val result = xBarRChart(subgroups)

        // numpy: means=[15, 20, 15], xbar_bar=16.667
        assertEquals(50.0 / 3.0, result.centerLine, tol, "centerLine")
        // numpy: ranges=[10, 10, 6], r_bar=8.667
        assertEquals(26.0 / 3.0, result.rChart.centerLine, tol, "R-chart centerLine")
        // numpy: ucl = 16.667 + 1.880 * 8.667 = 32.96
        assertEquals(50.0 / 3.0 + 1.880 * 26.0 / 3.0, result.ucl, tol, "ucl")
    }

    @Test
    fun testXBarRChartMaxSubgroupSize() {
        // Maximum subgroup size: n=25
        // Use reproducible data
        val subgroups = listOf(
            DoubleArray(25) { 100.0 + it.toDouble() },
            DoubleArray(25) { 105.0 + it.toDouble() },
        )
        val result = xBarRChart(subgroups)
        assertTrue(result.centerLine.isFinite(), "centerLine should be finite for n=25")
        assertTrue(result.ucl > result.centerLine, "ucl > centerLine")
        assertTrue(result.lcl < result.centerLine, "lcl < centerLine")
    }

    @Test
    fun testXBarRChartConstantSubgroups() {
        // All subgroups have the same values => range = 0, std = 0
        val subgroups = listOf(
            doubleArrayOf(5.0, 5.0, 5.0),
            doubleArrayOf(5.0, 5.0, 5.0),
        )
        val result = xBarRChart(subgroups)

        assertEquals(5.0, result.centerLine, tol, "centerLine")
        // r_bar = 0, so ucl = lcl = centerLine
        assertEquals(5.0, result.ucl, tol, "ucl = centerLine when all constant")
        assertEquals(5.0, result.lcl, tol, "lcl = centerLine when all constant")
        assertEquals(0.0, result.rChart.centerLine, tol, "R-chart centerLine = 0")
        assertEquals(0.0, result.rChart.ucl, tol, "R-chart ucl = 0")
        assertEquals(0.0, result.rChart.lcl, tol, "R-chart lcl = 0")
    }

    // ===== xBarRChart: Degenerate input =====

    @Test
    fun testXBarRChartEmptyList() {
        assertFailsWith<InsufficientDataException> {
            xBarRChart(emptyList())
        }
    }

    @Test
    fun testXBarRChartSingleSubgroup() {
        assertFailsWith<InsufficientDataException> {
            xBarRChart(listOf(doubleArrayOf(1.0, 2.0, 3.0)))
        }
    }

    @Test
    fun testXBarRChartSubgroupSizeOne() {
        assertFailsWith<InsufficientDataException> {
            xBarRChart(listOf(doubleArrayOf(1.0), doubleArrayOf(2.0)))
        }
    }

    @Test
    fun testXBarRChartSubgroupSizeTooLarge() {
        assertFailsWith<InvalidParameterException> {
            xBarRChart(listOf(DoubleArray(26) { it.toDouble() }, DoubleArray(26) { it.toDouble() }))
        }
    }

    @Test
    fun testXBarRChartUnequalSubgroups() {
        assertFailsWith<InvalidParameterException> {
            xBarRChart(
                listOf(
                    doubleArrayOf(1.0, 2.0, 3.0),
                    doubleArrayOf(4.0, 5.0),
                )
            )
        }
    }

    @Test
    fun testXBarRChartUnequalSubgroupsThreeGroups() {
        assertFailsWith<InvalidParameterException> {
            xBarRChart(
                listOf(
                    doubleArrayOf(1.0, 2.0, 3.0),
                    doubleArrayOf(4.0, 5.0, 6.0),
                    doubleArrayOf(7.0, 8.0),
                )
            )
        }
    }

    // ===== xBarRChart: Extreme parameters =====

    @Test
    fun testXBarRChartLargeOffsetData() {
        // Test numerical stability with large offset data
        val offset = 1e12
        val subgroups = listOf(
            doubleArrayOf(offset + 1.0, offset + 2.0, offset + 3.0),
            doubleArrayOf(offset + 2.0, offset + 3.0, offset + 4.0),
            doubleArrayOf(offset + 1.5, offset + 2.5, offset + 3.5),
        )
        val result = xBarRChart(subgroups)

        // numpy: xbar_bar = 1e12 + 2.5
        assertEquals(offset + 2.5, result.centerLine, 1e-2, "centerLine with large offset")
        // numpy: r_bar = 2.0
        assertEquals(2.0, result.rChart.centerLine, tol, "R-chart centerLine with large offset")
        assertTrue(result.ucl.isFinite(), "ucl should be finite with large offset")
        assertTrue(result.lcl.isFinite(), "lcl should be finite with large offset")
    }

    @Test
    fun testXBarRChartLargeValues() {
        // Very large measurement values
        val subgroups = listOf(
            doubleArrayOf(1e10, 1.1e10, 0.9e10),
            doubleArrayOf(1.05e10, 0.95e10, 1.0e10),
        )
        val result = xBarRChart(subgroups)
        assertTrue(result.centerLine.isFinite(), "centerLine should be finite for large values")
        assertTrue(result.ucl > result.lcl, "ucl > lcl for large values")
        assertTrue(result.rChart.centerLine > 0.0, "R-chart centerLine > 0")
    }

    @Test
    fun testXBarRChartVerySmallValues() {
        // Very small measurement values
        val subgroups = listOf(
            doubleArrayOf(1e-10, 2e-10, 3e-10),
            doubleArrayOf(1.5e-10, 2.5e-10, 3.5e-10),
        )
        val result = xBarRChart(subgroups)
        assertTrue(result.centerLine.isFinite(), "centerLine should be finite for small values")
        assertTrue(result.ucl > result.lcl, "ucl > lcl for small values")
    }

    @Test
    fun testXBarRChartManySubgroups() {
        // Many subgroups (stress test for mean computation)
        val subgroups = List(100) { doubleArrayOf(10.0 + it * 0.01, 11.0 + it * 0.01, 12.0 + it * 0.01) }
        val result = xBarRChart(subgroups)
        assertTrue(result.centerLine.isFinite(), "centerLine should be finite for many subgroups")
        assertTrue(result.rChart.centerLine > 0.0, "R-chart centerLine > 0 for many subgroups")
    }

    // ===== xBarRChart: Non-finite input =====

    @Test
    fun testXBarRChartNaNInData() {
        // NaN in data should propagate through mean and range
        val subgroups = listOf(
            doubleArrayOf(1.0, Double.NaN, 3.0),
            doubleArrayOf(4.0, 5.0, 6.0),
        )
        val result = xBarRChart(subgroups)
        assertTrue(result.centerLine.isNaN(), "centerLine should be NaN when data contains NaN")
        assertTrue(result.ucl.isNaN(), "ucl should be NaN when data contains NaN")
    }

    @Test
    fun testXBarRChartInfinityInData() {
        // Infinity in data
        val subgroups = listOf(
            doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0),
            doubleArrayOf(4.0, 5.0, 6.0),
        )
        val result = xBarRChart(subgroups)
        // mean with infinity -> infinity, range with infinity -> infinity
        // Various NaN/Infinity propagation patterns are acceptable
        assertTrue(
            !result.centerLine.isFinite() || result.centerLine.isNaN(),
            "centerLine should be non-finite when data contains Infinity"
        )
    }

    @Test
    fun testXBarRChartNegativeInfinityInData() {
        val subgroups = listOf(
            doubleArrayOf(Double.NEGATIVE_INFINITY, 5.0, 6.0),
            doubleArrayOf(4.0, 5.0, 6.0),
        )
        val result = xBarRChart(subgroups)
        assertTrue(
            !result.centerLine.isFinite() || result.centerLine.isNaN(),
            "centerLine should be non-finite when data contains -Infinity"
        )
    }

    // ===== xBarRChart: Property-based =====

    @Test
    fun testXBarRChartUclGtLcl() {
        // UCL should always be >= LCL (equals when r_bar = 0)
        val datasets = listOf(
            listOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(2.0, 3.0, 4.0)),
            listOf(doubleArrayOf(10.0, 20.0), doubleArrayOf(15.0, 25.0)),
            listOf(
                doubleArrayOf(100.0, 101.0, 102.0, 103.0, 104.0),
                doubleArrayOf(99.0, 100.0, 101.0, 102.0, 103.0),
                doubleArrayOf(100.5, 101.5, 102.5, 103.5, 104.5),
            ),
        )
        for ((i, subgroups) in datasets.withIndex()) {
            val result = xBarRChart(subgroups)
            assertTrue(result.ucl >= result.lcl, "ucl >= lcl for dataset $i")
        }
    }

    @Test
    fun testXBarRChartCenterLineBetweenLimits() {
        // Center line should be between LCL and UCL
        val subgroups = listOf(
            doubleArrayOf(72.0, 84.0, 79.0, 49.0),
            doubleArrayOf(56.0, 87.0, 33.0, 42.0),
            doubleArrayOf(55.0, 73.0, 22.0, 60.0),
        )
        val result = xBarRChart(subgroups)
        assertTrue(
            result.centerLine >= result.lcl,
            "centerLine (${result.centerLine}) >= lcl (${result.lcl})"
        )
        assertTrue(
            result.centerLine <= result.ucl,
            "centerLine (${result.centerLine}) <= ucl (${result.ucl})"
        )
    }

    @Test
    fun testXBarRChartSymmetricLimits() {
        // UCL - centerLine should equal centerLine - LCL
        val subgroups = listOf(
            doubleArrayOf(10.0, 12.0, 11.0, 13.0, 9.0),
            doubleArrayOf(11.0, 10.0, 12.0, 11.0, 14.0),
        )
        val result = xBarRChart(subgroups)
        val upperSpread = result.ucl - result.centerLine
        val lowerSpread = result.centerLine - result.lcl
        assertEquals(upperSpread, lowerSpread, tol, "X-bar limits should be symmetric")
    }

    @Test
    fun testXBarRChartRChartLclNonNegative() {
        // R-chart LCL should be non-negative (ranges are non-negative)
        val subgroups = listOf(
            doubleArrayOf(1.0, 100.0, 50.0),
            doubleArrayOf(10.0, 90.0, 45.0),
        )
        val result = xBarRChart(subgroups)
        assertTrue(
            result.rChart.lcl >= 0.0,
            "R-chart lcl should be non-negative, got ${result.rChart.lcl}"
        )
    }

    @Test
    fun testXBarRChartRChartCenterLineNonNegative() {
        // R-chart center line (mean of ranges) should be non-negative
        val subgroups = listOf(
            doubleArrayOf(1.0, 2.0, 3.0),
            doubleArrayOf(4.0, 5.0, 6.0),
            doubleArrayOf(7.0, 8.0, 9.0),
        )
        val result = xBarRChart(subgroups)
        assertTrue(
            result.rChart.centerLine >= 0.0,
            "R-chart centerLine should be non-negative"
        )
    }

    @Test
    fun testXBarRChartScaleInvariance() {
        // Multiplying all data by a constant c should scale limits by c
        val subgroups = listOf(
            doubleArrayOf(10.0, 12.0, 11.0),
            doubleArrayOf(11.0, 10.0, 13.0),
        )
        val scaled = subgroups.map { sg -> DoubleArray(sg.size) { sg[it] * 3.0 } }

        val result1 = xBarRChart(subgroups)
        val result2 = xBarRChart(scaled)

        assertEquals(result1.centerLine * 3.0, result2.centerLine, tol, "centerLine scales linearly")
        assertEquals(result1.ucl * 3.0, result2.ucl, tol, "ucl scales linearly")
        assertEquals(result1.lcl * 3.0, result2.lcl, tol, "lcl scales linearly")
        assertEquals(result1.rChart.centerLine * 3.0, result2.rChart.centerLine, tol, "R-chart centerLine scales linearly")
    }

    @Test
    fun testXBarRChartTranslationInvariance() {
        // Shifting all data by constant c: centerLine shifts by c, rChart unchanged
        val subgroups = listOf(
            doubleArrayOf(10.0, 12.0, 11.0),
            doubleArrayOf(11.0, 10.0, 13.0),
        )
        val c = 1000.0
        val shifted = subgroups.map { sg -> DoubleArray(sg.size) { sg[it] + c } }

        val result1 = xBarRChart(subgroups)
        val result2 = xBarRChart(shifted)

        assertEquals(result1.centerLine + c, result2.centerLine, tol, "centerLine shifts by c")
        assertEquals(result1.rChart.centerLine, result2.rChart.centerLine, tol, "R-chart centerLine unchanged by shift")
        assertEquals(result1.rChart.ucl, result2.rChart.ucl, tol, "R-chart ucl unchanged by shift")
    }

    @Test
    fun testXBarRChartDataClassEquality() {
        val subgroups = listOf(
            doubleArrayOf(10.0, 12.0, 11.0),
            doubleArrayOf(11.0, 10.0, 13.0),
        )
        val r1 = xBarRChart(subgroups)
        val r2 = xBarRChart(subgroups)
        assertEquals(r1, r2, "Same input should produce equal XBarRChartResult")
    }

    // ===== xBarSChart: Basic correctness =====

    @Test
    fun testXBarSChartKnownValues() {
        // numpy: 5 subgroups of size 4
        val subgroups = listOf(
            doubleArrayOf(72.0, 84.0, 79.0, 49.0),
            doubleArrayOf(56.0, 87.0, 33.0, 42.0),
            doubleArrayOf(55.0, 73.0, 22.0, 60.0),
            doubleArrayOf(44.0, 80.0, 54.0, 74.0),
            doubleArrayOf(97.0, 26.0, 48.0, 58.0),
        )
        val result = xBarSChart(subgroups)

        // numpy: xbar_bar = 59.65
        assertEquals(59.65, result.centerLine, tol, "centerLine")
        // numpy: s_bar = 21.4697313967827
        assertEquals(21.4697313967827, result.sChart.centerLine, 1e-7, "S-chart centerLine")
        // numpy: ucl = 59.65 + 1.628 * 21.4697313967827 = 94.6027227139622
        assertEquals(94.6027227139622, result.ucl, 1e-4, "ucl")
        // numpy: lcl = 59.65 - 1.628 * 21.4697313967827 = 24.6972772860378
        assertEquals(24.6972772860378, result.lcl, 1e-4, "lcl")
        // numpy: B4 * s_bar = 2.266 * 21.4697313967827 = 48.6504113451096
        assertEquals(48.6504113451096, result.sChart.ucl, 1e-4, "S-chart ucl")
        // numpy: B3 * s_bar = 0.0 * 21.4697313967827 = 0
        assertEquals(0.0, result.sChart.lcl, tol, "S-chart lcl")
    }

    @Test
    fun testXBarSChartSimpleValues() {
        // numpy: 3 subgroups of size 5
        val subgroups = listOf(
            doubleArrayOf(10.0, 12.0, 11.0, 13.0, 9.0),
            doubleArrayOf(11.0, 10.0, 12.0, 11.0, 14.0),
            doubleArrayOf(9.0, 13.0, 10.0, 12.0, 11.0),
        )
        val result = xBarSChart(subgroups)

        // numpy: xbar_bar = 11.2
        assertEquals(11.2, result.centerLine, tol, "centerLine")
        // numpy: s_bar = 1.5596175829929
        assertEquals(1.5596175829929, result.sChart.centerLine, 1e-7, "S-chart centerLine")
        // numpy: ucl = 11.2 + 1.427 * 1.5596175829929 = 13.4255742909309
        assertEquals(13.4255742909309, result.ucl, 1e-4, "ucl")
        // numpy: lcl = 11.2 - 1.427 * 1.5596175829929 = 8.97442570906914
        assertEquals(8.97442570906914, result.lcl, 1e-4, "lcl")
    }

    // ===== xBarSChart: Edge cases =====

    @Test
    fun testXBarSChartMinimumSubgroups() {
        // Minimum: 2 subgroups
        val subgroups = listOf(
            doubleArrayOf(10.0, 20.0),
            doubleArrayOf(15.0, 25.0),
        )
        val result = xBarSChart(subgroups)
        assertTrue(result.centerLine.isFinite(), "centerLine should be finite")
        assertTrue(result.ucl.isFinite(), "ucl should be finite")
        assertTrue(result.lcl.isFinite(), "lcl should be finite")
    }

    @Test
    fun testXBarSChartMinimumSubgroupSize() {
        // Minimum subgroup size: n=2
        val subgroups = listOf(
            doubleArrayOf(10.0, 20.0),
            doubleArrayOf(15.0, 25.0),
            doubleArrayOf(12.0, 18.0),
        )
        val result = xBarSChart(subgroups)

        assertEquals(50.0 / 3.0, result.centerLine, tol, "centerLine")
        assertTrue(result.sChart.centerLine > 0.0, "S-chart centerLine > 0")
    }

    @Test
    fun testXBarSChartConstantSubgroups() {
        // All subgroups have the same values => std = 0
        val subgroups = listOf(
            doubleArrayOf(5.0, 5.0, 5.0),
            doubleArrayOf(5.0, 5.0, 5.0),
        )
        val result = xBarSChart(subgroups)

        assertEquals(5.0, result.centerLine, tol, "centerLine")
        assertEquals(5.0, result.ucl, tol, "ucl = centerLine when all constant")
        assertEquals(5.0, result.lcl, tol, "lcl = centerLine when all constant")
        assertEquals(0.0, result.sChart.centerLine, tol, "S-chart centerLine = 0")
        assertEquals(0.0, result.sChart.ucl, tol, "S-chart ucl = 0")
        assertEquals(0.0, result.sChart.lcl, tol, "S-chart lcl = 0")
    }

    // ===== xBarSChart: Degenerate input =====

    @Test
    fun testXBarSChartEmptyList() {
        assertFailsWith<InsufficientDataException> {
            xBarSChart(emptyList())
        }
    }

    @Test
    fun testXBarSChartSingleSubgroup() {
        assertFailsWith<InsufficientDataException> {
            xBarSChart(listOf(doubleArrayOf(1.0, 2.0, 3.0)))
        }
    }

    @Test
    fun testXBarSChartSubgroupSizeOne() {
        assertFailsWith<InsufficientDataException> {
            xBarSChart(listOf(doubleArrayOf(1.0), doubleArrayOf(2.0)))
        }
    }

    @Test
    fun testXBarSChartSubgroupSizeTooLarge() {
        assertFailsWith<InvalidParameterException> {
            xBarSChart(listOf(DoubleArray(26) { it.toDouble() }, DoubleArray(26) { it.toDouble() }))
        }
    }

    @Test
    fun testXBarSChartUnequalSubgroups() {
        assertFailsWith<InvalidParameterException> {
            xBarSChart(
                listOf(
                    doubleArrayOf(1.0, 2.0, 3.0),
                    doubleArrayOf(4.0, 5.0),
                )
            )
        }
    }

    // ===== xBarSChart: Extreme parameters =====

    @Test
    fun testXBarSChartLargeOffsetData() {
        val offset = 1e12
        val subgroups = listOf(
            doubleArrayOf(offset + 1.0, offset + 2.0, offset + 3.0),
            doubleArrayOf(offset + 2.0, offset + 3.0, offset + 4.0),
            doubleArrayOf(offset + 1.5, offset + 2.5, offset + 3.5),
        )
        val result = xBarSChart(subgroups)

        assertEquals(offset + 2.5, result.centerLine, 1e-2, "centerLine with large offset")
        assertTrue(result.sChart.centerLine > 0.0, "S-chart centerLine > 0")
        assertTrue(result.ucl.isFinite(), "ucl should be finite with large offset")
        assertTrue(result.lcl.isFinite(), "lcl should be finite with large offset")
    }

    @Test
    fun testXBarSChartVerySmallValues() {
        val subgroups = listOf(
            doubleArrayOf(1e-10, 2e-10, 3e-10),
            doubleArrayOf(1.5e-10, 2.5e-10, 3.5e-10),
        )
        val result = xBarSChart(subgroups)
        assertTrue(result.centerLine.isFinite(), "centerLine should be finite for small values")
        assertTrue(result.ucl > result.lcl, "ucl > lcl for small values")
    }

    // ===== xBarSChart: Non-finite input =====

    @Test
    fun testXBarSChartNaNInData() {
        val subgroups = listOf(
            doubleArrayOf(1.0, Double.NaN, 3.0),
            doubleArrayOf(4.0, 5.0, 6.0),
        )
        val result = xBarSChart(subgroups)
        assertTrue(result.centerLine.isNaN(), "centerLine should be NaN when data contains NaN")
        assertTrue(result.ucl.isNaN(), "ucl should be NaN when data contains NaN")
    }

    @Test
    fun testXBarSChartInfinityInData() {
        val subgroups = listOf(
            doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0),
            doubleArrayOf(4.0, 5.0, 6.0),
        )
        val result = xBarSChart(subgroups)
        assertTrue(
            !result.centerLine.isFinite() || result.centerLine.isNaN(),
            "centerLine should be non-finite when data contains Infinity"
        )
    }

    // ===== xBarSChart: Property-based =====

    @Test
    fun testXBarSChartUclGtLcl() {
        val datasets = listOf(
            listOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(2.0, 3.0, 4.0)),
            listOf(doubleArrayOf(10.0, 20.0), doubleArrayOf(15.0, 25.0)),
            listOf(
                doubleArrayOf(100.0, 101.0, 102.0, 103.0, 104.0),
                doubleArrayOf(99.0, 100.0, 101.0, 102.0, 103.0),
            ),
        )
        for ((i, subgroups) in datasets.withIndex()) {
            val result = xBarSChart(subgroups)
            assertTrue(result.ucl >= result.lcl, "ucl >= lcl for dataset $i")
        }
    }

    @Test
    fun testXBarSChartCenterLineBetweenLimits() {
        val subgroups = listOf(
            doubleArrayOf(72.0, 84.0, 79.0, 49.0),
            doubleArrayOf(56.0, 87.0, 33.0, 42.0),
            doubleArrayOf(55.0, 73.0, 22.0, 60.0),
        )
        val result = xBarSChart(subgroups)
        assertTrue(
            result.centerLine >= result.lcl,
            "centerLine (${result.centerLine}) >= lcl (${result.lcl})"
        )
        assertTrue(
            result.centerLine <= result.ucl,
            "centerLine (${result.centerLine}) <= ucl (${result.ucl})"
        )
    }

    @Test
    fun testXBarSChartSymmetricLimits() {
        val subgroups = listOf(
            doubleArrayOf(10.0, 12.0, 11.0, 13.0, 9.0),
            doubleArrayOf(11.0, 10.0, 12.0, 11.0, 14.0),
        )
        val result = xBarSChart(subgroups)
        val upperSpread = result.ucl - result.centerLine
        val lowerSpread = result.centerLine - result.lcl
        assertEquals(upperSpread, lowerSpread, tol, "X-bar limits should be symmetric")
    }

    @Test
    fun testXBarSChartSChartLclNonNegative() {
        val subgroups = listOf(
            doubleArrayOf(1.0, 100.0, 50.0),
            doubleArrayOf(10.0, 90.0, 45.0),
        )
        val result = xBarSChart(subgroups)
        assertTrue(
            result.sChart.lcl >= 0.0,
            "S-chart lcl should be non-negative, got ${result.sChart.lcl}"
        )
    }

    @Test
    fun testXBarSChartScaleInvariance() {
        val subgroups = listOf(
            doubleArrayOf(10.0, 12.0, 11.0),
            doubleArrayOf(11.0, 10.0, 13.0),
        )
        val factor = 5.0
        val scaled = subgroups.map { sg -> DoubleArray(sg.size) { sg[it] * factor } }

        val result1 = xBarSChart(subgroups)
        val result2 = xBarSChart(scaled)

        assertEquals(result1.centerLine * factor, result2.centerLine, tol, "centerLine scales linearly")
        assertEquals(result1.sChart.centerLine * factor, result2.sChart.centerLine, 1e-6, "S-chart centerLine scales linearly")
    }

    @Test
    fun testXBarSChartTranslationInvariance() {
        val subgroups = listOf(
            doubleArrayOf(10.0, 12.0, 11.0),
            doubleArrayOf(11.0, 10.0, 13.0),
        )
        val c = 500.0
        val shifted = subgroups.map { sg -> DoubleArray(sg.size) { sg[it] + c } }

        val result1 = xBarSChart(subgroups)
        val result2 = xBarSChart(shifted)

        assertEquals(result1.centerLine + c, result2.centerLine, tol, "centerLine shifts by c")
        assertEquals(result1.sChart.centerLine, result2.sChart.centerLine, 1e-6, "S-chart centerLine unchanged by shift")
    }

    @Test
    fun testXBarSChartDataClassEquality() {
        val subgroups = listOf(
            doubleArrayOf(10.0, 12.0, 11.0),
            doubleArrayOf(11.0, 10.0, 13.0),
        )
        val r1 = xBarSChart(subgroups)
        val r2 = xBarSChart(subgroups)
        assertEquals(r1, r2, "Same input should produce equal XBarSChartResult")
    }

    // ===== Cross-chart consistency =====

    @Test
    fun testXBarRAndXBarSShareCenterLine() {
        // Both charts should have the same x-bar center line (grand mean)
        val subgroups = listOf(
            doubleArrayOf(72.0, 84.0, 79.0, 49.0),
            doubleArrayOf(56.0, 87.0, 33.0, 42.0),
            doubleArrayOf(55.0, 73.0, 22.0, 60.0),
        )
        val rResult = xBarRChart(subgroups)
        val sResult = xBarSChart(subgroups)

        assertEquals(
            rResult.centerLine,
            sResult.centerLine,
            tol,
            "X-bar center lines should be identical"
        )
    }

    @Test
    fun testControlChartLimitsDataClass() {
        // Test the ControlChartLimits data class
        val limits = ControlChartLimits(centerLine = 50.0, ucl = 60.0, lcl = 40.0)
        assertEquals(50.0, limits.centerLine, 0.0)
        assertEquals(60.0, limits.ucl, 0.0)
        assertEquals(40.0, limits.lcl, 0.0)

        // Copy
        val modified = limits.copy(ucl = 70.0)
        assertEquals(70.0, modified.ucl, 0.0)
        assertEquals(50.0, modified.centerLine, 0.0)
    }

    @Test
    fun testXBarRChartResultDataClass() {
        val rChart = ControlChartLimits(centerLine = 5.0, ucl = 10.0, lcl = 0.0)
        val result = XBarRChartResult(centerLine = 50.0, ucl = 55.0, lcl = 45.0, rChart = rChart)
        assertEquals(50.0, result.centerLine, 0.0)
        assertEquals(55.0, result.ucl, 0.0)
        assertEquals(45.0, result.lcl, 0.0)
        assertEquals(rChart, result.rChart)
    }

    @Test
    fun testXBarSChartResultDataClass() {
        val sChart = ControlChartLimits(centerLine = 2.0, ucl = 4.0, lcl = 0.0)
        val result = XBarSChartResult(centerLine = 50.0, ucl = 55.0, lcl = 45.0, sChart = sChart)
        assertEquals(50.0, result.centerLine, 0.0)
        assertEquals(55.0, result.ucl, 0.0)
        assertEquals(45.0, result.lcl, 0.0)
        assertEquals(sChart, result.sChart)
    }

    // ===== cusum: Basic correctness =====

    @Test
    fun testCusumIssueExample() {
        // Hand-computed CUSUM from issue #36 example:
        // observations = [10.1, 10.3, 10.5, 10.8, 11.0, 11.3], target=10.0, k=0.25, h=4.0
        // C+[0] = max(0, 0 + (10.1 - 10.0 - 0.25)) = max(0, -0.15) = 0
        // C+[1] = max(0, 0 + (10.3 - 10.25)) = 0.05
        // C+[2] = max(0, 0.05 + (10.5 - 10.25)) = 0.30
        // C+[3] = max(0, 0.30 + (10.8 - 10.25)) = 0.85
        // C+[4] = max(0, 0.85 + (11.0 - 10.25)) = 1.60
        // C+[5] = max(0, 1.60 + (11.3 - 10.25)) = 2.65
        // C-[i] = 0 for all i (all observations above target - k = 9.75)
        // alarmIndex = -1 (C+ never exceeds 4.0)
        val obs = doubleArrayOf(10.1, 10.3, 10.5, 10.8, 11.0, 11.3)
        val result = cusum(obs, target = 10.0, k = 0.25, h = 4.0)

        // numpy: manually computed as above
        assertEquals(0.0, result.sPlus[0], tol, "sPlus[0]")
        assertEquals(0.05, result.sPlus[1], tol, "sPlus[1]")
        assertEquals(0.30, result.sPlus[2], tol, "sPlus[2]")
        assertEquals(0.85, result.sPlus[3], tol, "sPlus[3]")
        assertEquals(1.60, result.sPlus[4], tol, "sPlus[4]")
        assertEquals(2.65, result.sPlus[5], tol, "sPlus[5]")
        for (i in obs.indices) {
            assertEquals(0.0, result.sMinus[i], tol, "sMinus[$i]")
        }
        assertEquals(-1, result.alarmIndex, "alarmIndex")
    }

    @Test
    fun testCusumMontgomeryTable92() {
        // Montgomery "Introduction to Statistical Quality Control" (7th ed.), §9.1.1, Table 9.2, p.416
        // 30 observations from a process with mu_0=10, sigma=1, monitored with K=0.5, H=5
        // The process shifts upward around observation 20; CUSUM detects the shift
        // numpy: canonical 30-observation Montgomery CUSUM example
        val obs = doubleArrayOf(
            9.45, 7.99, 9.29, 11.66, 12.16, 10.18, 8.04, 11.46, 9.20, 10.34,
            9.03, 11.47, 10.51, 9.40, 10.08, 9.37, 10.62, 10.31, 8.52, 10.84,
            10.90, 9.33, 12.29, 11.50, 10.60, 11.08, 10.38, 11.62, 11.31, 10.52
        )
        val result = cusum(obs, target = 10.0, k = 0.5, h = 5.0)

        // numpy: last values should be sPlus[29] ≈ 5.30, sMinus[29] ≈ 0
        assertEquals(5.30, result.sPlus[29], tol, "sPlus[29]")
        assertEquals(0.0, result.sMinus[29], tol, "sMinus[29]")
        // numpy: alarm at i=28 (sPlus[28]=5.28 > 5, first violation is actually at 28)
        assertEquals(28, result.alarmIndex, "alarmIndex (Montgomery example)")

        // numpy: spot-check a few mid-series values
        assertEquals(1.16, result.sPlus[3], tol, "sPlus[3]")
        assertEquals(2.82, result.sPlus[4], tol, "sPlus[4]")
        assertEquals(1.56, result.sMinus[1], tol, "sMinus[1]")
        assertEquals(1.77, result.sMinus[2], tol, "sMinus[2]")
    }

    @Test
    fun testCusumNoAlarm() {
        // Observations all near target, variations well below h
        // numpy: cusum([10.05, 9.95, 10.03, 9.97, 10.02, 9.98], 10.0, 0.5, 5.0)
        val obs = doubleArrayOf(10.05, 9.95, 10.03, 9.97, 10.02, 9.98)
        val result = cusum(obs, target = 10.0, k = 0.5, h = 5.0)

        for (i in obs.indices) {
            assertEquals(0.0, result.sPlus[i], tol, "sPlus[$i] near target")
            assertEquals(0.0, result.sMinus[i], tol, "sMinus[$i] near target")
        }
        assertEquals(-1, result.alarmIndex, "alarmIndex should be -1")
    }

    @Test
    fun testCusumUpperAlarm() {
        // Upward drift: [10.2, 10.4, ..., 12.0]; alarm at i=6 (sPlus=3.5>3)
        // numpy: cusum([10.2, 10.4, 10.6, 10.9, 11.2, 11.5, 11.8, 12.0], 10.0, 0.5, 3.0)
        val obs = doubleArrayOf(10.2, 10.4, 10.6, 10.9, 11.2, 11.5, 11.8, 12.0)
        val result = cusum(obs, target = 10.0, k = 0.5, h = 3.0)

        // numpy: sPlus=[0, 0, 0.1, 0.5, 1.2, 2.2, 3.5, 5.0]
        assertEquals(0.0, result.sPlus[0], tol, "sPlus[0]")
        assertEquals(0.0, result.sPlus[1], tol, "sPlus[1]")
        assertEquals(0.1, result.sPlus[2], tol, "sPlus[2]")
        assertEquals(0.5, result.sPlus[3], tol, "sPlus[3]")
        assertEquals(1.2, result.sPlus[4], tol, "sPlus[4]")
        assertEquals(2.2, result.sPlus[5], tol, "sPlus[5]")
        assertEquals(3.5, result.sPlus[6], tol, "sPlus[6]")
        assertEquals(5.0, result.sPlus[7], tol, "sPlus[7]")
        for (i in obs.indices) {
            assertEquals(0.0, result.sMinus[i], tol, "sMinus[$i]")
        }
        assertEquals(6, result.alarmIndex, "alarmIndex should be 6 (first sPlus>3)")
    }

    @Test
    fun testCusumLowerAlarm() {
        // Downward drift: mirror of upper-alarm case, swaps sPlus and sMinus
        // numpy: cusum([9.8, 9.6, 9.4, 9.1, 8.8, 8.5, 8.2, 8.0], 10.0, 0.5, 3.0)
        val obs = doubleArrayOf(9.8, 9.6, 9.4, 9.1, 8.8, 8.5, 8.2, 8.0)
        val result = cusum(obs, target = 10.0, k = 0.5, h = 3.0)

        // numpy: sMinus=[0, 0, 0.1, 0.5, 1.2, 2.2, 3.5, 5.0]
        assertEquals(0.0, result.sMinus[0], tol, "sMinus[0]")
        assertEquals(0.0, result.sMinus[1], tol, "sMinus[1]")
        assertEquals(0.1, result.sMinus[2], tol, "sMinus[2]")
        assertEquals(0.5, result.sMinus[3], tol, "sMinus[3]")
        assertEquals(1.2, result.sMinus[4], tol, "sMinus[4]")
        assertEquals(2.2, result.sMinus[5], tol, "sMinus[5]")
        assertEquals(3.5, result.sMinus[6], tol, "sMinus[6]")
        assertEquals(5.0, result.sMinus[7], tol, "sMinus[7]")
        for (i in obs.indices) {
            assertEquals(0.0, result.sPlus[i], tol, "sPlus[$i]")
        }
        assertEquals(6, result.alarmIndex, "alarmIndex should be 6 (first sMinus>3)")
    }

    @Test
    fun testCusumAlarmAtFirstObservation() {
        // Very strong upward shift at i=0: x-target-k = 20-10-0.5 = 9.5 > 5
        // numpy: cusum([20.0, 10.0, 10.0], 10.0, 0.5, 5.0)
        val obs = doubleArrayOf(20.0, 10.0, 10.0)
        val result = cusum(obs, target = 10.0, k = 0.5, h = 5.0)

        // numpy: sPlus=[9.5, 9.0, 8.5]; sPlus[0]=9.5 > h=5
        assertEquals(9.5, result.sPlus[0], tol, "sPlus[0]")
        assertEquals(9.0, result.sPlus[1], tol, "sPlus[1]")
        assertEquals(8.5, result.sPlus[2], tol, "sPlus[2]")
        assertEquals(0, result.alarmIndex, "alarmIndex should be 0")
    }

    // ===== cusum: Edge cases =====

    @Test
    fun testCusumSingleObservationAboveTarget() {
        // Single observation with x > target + k
        // numpy: sPlus[0] = max(0, 12-10-0.5) = 1.5, sMinus[0] = 0
        val result = cusum(doubleArrayOf(12.0), target = 10.0, k = 0.5, h = 5.0)

        assertEquals(1, result.sPlus.size, "sPlus.size")
        assertEquals(1, result.sMinus.size, "sMinus.size")
        assertEquals(1.5, result.sPlus[0], tol, "sPlus[0]")
        assertEquals(0.0, result.sMinus[0], tol, "sMinus[0]")
        assertEquals(-1, result.alarmIndex, "no alarm (1.5 < 5.0)")
    }

    @Test
    fun testCusumSingleObservationBelowTarget() {
        // Single observation with x < target - k
        // numpy: sPlus[0] = 0, sMinus[0] = max(0, 10-0.5-7) = 2.5
        val result = cusum(doubleArrayOf(7.0), target = 10.0, k = 0.5, h = 5.0)

        assertEquals(0.0, result.sPlus[0], tol, "sPlus[0]")
        assertEquals(2.5, result.sMinus[0], tol, "sMinus[0]")
        assertEquals(-1, result.alarmIndex, "no alarm (2.5 < 5.0)")
    }

    @Test
    fun testCusumSingleObservationAtTarget() {
        // Single observation equal to target: both zero
        val result = cusum(doubleArrayOf(10.0), target = 10.0, k = 0.5, h = 5.0)

        assertEquals(0.0, result.sPlus[0], tol, "sPlus[0]")
        assertEquals(0.0, result.sMinus[0], tol, "sMinus[0]")
        assertEquals(-1, result.alarmIndex, "no alarm")
    }

    @Test
    fun testCusumAllObservationsEqualToTarget() {
        // All observations == target => both sPlus and sMinus are all zeros
        val obs = DoubleArray(10) { 5.0 }
        val result = cusum(obs, target = 5.0, k = 0.5, h = 3.0)

        for (i in obs.indices) {
            assertEquals(0.0, result.sPlus[i], tol, "sPlus[$i] should be 0")
            assertEquals(0.0, result.sMinus[i], tol, "sMinus[$i] should be 0")
        }
        assertEquals(-1, result.alarmIndex, "no alarm when all equal to target")
    }

    @Test
    fun testCusumKEqualsZero() {
        // k=0 (zero allowance): pure cumulative sum of deviations from target
        // With target=0, sPlus is cumulative positive excursions, sMinus of negative
        // numpy: cusum([1, -0.5, 2, -1.5, 3], 0, 0, 100)
        val obs = doubleArrayOf(1.0, -0.5, 2.0, -1.5, 3.0)
        val result = cusum(obs, target = 0.0, k = 0.0, h = 100.0)

        // numpy: sPlus=[1, 0.5, 2.5, 1, 4]; sMinus=[0, 0.5, 0, 1.5, 0]
        val expectedPlus = doubleArrayOf(1.0, 0.5, 2.5, 1.0, 4.0)
        val expectedMinus = doubleArrayOf(0.0, 0.5, 0.0, 1.5, 0.0)
        for (i in obs.indices) {
            assertEquals(expectedPlus[i], result.sPlus[i], tol, "sPlus[$i]")
            assertEquals(expectedMinus[i], result.sMinus[i], tol, "sMinus[$i]")
        }
        assertEquals(-1, result.alarmIndex, "no alarm with h=100")
    }

    @Test
    fun testCusumVeryLargeH() {
        // Very large h: never triggers alarm regardless of data
        val obs = doubleArrayOf(100.0, 200.0, 300.0, 400.0, 500.0)
        val result = cusum(obs, target = 0.0, k = 0.5, h = 1e100)

        assertEquals(-1, result.alarmIndex, "very large h => no alarm")
        // sPlus should be cumulative (no capping)
        assertTrue(result.sPlus[4] > 1000.0, "sPlus should accumulate")
    }

    @Test
    fun testCusumVerySmallH() {
        // Very small h: alarm triggers at first deviation beyond k
        // numpy: cusum([10.5, 10.0], 10.0, 0.25, 0.1) => sPlus[0]=0.25, alarm at 0
        val obs = doubleArrayOf(10.5, 10.0)
        val result = cusum(obs, target = 10.0, k = 0.25, h = 0.1)

        assertEquals(0.25, result.sPlus[0], tol, "sPlus[0]")
        assertEquals(0, result.alarmIndex, "alarm at 0 with tiny h")
    }

    @Test
    fun testCusumIterableOverload() {
        // Iterable overload should give identical result to DoubleArray overload
        val array = doubleArrayOf(10.1, 10.3, 10.5, 10.8, 11.0, 11.3)
        val list: Iterable<Double> = array.toList()
        val expected = cusum(array, target = 10.0, k = 0.25, h = 4.0)
        val actual = cusum(list, target = 10.0, k = 0.25, h = 4.0)

        assertEquals(expected, actual, "Iterable overload should match DoubleArray overload")
    }

    @Test
    fun testCusumSequenceOverload() {
        // Sequence overload should give identical result to DoubleArray overload
        val array = doubleArrayOf(10.1, 10.3, 10.5, 10.8, 11.0, 11.3)
        val seq: Sequence<Double> = array.asSequence()
        val expected = cusum(array, target = 10.0, k = 0.25, h = 4.0)
        val actual = cusum(seq, target = 10.0, k = 0.25, h = 4.0)

        assertEquals(expected, actual, "Sequence overload should match DoubleArray overload")
    }

    @Test
    fun testCusumResultArrayLengths() {
        // sPlus and sMinus arrays have same length as input
        val obs = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        val result = cusum(obs, target = 3.0, k = 0.5, h = 10.0)
        assertEquals(obs.size, result.sPlus.size, "sPlus.size")
        assertEquals(obs.size, result.sMinus.size, "sMinus.size")
    }

    // ===== cusum: Degenerate input =====

    @Test
    fun testCusumEmptyArray() {
        assertFailsWith<InsufficientDataException> {
            cusum(doubleArrayOf(), target = 0.0, k = 0.5, h = 5.0)
        }
    }

    @Test
    fun testCusumEmptyIterable() {
        assertFailsWith<InsufficientDataException> {
            cusum(emptyList<Double>(), target = 0.0, k = 0.5, h = 5.0)
        }
    }

    @Test
    fun testCusumEmptySequence() {
        assertFailsWith<InsufficientDataException> {
            cusum(emptySequence<Double>(), target = 0.0, k = 0.5, h = 5.0)
        }
    }

    @Test
    fun testCusumNegativeK() {
        assertFailsWith<InvalidParameterException> {
            cusum(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, k = -0.1, h = 5.0)
        }
    }

    @Test
    fun testCusumNegativeKIterable() {
        assertFailsWith<InvalidParameterException> {
            cusum(listOf(1.0, 2.0, 3.0), target = 0.0, k = -0.5, h = 5.0)
        }
    }

    @Test
    fun testCusumNegativeKSequence() {
        assertFailsWith<InvalidParameterException> {
            cusum(sequenceOf(1.0, 2.0, 3.0), target = 0.0, k = -1.0, h = 5.0)
        }
    }

    @Test
    fun testCusumZeroH() {
        assertFailsWith<InvalidParameterException> {
            cusum(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, k = 0.5, h = 0.0)
        }
    }

    @Test
    fun testCusumNegativeH() {
        assertFailsWith<InvalidParameterException> {
            cusum(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, k = 0.5, h = -1.0)
        }
    }

    @Test
    fun testCusumNegativeHIterable() {
        assertFailsWith<InvalidParameterException> {
            cusum(listOf(1.0, 2.0, 3.0), target = 0.0, k = 0.5, h = -0.1)
        }
    }

    @Test
    fun testCusumNegativeHSequence() {
        assertFailsWith<InvalidParameterException> {
            cusum(sequenceOf(1.0, 2.0, 3.0), target = 0.0, k = 0.5, h = -5.0)
        }
    }

    // ===== cusum: Non-finite input =====

    @Test
    fun testCusumNaNInObservations() {
        // NaN propagates from its index onward (IEEE 754)
        val obs = doubleArrayOf(10.0, 10.1, Double.NaN, 10.3, 10.4)
        val result = cusum(obs, target = 10.0, k = 0.5, h = 5.0)

        // Before NaN: normal computation
        assertEquals(0.0, result.sPlus[0], tol, "sPlus[0] before NaN")
        assertEquals(0.0, result.sPlus[1], tol, "sPlus[1] before NaN")
        // NaN index and beyond propagate NaN
        assertTrue(result.sPlus[2].isNaN(), "sPlus[2] should be NaN")
        assertTrue(result.sMinus[2].isNaN(), "sMinus[2] should be NaN")
        // Subsequent indices remain NaN due to running accumulator
        assertTrue(result.sPlus[3].isNaN(), "sPlus[3] propagates NaN")
        assertTrue(result.sPlus[4].isNaN(), "sPlus[4] propagates NaN")
    }

    @Test
    fun testCusumPositiveInfinityInObservations() {
        // +Infinity in observations: result is non-finite from that index
        val obs = doubleArrayOf(10.0, Double.POSITIVE_INFINITY, 10.0)
        val result = cusum(obs, target = 10.0, k = 0.5, h = 5.0)

        assertTrue(!result.sPlus[1].isFinite(), "sPlus[1] with +Infinity should be non-finite")
    }

    @Test
    fun testCusumNegativeInfinityInObservations() {
        // -Infinity in observations: result is non-finite from that index
        val obs = doubleArrayOf(10.0, Double.NEGATIVE_INFINITY, 10.0)
        val result = cusum(obs, target = 10.0, k = 0.5, h = 5.0)

        assertTrue(!result.sMinus[1].isFinite(), "sMinus[1] with -Infinity should be non-finite")
    }

    @Test
    fun testCusumNaNTargetDoesNotThrow() {
        // NaN passes validation (per project convention); results become NaN
        val obs = doubleArrayOf(1.0, 2.0, 3.0)
        val result = cusum(obs, target = Double.NaN, k = 0.5, h = 5.0)

        for (i in obs.indices) {
            assertTrue(result.sPlus[i].isNaN(), "sPlus[$i] should be NaN with NaN target")
            assertTrue(result.sMinus[i].isNaN(), "sMinus[$i] should be NaN with NaN target")
        }
    }

    @Test
    fun testCusumNaNKDoesNotThrow() {
        // NaN k passes validation (NaN < 0.0 is false per IEEE 754); results become NaN
        val obs = doubleArrayOf(1.0, 2.0, 3.0)
        val result = cusum(obs, target = 1.0, k = Double.NaN, h = 5.0)

        for (i in obs.indices) {
            assertTrue(result.sPlus[i].isNaN(), "sPlus[$i] should be NaN with NaN k")
            assertTrue(result.sMinus[i].isNaN(), "sMinus[$i] should be NaN with NaN k")
        }
    }

    @Test
    fun testCusumNaNHDoesNotThrow() {
        // NaN h passes validation (NaN <= 0.0 is false per IEEE 754)
        // The alarm comparison `cPlus > h` with NaN h is always false, so no alarm
        val obs = doubleArrayOf(1.0, 2.0, 3.0)
        val result = cusum(obs, target = 1.0, k = 0.5, h = Double.NaN)

        // Computation proceeds normally; no alarm triggers because x > NaN is always false
        assertEquals(-1, result.alarmIndex, "NaN h produces no alarm")
    }

    // ===== cusum: Property-based =====

    @Test
    fun testCusumNonNegativeForFiniteInput() {
        // sPlus and sMinus are always >= 0 for finite inputs (since max(0, ·))
        val obs = doubleArrayOf(9.0, 11.0, 8.5, 12.0, 7.5, 13.0, 10.0, 9.0, 11.5, 8.0)
        val result = cusum(obs, target = 10.0, k = 0.3, h = 4.0)

        for (i in obs.indices) {
            assertTrue(result.sPlus[i] >= 0.0, "sPlus[$i]=${result.sPlus[i]} should be >= 0")
            assertTrue(result.sMinus[i] >= 0.0, "sMinus[$i]=${result.sMinus[i]} should be >= 0")
        }
    }

    @Test
    fun testCusumTranslationInvariance() {
        // Shifting all observations AND target by the same constant gives identical sPlus/sMinus
        val obs = doubleArrayOf(10.1, 10.3, 10.5, 10.8, 11.0, 11.3)
        val shift = 1000.0
        val shiftedObs = DoubleArray(obs.size) { obs[it] + shift }

        val original = cusum(obs, target = 10.0, k = 0.25, h = 4.0)
        val shifted = cusum(shiftedObs, target = 10.0 + shift, k = 0.25, h = 4.0)

        for (i in obs.indices) {
            assertEquals(original.sPlus[i], shifted.sPlus[i], 1e-7, "sPlus[$i] translation invariance")
            assertEquals(original.sMinus[i], shifted.sMinus[i], 1e-7, "sMinus[$i] translation invariance")
        }
        assertEquals(original.alarmIndex, shifted.alarmIndex, "alarmIndex unchanged by translation")
    }

    @Test
    fun testCusumSymmetryByNegation() {
        // Negating observations AND target swaps sPlus and sMinus
        val obs = doubleArrayOf(10.2, 10.4, 10.6, 10.9, 11.2, 11.5, 11.8, 12.0)
        val negObs = DoubleArray(obs.size) { -obs[it] }

        val original = cusum(obs, target = 10.0, k = 0.5, h = 3.0)
        val negated = cusum(negObs, target = -10.0, k = 0.5, h = 3.0)

        for (i in obs.indices) {
            assertEquals(original.sPlus[i], negated.sMinus[i], tol, "sPlus[$i] <-> sMinus[$i] under negation")
            assertEquals(original.sMinus[i], negated.sPlus[i], tol, "sMinus[$i] <-> sPlus[$i] under negation")
        }
        assertEquals(original.alarmIndex, negated.alarmIndex, "alarmIndex unchanged by negation")
    }

    @Test
    fun testCusumSingleObservationFormula() {
        // For n=1: sPlus[0] = max(0, x-target-k), sMinus[0] = max(0, target-k-x)
        val target = 10.0
        val k = 0.5
        val h = 100.0 // large h to avoid alarm

        for (x in listOf(5.0, 9.0, 10.0, 10.25, 11.0, 15.0)) {
            val result = cusum(doubleArrayOf(x), target = target, k = k, h = h)
            val expectedPlus = maxOf(0.0, x - target - k)
            val expectedMinus = maxOf(0.0, target - k - x)
            assertEquals(expectedPlus, result.sPlus[0], tol, "sPlus[0] for x=$x")
            assertEquals(expectedMinus, result.sMinus[0], tol, "sMinus[0] for x=$x")
        }
    }

    @Test
    fun testCusumArrayLengthProperty() {
        // For various input sizes, output arrays match input length
        for (n in listOf(1, 2, 5, 10, 50, 100)) {
            val obs = DoubleArray(n) { (it + 1).toDouble() }
            val result = cusum(obs, target = 5.0, k = 0.5, h = 100.0)
            assertEquals(n, result.sPlus.size, "sPlus.size for n=$n")
            assertEquals(n, result.sMinus.size, "sMinus.size for n=$n")
        }
    }

    @Test
    fun testCusumLargeHNoAlarm() {
        // If h is very large, alarmIndex = -1 regardless of data
        val obs = doubleArrayOf(100.0, -200.0, 300.0, -400.0)
        val result = cusum(obs, target = 0.0, k = 0.5, h = 1e308)

        assertEquals(-1, result.alarmIndex, "very large h => no alarm")
    }

    @Test
    fun testCusumAlarmIndexIsFirst() {
        // alarmIndex should be the earliest index where sPlus>h or sMinus>h
        val obs = doubleArrayOf(10.2, 10.4, 10.6, 10.9, 11.2, 11.5, 11.8, 12.0)
        val result = cusum(obs, target = 10.0, k = 0.5, h = 3.0)

        val alarm = result.alarmIndex
        assertTrue(alarm >= 0, "alarm should fire")
        // No earlier index should have fired
        for (i in 0 until alarm) {
            assertTrue(
                result.sPlus[i] <= 3.0 && result.sMinus[i] <= 3.0,
                "Index $i before alarm: sPlus=${result.sPlus[i]}, sMinus=${result.sMinus[i]}"
            )
        }
        // At the alarm index, at least one exceeds h
        assertTrue(
            result.sPlus[alarm] > 3.0 || result.sMinus[alarm] > 3.0,
            "At alarm: sPlus=${result.sPlus[alarm]}, sMinus=${result.sMinus[alarm]}"
        )
    }

    @Test
    fun testCusumRecursiveDefinition() {
        // Property: for all i >= 1,
        //   sPlus[i] == max(0, sPlus[i-1] + (obs[i] - target - k))
        //   sMinus[i] == max(0, sMinus[i-1] + (target - k - obs[i]))
        val obs = doubleArrayOf(10.1, 10.3, 10.5, 10.8, 11.0, 11.3, 9.5, 9.2, 10.7, 11.5)
        val target = 10.0
        val k = 0.25
        val result = cusum(obs, target = target, k = k, h = 10.0)

        for (i in 1 until obs.size) {
            val expectedPlus = maxOf(0.0, result.sPlus[i - 1] + (obs[i] - target - k))
            val expectedMinus = maxOf(0.0, result.sMinus[i - 1] + (target - k - obs[i]))
            assertEquals(expectedPlus, result.sPlus[i], tol, "sPlus recursion at i=$i")
            assertEquals(expectedMinus, result.sMinus[i], tol, "sMinus recursion at i=$i")
        }
    }

    // ===== CusumResult: data class =====

    @Test
    fun testCusumResultEquality() {
        // Two identical calls produce equal results (via equals/hashCode)
        val obs = doubleArrayOf(10.1, 10.3, 10.5, 10.8, 11.0, 11.3)
        val r1 = cusum(obs, target = 10.0, k = 0.25, h = 4.0)
        val r2 = cusum(obs, target = 10.0, k = 0.25, h = 4.0)

        assertEquals(r1, r2, "Same input should produce equal CusumResult")
        assertEquals(r1.hashCode(), r2.hashCode(), "Equal results should have equal hashCode")
    }

    @Test
    fun testCusumResultEqualityDifferentInstances() {
        // equals uses contentEquals, so different array instances with same values are equal
        val r1 = CusumResult(
            sPlus = doubleArrayOf(0.0, 0.1, 0.3),
            sMinus = doubleArrayOf(0.0, 0.0, 0.0),
            alarmIndex = -1,
        )
        val r2 = CusumResult(
            sPlus = doubleArrayOf(0.0, 0.1, 0.3),
            sMinus = doubleArrayOf(0.0, 0.0, 0.0),
            alarmIndex = -1,
        )
        assertTrue(r1 !== r2, "Different instances")
        assertEquals(r1, r2, "Different array instances with same content should be equal")
        assertEquals(r1.hashCode(), r2.hashCode(), "hashCode consistent with equals")
    }

    @Test
    fun testCusumResultInequality() {
        // Different alarmIndex => not equal
        val r1 = CusumResult(
            sPlus = doubleArrayOf(0.0, 0.1),
            sMinus = doubleArrayOf(0.0, 0.0),
            alarmIndex = -1,
        )
        val r2 = CusumResult(
            sPlus = doubleArrayOf(0.0, 0.1),
            sMinus = doubleArrayOf(0.0, 0.0),
            alarmIndex = 1,
        )
        assertTrue(r1 != r2, "Different alarmIndex => not equal")

        // Different sPlus contents => not equal
        val r3 = CusumResult(
            sPlus = doubleArrayOf(0.0, 0.2),
            sMinus = doubleArrayOf(0.0, 0.0),
            alarmIndex = -1,
        )
        assertTrue(r1 != r3, "Different sPlus => not equal")

        // Different sMinus contents => not equal
        val r4 = CusumResult(
            sPlus = doubleArrayOf(0.0, 0.1),
            sMinus = doubleArrayOf(0.1, 0.0),
            alarmIndex = -1,
        )
        assertTrue(r1 != r4, "Different sMinus => not equal")
    }

    @Test
    fun testCusumResultEqualsSelf() {
        val r = CusumResult(
            sPlus = doubleArrayOf(0.0, 0.1, 0.3),
            sMinus = doubleArrayOf(0.0, 0.0, 0.0),
            alarmIndex = -1,
        )
        assertEquals(r, r, "equals with self")
    }

    @Test
    fun testCusumResultEqualsNonCusumResult() {
        val r = CusumResult(
            sPlus = doubleArrayOf(0.0),
            sMinus = doubleArrayOf(0.0),
            alarmIndex = -1,
        )
        assertTrue(!r.equals("not a CusumResult"), "equals returns false for non-CusumResult")
        assertTrue(!r.equals(null), "equals returns false for null")
    }

    @Test
    fun testCusumResultDestructuring() {
        // componentN (destructuring) works on the data class
        val obs = doubleArrayOf(10.1, 10.3, 10.5)
        val result = cusum(obs, target = 10.0, k = 0.25, h = 4.0)
        val (sPlus, sMinus, alarmIndex) = result

        assertEquals(result.sPlus.size, sPlus.size, "sPlus via destructuring")
        assertEquals(result.sMinus.size, sMinus.size, "sMinus via destructuring")
        assertEquals(result.alarmIndex, alarmIndex, "alarmIndex via destructuring")
    }

    @Test
    fun testCusumResultToStringRendersArrayContents() {
        // toString must use contentToString() for DoubleArray fields — the default
        // data-class toString would print `[D@<hash>` which is useless for diagnostics.
        val result = CusumResult(
            sPlus = doubleArrayOf(0.0, 0.1, 0.3),
            sMinus = doubleArrayOf(0.0, 0.0, 0.05),
            alarmIndex = -1,
        )
        val s = result.toString()
        assertTrue(s.contains("sPlus=[0.0, 0.1, 0.3]"), "toString should render sPlus elements, got: $s")
        assertTrue(s.contains("sMinus=[0.0, 0.0, 0.05]"), "toString should render sMinus elements, got: $s")
        assertTrue(s.contains("alarmIndex=-1"), "toString should render alarmIndex, got: $s")
        assertTrue(!s.contains("[D@"), "toString must not leak default array identity, got: $s")
    }

    // ===== ewma: Basic correctness =====

    @Test
    fun testEwmaSimpleHandComputable() {
        // Hand-computable example: target=10, sigma=1, lambda=0.5, L=3
        // Z_0 = 0.5*10.5 + 0.5*10 = 10.25
        // Z_1 = 0.5*11.0 + 0.5*10.25 = 10.625
        // Z_2 = 0.5*11.5 + 0.5*10.625 = 11.0625
        // Z_3 = 0.5*12.0 + 0.5*11.0625 = 11.53125
        // Z_4 = 0.5*12.5 + 0.5*11.53125 = 12.015625
        // Z_5 = 0.5*13.0 + 0.5*12.015625 = 12.5078125
        // sigma_Z_1 = 1 * sqrt(0.5/1.5 * (1 - 0.25)) = sqrt(1/3 * 0.75) = sqrt(0.25) = 0.5
        // UCL_1 = 10 + 3*0.5 = 11.5
        val obs = doubleArrayOf(10.5, 11.0, 11.5, 12.0, 12.5, 13.0)
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.5, controlLimitWidth = 3.0)

        // numpy: Z values
        assertEquals(10.25, result.smoothedValues[0], tol, "Z_0")
        assertEquals(10.625, result.smoothedValues[1], tol, "Z_1")
        assertEquals(11.0625, result.smoothedValues[2], tol, "Z_2")
        assertEquals(11.53125, result.smoothedValues[3], tol, "Z_3")
        assertEquals(12.015625, result.smoothedValues[4], tol, "Z_4")
        assertEquals(12.5078125, result.smoothedValues[5], tol, "Z_5")

        // numpy: UCL/LCL values
        assertEquals(11.5, result.ucl[0], tol, "UCL_0")
        assertEquals(8.5, result.lcl[0], tol, "LCL_0")
        assertEquals(11.6770509831248, result.ucl[1], tol, "UCL_1")
        assertEquals(8.32294901687516, result.lcl[1], tol, "LCL_1")
        assertEquals(11.7184658856084, result.ucl[2], tol, "UCL_2")
        assertEquals(11.7286645857424, result.ucl[3], tol, "UCL_3")
        assertEquals(11.7312048730581, result.ucl[4], tol, "UCL_4")
        assertEquals(11.7318393626792, result.ucl[5], tol, "UCL_5")

        // numpy: Z_4=12.015625 > UCL_4=11.7312, Z_5=12.5078 > UCL_5=11.7318 => OOC at 4, 5
        assertTrue(result.outOfControl.contentEquals(intArrayOf(4, 5)), "OOC indices")
    }

    @Test
    fun testEwmaDocstringExample() {
        // Example from the ewma() docstring
        // numpy: target=25, sigma=1, lambda=0.2, L=3
        val obs = doubleArrayOf(25.0, 24.5, 25.2, 26.1, 25.8, 27.0, 26.5, 28.0)
        val result = ewma(obs, target = 25.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)

        // numpy: Z_0 = 0.2*25 + 0.8*25 = 25; Z_1 = 0.2*24.5 + 0.8*25 = 24.9
        assertEquals(25.0, result.smoothedValues[0], tol, "Z_0")
        assertEquals(24.9, result.smoothedValues[1], tol, "Z_1")
        assertEquals(24.96, result.smoothedValues[2], tol, "Z_2")
        assertEquals(25.188, result.smoothedValues[3], tol, "Z_3")
        assertEquals(25.3104, result.smoothedValues[4], tol, "Z_4")
        assertEquals(25.64832, result.smoothedValues[5], tol, "Z_5")
        assertEquals(25.818656, result.smoothedValues[6], tol, "Z_6")
        assertEquals(26.2549248, result.smoothedValues[7], tol, "Z_7")

        // numpy: UCL at steady state with lam=0.2 approaches 25 + 3*sqrt(0.2/1.8) ≈ 25.9999
        assertEquals(25.6, result.ucl[0], tol, "UCL_0")
        assertEquals(25.9858257971513, result.ucl[7], tol, "UCL_7")

        // numpy: Z_7=26.255 > UCL_7=25.986 => OOC at 7
        assertTrue(result.outOfControl.contentEquals(intArrayOf(7)), "OOC indices")
    }

    @Test
    fun testEwmaMontgomeryExample() {
        // Montgomery "Introduction to Statistical Quality Control" (7th ed.), §9.2
        // Same 30 observations as CUSUM example, target=10, sigma=1, lambda=0.1, L=2.7
        val obs = doubleArrayOf(
            9.45, 7.99, 9.29, 11.66, 12.16, 10.18, 8.04, 11.46, 9.20, 10.34,
            9.03, 11.47, 10.51, 9.40, 10.08, 9.37, 10.62, 10.31, 8.52, 10.84,
            10.90, 9.33, 12.29, 11.50, 10.60, 11.08, 10.38, 11.62, 11.31, 10.52
        )
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.1, controlLimitWidth = 2.7)

        // numpy: Z_0 = 0.1*9.45 + 0.9*10 = 9.945
        assertEquals(9.945, result.smoothedValues[0], tol, "Z_0")
        // numpy: Z_1 = 0.1*7.99 + 0.9*9.945 = 0.799 + 8.9505 = 9.7495
        assertEquals(9.7495, result.smoothedValues[1], tol, "Z_1")
        // numpy: Z_4 = 10.1252755
        assertEquals(10.1252755, result.smoothedValues[4], tol, "Z_4")
        // numpy: Z_9 = 10.023159729995
        assertEquals(10.023159729995, result.smoothedValues[9], tol, "Z_9")
        // numpy: Z_28 = 10.6468234545764
        assertEquals(10.6468234545764, result.smoothedValues[28], tol, "Z_28")
        // numpy: Z_29 = 10.6341411091187
        assertEquals(10.6341411091187, result.smoothedValues[29], tol, "Z_29")

        // numpy: UCL_0 = 10 + 2.7 * sqrt(0.1/1.9 * (1-0.81)) = 10 + 2.7 * 0.1 = 10.27
        assertEquals(10.27, result.ucl[0], tol, "UCL_0")
        // numpy: UCL_29 (approaching steady state) = 10.6188656769026
        assertEquals(10.6188656769026, result.ucl[29], tol, "UCL_29")
        assertEquals(9.38113432309742, result.lcl[29], tol, "LCL_29")

        // numpy: Z_28=10.6468 > UCL_28=10.6187, Z_29=10.6341 > UCL_29=10.6189 => OOC at 28, 29
        assertTrue(
            result.outOfControl.contentEquals(intArrayOf(28, 29)),
            "OOC indices should be [28, 29], got ${result.outOfControl.toList()}"
        )
    }

    @Test
    fun testEwmaLambdaOneIsShewhart() {
        // Special case: lambda=1 reduces to Shewhart individuals chart
        //   Z_t = x_t, sigma_Zt = sigma constant for all t (since (1-lam)^(2t) = 0)
        //   UCL = target + L*sigma, LCL = target - L*sigma (both constant)
        val obs = doubleArrayOf(0.0, 1.0, -1.0, 5.0, -5.0)
        val result = ewma(obs, target = 0.0, sigma = 1.0, lambda = 1.0, controlLimitWidth = 3.0)

        // numpy: Z = obs, UCL = 3, LCL = -3 for all t
        for (i in obs.indices) {
            assertEquals(obs[i], result.smoothedValues[i], tol, "Z[$i] = obs[$i] when lambda=1")
            assertEquals(3.0, result.ucl[i], tol, "UCL[$i] = 3 when lambda=1")
            assertEquals(-3.0, result.lcl[i], tol, "LCL[$i] = -3 when lambda=1")
        }
        // numpy: obs[3]=5 > UCL=3, obs[4]=-5 < LCL=-3 => OOC at 3, 4
        assertTrue(result.outOfControl.contentEquals(intArrayOf(3, 4)), "OOC indices")
    }

    @Test
    fun testEwmaSingleObservation() {
        // Single observation with lambda=0.2
        // Z_0 = 0.2*12 + 0.8*10 = 10.4
        // sigma_Z_1 = 1 * sqrt(0.2/1.8 * (1 - 0.64)) = sqrt(0.04) = 0.2
        // UCL = 10 + 3*0.2 = 10.6, LCL = 10 - 0.6 = 9.4
        val result = ewma(doubleArrayOf(12.0), target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)

        assertEquals(1, result.smoothedValues.size, "smoothedValues.size")
        assertEquals(1, result.ucl.size, "ucl.size")
        assertEquals(1, result.lcl.size, "lcl.size")
        assertEquals(10.4, result.smoothedValues[0], tol, "Z_0")
        assertEquals(10.6, result.ucl[0], tol, "UCL_0")
        assertEquals(9.4, result.lcl[0], tol, "LCL_0")
        // 10.4 is between 9.4 and 10.6 => no OOC
        assertEquals(0, result.outOfControl.size, "no OOC for single in-control observation")
    }

    @Test
    fun testEwmaSingleObservationOutOfControl() {
        // With lambda=1, UCL=target+L*sigma immediately fires
        // Z_0 = 20 > UCL = 10 + 3 = 13
        val result = ewma(doubleArrayOf(20.0), target = 10.0, sigma = 1.0, lambda = 1.0, controlLimitWidth = 3.0)
        assertEquals(20.0, result.smoothedValues[0], tol, "Z_0")
        assertEquals(13.0, result.ucl[0], tol, "UCL_0")
        assertEquals(7.0, result.lcl[0], tol, "LCL_0")
        assertTrue(result.outOfControl.contentEquals(intArrayOf(0)), "alarm at i=0")
    }

    // ===== ewma: Edge cases =====

    @Test
    fun testEwmaAllObservationsAtTarget() {
        // All observations equal target => Z_t = target for all t, no OOC
        val obs = DoubleArray(10) { 10.0 }
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)

        for (i in obs.indices) {
            assertEquals(10.0, result.smoothedValues[i], tol, "Z[$i] = target")
            assertTrue(result.ucl[i] > 10.0, "UCL[$i] > target")
            assertTrue(result.lcl[i] < 10.0, "LCL[$i] < target")
        }
        assertEquals(0, result.outOfControl.size, "no OOC when all obs at target")

        // numpy: steady-state UCL approaches target + L*sigma*sqrt(lambda/(2-lambda))
        //        = 10 + 3*sqrt(0.2/1.8) ≈ 10.99999 at large t
        assertEquals(10.6, result.ucl[0], tol, "UCL_0")
        assertEquals(10.9942186806503, result.ucl[9], tol, "UCL_9 approaches steady state")
    }

    @Test
    fun testEwmaLambdaMinuscule() {
        // Very small lambda (near zero is allowed): strong memory of past
        // Z values barely move from target
        // numpy: lambda=0.001, target=100, sigma=1, L=3
        val obs = doubleArrayOf(100.0, 100.5, 101.0)
        val result = ewma(obs, target = 100.0, sigma = 1.0, lambda = 0.001, controlLimitWidth = 3.0)

        // numpy: Z_0 = 0.001*100 + 0.999*100 = 100
        assertEquals(100.0, result.smoothedValues[0], tol, "Z_0 with tiny lambda")
        // numpy: Z_1 = 0.001*100.5 + 0.999*100 = 100.0005
        assertEquals(100.0005, result.smoothedValues[1], tol, "Z_1 with tiny lambda")
        // numpy: Z_2 = 0.001*101 + 0.999*100.0005 = 100.0014995
        assertEquals(100.0014995, result.smoothedValues[2], tol, "Z_2 with tiny lambda")

        // numpy: sigma_Z_1^2 = 0.001/1.999 * (1 - 0.999^2) = 0.001/1.999 * 0.001999 = 1e-6
        // so sigma_Z_1 = 0.001, UCL_0 = 100 + 3*0.001 = 100.003
        assertEquals(100.003, result.ucl[0], tol, "UCL_0 tiny lambda")
    }

    @Test
    fun testEwmaLambdaNearOne() {
        // lambda very close to 1 but not equal: behavior approaches Shewhart
        // sigma_Z_1^2 = sigma^2 * (1/(2-lam)) * (2lam-lam^2) = lam*sigma^2 (approx)
        val obs = doubleArrayOf(0.0, 0.0, 0.0, 0.0)
        val result = ewma(obs, target = 0.0, sigma = 1.0, lambda = 0.99, controlLimitWidth = 3.0)

        // All obs = target => all Z = target
        for (i in obs.indices) {
            assertEquals(0.0, result.smoothedValues[i], tol, "Z[$i] = 0")
        }
        // UCL_0 = 0 + 3*sqrt(0.99/1.01 * (1 - 0.01^2)) ≈ 2.97
        assertTrue(result.ucl[0] > 2.95, "UCL_0 close to 3 for lam near 1")
        assertTrue(result.ucl[0] < 3.0, "UCL_0 < 3 for lam < 1")
    }

    @Test
    fun testEwmaControlLimitsIncreaseThenPlateau() {
        // Control limits |UCL - target| and |target - LCL| widen monotonically toward steady state
        val obs = DoubleArray(50) { 10.0 } // stays at target to isolate limit progression
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.1, controlLimitWidth = 3.0)

        for (i in 1 until obs.size) {
            assertTrue(
                result.ucl[i] >= result.ucl[i - 1],
                "UCL[$i]=${result.ucl[i]} >= UCL[${i - 1}]=${result.ucl[i - 1]}"
            )
            assertTrue(
                result.lcl[i] <= result.lcl[i - 1],
                "LCL[$i]=${result.lcl[i]} <= LCL[${i - 1}]=${result.lcl[i - 1]}"
            )
        }
        // numpy: Steady state UCL = 10 + 3 * sqrt(0.1/1.9) = 10.6887...
        val steadyUcl = 10.0 + 3.0 * sqrt(0.1 / 1.9)
        assertEquals(steadyUcl, result.ucl[49], 1e-4, "UCL approaches steady state")
    }

    @Test
    fun testEwmaTwoSidedAlarm() {
        // Observations cross both UCL (upward) and LCL (downward) in the same series
        // numpy: lambda=0.3, target=10, sigma=1, L=3
        val obs = doubleArrayOf(15.0, 15.0, 15.0, 15.0, 5.0, 5.0, 5.0, 5.0)
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.3, controlLimitWidth = 3.0)

        // numpy: Z_0=11.5 > 10.9 => OOC, Z_6=8.018 < 8.744 => OOC
        assertEquals(11.5, result.smoothedValues[0], tol, "Z_0")
        assertEquals(12.55, result.smoothedValues[1], tol, "Z_1")
        assertEquals(13.285, result.smoothedValues[2], tol, "Z_2")
        assertEquals(13.7995, result.smoothedValues[3], tol, "Z_3")
        assertEquals(11.15965, result.smoothedValues[4], tol, "Z_4")
        assertEquals(9.311755, result.smoothedValues[5], tol, "Z_5")
        assertEquals(8.0182285, result.smoothedValues[6], tol, "Z_6")
        assertEquals(7.11275995, result.smoothedValues[7], tol, "Z_7")

        // numpy: UCL and LCL at index 7
        assertEquals(11.2581562394202, result.ucl[7], tol, "UCL_7")
        assertEquals(8.74184376057984, result.lcl[7], tol, "LCL_7")

        // numpy: Indices where Z > UCL or Z < LCL => [0, 1, 2, 3, 6, 7]
        assertTrue(
            result.outOfControl.contentEquals(intArrayOf(0, 1, 2, 3, 6, 7)),
            "two-sided OOC: ${result.outOfControl.toList()}"
        )
    }

    @Test
    fun testEwmaReturnsArraysOfCorrectLength() {
        for (n in listOf(1, 2, 5, 10, 50, 100)) {
            val obs = DoubleArray(n) { (it + 1).toDouble() }
            val result = ewma(obs, target = 5.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
            assertEquals(n, result.smoothedValues.size, "smoothedValues.size for n=$n")
            assertEquals(n, result.ucl.size, "ucl.size for n=$n")
            assertEquals(n, result.lcl.size, "lcl.size for n=$n")
        }
    }

    // ===== ewma: Degenerate input =====

    @Test
    fun testEwmaEmptyArray() {
        assertFailsWith<InsufficientDataException> {
            ewma(doubleArrayOf(), target = 0.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
        }
    }

    @Test
    fun testEwmaEmptyIterable() {
        assertFailsWith<InsufficientDataException> {
            ewma(emptyList<Double>(), target = 0.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
        }
    }

    @Test
    fun testEwmaEmptySequence() {
        assertFailsWith<InsufficientDataException> {
            ewma(emptySequence<Double>(), target = 0.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
        }
    }

    @Test
    fun testEwmaSigmaZero() {
        assertFailsWith<InvalidParameterException> {
            ewma(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, sigma = 0.0, lambda = 0.2, controlLimitWidth = 3.0)
        }
    }

    @Test
    fun testEwmaSigmaNegative() {
        assertFailsWith<InvalidParameterException> {
            ewma(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, sigma = -0.5, lambda = 0.2, controlLimitWidth = 3.0)
        }
    }

    @Test
    fun testEwmaLambdaZero() {
        assertFailsWith<InvalidParameterException> {
            ewma(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, sigma = 1.0, lambda = 0.0, controlLimitWidth = 3.0)
        }
    }

    @Test
    fun testEwmaLambdaNegative() {
        assertFailsWith<InvalidParameterException> {
            ewma(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, sigma = 1.0, lambda = -0.1, controlLimitWidth = 3.0)
        }
    }

    @Test
    fun testEwmaLambdaGreaterThanOne() {
        assertFailsWith<InvalidParameterException> {
            ewma(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, sigma = 1.0, lambda = 1.0001, controlLimitWidth = 3.0)
        }
    }

    @Test
    fun testEwmaLambdaExactlyOne() {
        // lambda=1 is the allowed boundary, should NOT throw
        val result = ewma(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, sigma = 1.0, lambda = 1.0, controlLimitWidth = 3.0)
        assertEquals(3, result.smoothedValues.size, "lambda=1 is allowed")
    }

    @Test
    fun testEwmaControlLimitWidthZero() {
        assertFailsWith<InvalidParameterException> {
            ewma(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 0.0)
        }
    }

    @Test
    fun testEwmaControlLimitWidthNegative() {
        assertFailsWith<InvalidParameterException> {
            ewma(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = -1.0)
        }
    }

    @Test
    fun testEwmaSigmaNegativeIterable() {
        assertFailsWith<InvalidParameterException> {
            ewma(listOf(1.0, 2.0, 3.0), target = 0.0, sigma = -1.0, lambda = 0.2, controlLimitWidth = 3.0)
        }
    }

    @Test
    fun testEwmaLambdaOutOfRangeSequence() {
        assertFailsWith<InvalidParameterException> {
            ewma(sequenceOf(1.0, 2.0, 3.0), target = 0.0, sigma = 1.0, lambda = 1.5, controlLimitWidth = 3.0)
        }
    }

    // ===== ewma: Extreme parameters =====

    @Test
    fun testEwmaLargeOffsetData() {
        // Large offset tests numerical stability
        val offset = 1e12
        val obs = doubleArrayOf(offset + 0.1, offset + 0.2, offset - 0.1)
        val result = ewma(obs, target = offset, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)

        // numpy: Z_0 = 0.2*(1e12+0.1) + 0.8*1e12 = 1e12 + 0.02
        // Finite precision for 1e12 means ~1 ULP is a few microseconds
        assertTrue(result.smoothedValues[0].isFinite(), "Z_0 finite with large offset")
        assertEquals(offset + 0.02, result.smoothedValues[0], 1e-3, "Z_0")
        // numpy: UCL_0 = 1e12 + 0.6
        assertEquals(offset + 0.6, result.ucl[0], 1e-3, "UCL_0 with large offset")
    }

    @Test
    fun testEwmaVeryLargeSigma() {
        // Very large sigma => proportionally wide control limits
        val obs = doubleArrayOf(10.0, 15.0, 5.0)
        val result = ewma(obs, target = 10.0, sigma = 1e6, lambda = 0.2, controlLimitWidth = 3.0)

        // Z values unaffected by sigma
        assertEquals(10.0, result.smoothedValues[0], tol, "Z_0 unaffected by sigma")
        assertEquals(11.0, result.smoothedValues[1], tol, "Z_1 unaffected by sigma")
        assertEquals(9.8, result.smoothedValues[2], tol, "Z_2 unaffected by sigma")

        // Control limits scale linearly with sigma
        assertEquals(600010.0, result.ucl[0], 1e-3, "UCL_0 scales with sigma")
        assertEquals(-599990.0, result.lcl[0], 1e-3, "LCL_0 scales with sigma")
    }

    @Test
    fun testEwmaVerySmallSigma() {
        // Very small sigma => narrow control limits, OOC fires on tiny deviation
        val obs = doubleArrayOf(10.0 + 1e-5)
        val result = ewma(obs, target = 10.0, sigma = 1e-10, lambda = 0.2, controlLimitWidth = 3.0)

        // Z_0 = 10 + 0.2*1e-5 = 10.000002
        assertTrue(result.smoothedValues[0].isFinite(), "Z_0 finite with small sigma")
        // UCL_0 = 10 + 3 * 1e-10 * 0.2 = 10 + 6e-11
        assertTrue(
            result.ucl[0] - 10.0 < 1e-9,
            "UCL very close to target for tiny sigma, got ${result.ucl[0]}"
        )
        // deviation far exceeds UCL, so OOC
        assertTrue(result.outOfControl.isNotEmpty(), "tiny deviation > tiny UCL => OOC")
    }

    @Test
    fun testEwmaLongSeries() {
        // Stress test: 10000 observations with no shift should stay in control
        // Expected ARL for lambda=0.2, L=3 is > 400, so 10000 will likely show some OOC
        // from random fluctuations in deterministic data, but not crash.
        val n = 10000
        val obs = DoubleArray(n) { 10.0 + 0.01 * kotlin.math.sin(it * 0.01) }
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)

        assertEquals(n, result.smoothedValues.size, "all Z values computed")
        for (i in 0 until n) {
            assertTrue(result.smoothedValues[i].isFinite(), "Z[$i] finite")
            assertTrue(result.ucl[i].isFinite(), "UCL[$i] finite")
            assertTrue(result.lcl[i].isFinite(), "LCL[$i] finite")
        }
        // Control limits should converge and stabilize at steady state
        val steadyUcl = 10.0 + 3.0 * sqrt(0.2 / 1.8)
        assertEquals(steadyUcl, result.ucl[n - 1], 1e-12, "UCL at steady state after 10000 points")
    }

    @Test
    fun testEwmaLargeControlLimitWidth() {
        // Very large L => no OOC ever (for bounded data)
        val obs = doubleArrayOf(0.0, 100.0, -100.0, 50.0, -50.0)
        val result = ewma(obs, target = 0.0, sigma = 1.0, lambda = 0.5, controlLimitWidth = 1e6)

        assertEquals(0, result.outOfControl.size, "very wide L => no OOC")
        for (i in obs.indices) {
            assertTrue(result.ucl[i].isFinite(), "UCL[$i] finite for large L")
        }
    }

    // ===== ewma: Non-finite input =====

    @Test
    fun testEwmaNaNInObservations() {
        // NaN propagates through Z recursion (z_{t-1} becomes NaN)
        val obs = doubleArrayOf(10.0, 10.1, Double.NaN, 10.3, 10.4)
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)

        // Before NaN: normal computation
        assertTrue(result.smoothedValues[0].isFinite(), "Z[0] finite before NaN")
        assertTrue(result.smoothedValues[1].isFinite(), "Z[1] finite before NaN")
        // NaN at index 2 and propagates thereafter
        assertTrue(result.smoothedValues[2].isNaN(), "Z[2] NaN")
        assertTrue(result.smoothedValues[3].isNaN(), "Z[3] NaN propagates")
        assertTrue(result.smoothedValues[4].isNaN(), "Z[4] NaN propagates")
        // Control limits unaffected by data NaN (depend only on target, sigma, lambda, L, t)
        assertTrue(result.ucl[2].isFinite(), "UCL[2] finite despite NaN in data")
        assertTrue(result.lcl[4].isFinite(), "LCL[4] finite despite NaN in data")
        // NaN never triggers OOC (NaN > x and NaN < x are both false per IEEE 754)
        assertTrue(2 !in result.outOfControl, "NaN index not flagged as OOC")
    }

    @Test
    fun testEwmaPositiveInfinityInObservations() {
        // +Infinity in data: Z becomes +Infinity from that point; +Infinity > UCL is true => OOC
        val obs = doubleArrayOf(10.0, Double.POSITIVE_INFINITY, 10.0)
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)

        assertEquals(10.0, result.smoothedValues[0], tol, "Z[0] before infinity")
        assertTrue(result.smoothedValues[1].isInfinite(), "Z[1] infinite")
        assertTrue(result.smoothedValues[1] > 0, "Z[1] positive infinity")
        // Z[2] = 0.2*10 + 0.8 * inf = inf (still)
        assertTrue(result.smoothedValues[2].isInfinite(), "Z[2] propagates infinity")
        // +infinity > UCL is true, triggers OOC
        assertTrue(1 in result.outOfControl, "+Infinity at i=1 => OOC")
    }

    @Test
    fun testEwmaNegativeInfinityInObservations() {
        val obs = doubleArrayOf(10.0, Double.NEGATIVE_INFINITY, 10.0)
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)

        assertTrue(result.smoothedValues[1].isInfinite(), "Z[1] infinite")
        assertTrue(result.smoothedValues[1] < 0, "Z[1] negative infinity")
        // -infinity < LCL is true, triggers OOC
        assertTrue(1 in result.outOfControl, "-Infinity at i=1 => OOC")
    }

    @Test
    fun testEwmaNaNTargetDoesNotThrow() {
        // NaN parameters pass validation (NaN comparisons are false per IEEE 754)
        val result = ewma(doubleArrayOf(1.0, 2.0, 3.0), target = Double.NaN, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
        for (i in 0 until 3) {
            assertTrue(result.smoothedValues[i].isNaN(), "Z[$i] NaN when target is NaN")
            assertTrue(result.ucl[i].isNaN(), "UCL[$i] NaN when target is NaN")
            assertTrue(result.lcl[i].isNaN(), "LCL[$i] NaN when target is NaN")
        }
        assertEquals(0, result.outOfControl.size, "NaN never OOC")
    }

    @Test
    fun testEwmaNaNSigmaDoesNotThrow() {
        // NaN sigma: sigma <= 0 is false for NaN, validation passes; limits become NaN
        val result = ewma(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, sigma = Double.NaN, lambda = 0.2, controlLimitWidth = 3.0)
        for (i in 0 until 3) {
            assertTrue(result.ucl[i].isNaN(), "UCL[$i] NaN when sigma is NaN")
            assertTrue(result.lcl[i].isNaN(), "LCL[$i] NaN when sigma is NaN")
        }
        // Z unaffected by sigma, finite
        assertTrue(result.smoothedValues[0].isFinite(), "Z unaffected by sigma")
    }

    @Test
    fun testEwmaNaNLambdaDoesNotThrow() {
        // NaN lambda: lambda <= 0 and lambda > 1 are both false for NaN; results become NaN
        val result = ewma(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, sigma = 1.0, lambda = Double.NaN, controlLimitWidth = 3.0)
        for (i in 0 until 3) {
            assertTrue(result.smoothedValues[i].isNaN(), "Z[$i] NaN with NaN lambda")
        }
    }

    @Test
    fun testEwmaNaNControlLimitWidthDoesNotThrow() {
        // NaN L: L <= 0 is false for NaN; limits become NaN
        val result = ewma(doubleArrayOf(1.0, 2.0, 3.0), target = 0.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = Double.NaN)
        // Z unaffected by L
        assertEquals(0.2, result.smoothedValues[0], tol, "Z unaffected by L")
        for (i in 0 until 3) {
            assertTrue(result.ucl[i].isNaN(), "UCL[$i] NaN with NaN L")
            assertTrue(result.lcl[i].isNaN(), "LCL[$i] NaN with NaN L")
        }
    }

    // ===== ewma: Property-based =====

    @Test
    fun testEwmaRecursiveDefinition() {
        // Property: Z_t = lambda*x_t + (1-lambda)*Z_{t-1}, Z_0 = target (before first observation)
        val obs = doubleArrayOf(10.1, 10.3, 10.5, 10.8, 11.0, 11.3, 9.5, 9.2, 10.7, 11.5)
        val target = 10.0
        val lambda = 0.25
        val result = ewma(obs, target = target, sigma = 1.0, lambda = lambda, controlLimitWidth = 3.0)

        // Z_0 = lambda*x_0 + (1-lambda)*target
        val z0 = lambda * obs[0] + (1.0 - lambda) * target
        assertEquals(z0, result.smoothedValues[0], tol, "Z_0 recursion with Z_{-1}=target")

        for (i in 1 until obs.size) {
            val expected = lambda * obs[i] + (1.0 - lambda) * result.smoothedValues[i - 1]
            assertEquals(expected, result.smoothedValues[i], tol, "Z[$i] recursion")
        }
    }

    @Test
    fun testEwmaControlLimitsSymmetricAroundTarget() {
        // Property: UCL - target == target - LCL for all t
        val obs = doubleArrayOf(10.5, 11.0, 9.5, 10.3, 10.8)
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.3, controlLimitWidth = 3.0)

        val target = 10.0
        for (i in obs.indices) {
            val upperSpread = result.ucl[i] - target
            val lowerSpread = target - result.lcl[i]
            assertEquals(upperSpread, lowerSpread, tol, "symmetry at i=$i")
        }
    }

    @Test
    fun testEwmaUclAlwaysGreaterThanLcl() {
        // UCL > LCL strictly for all t, sigma > 0, L > 0
        val obs = doubleArrayOf(10.0, 11.0, 9.5, 10.2, 10.8, 9.1)
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
        for (i in obs.indices) {
            assertTrue(result.ucl[i] > result.lcl[i], "UCL > LCL at i=$i")
        }
    }

    @Test
    fun testEwmaFirstObservationSigmaEqualsLambdaSigma() {
        // Closed-form identity: sigma_Z_1 = sigma * |lambda|
        // Because sigma_Z_1^2 = sigma^2 * lam/(2-lam) * (1 - (1-lam)^2)
        //                     = sigma^2 * lam/(2-lam) * (2lam - lam^2)
        //                     = sigma^2 * lam/(2-lam) * lam*(2 - lam)
        //                     = sigma^2 * lam^2
        // => UCL_0 - target = L * sigma * lambda
        for (lambda in listOf(0.05, 0.1, 0.2, 0.5, 0.9, 1.0)) {
            val result = ewma(doubleArrayOf(0.0), target = 0.0, sigma = 1.0, lambda = lambda, controlLimitWidth = 3.0)
            val expectedUcl = 3.0 * lambda
            assertEquals(expectedUcl, result.ucl[0], 1e-14, "UCL_0 - target = L*sigma*lambda for lambda=$lambda")
        }
    }

    @Test
    fun testEwmaControlLimitsMonotonicWidening() {
        // For lambda < 1: |UCL_t - target| strictly increases with t (approaches steady state from below)
        val obs = DoubleArray(20) { 0.0 }
        val lambda = 0.15
        val result = ewma(obs, target = 0.0, sigma = 1.0, lambda = lambda, controlLimitWidth = 3.0)

        for (i in 1 until obs.size) {
            assertTrue(
                result.ucl[i] > result.ucl[i - 1],
                "UCL strictly widens: UCL[$i]=${result.ucl[i]} > UCL[${i - 1}]=${result.ucl[i - 1]}"
            )
        }
    }

    @Test
    fun testEwmaLambdaOneHasConstantLimits() {
        // For lambda=1: UCL_t and LCL_t are constant for all t
        val obs = doubleArrayOf(0.0, 1.0, 2.0, 3.0, 4.0)
        val result = ewma(obs, target = 0.0, sigma = 2.0, lambda = 1.0, controlLimitWidth = 3.0)

        for (i in 1 until obs.size) {
            assertEquals(result.ucl[0], result.ucl[i], 0.0, "UCL constant at lambda=1")
            assertEquals(result.lcl[0], result.lcl[i], 0.0, "LCL constant at lambda=1")
        }
    }

    @Test
    fun testEwmaTranslationInvariance() {
        // Shifting observations AND target by c shifts Z, UCL, LCL by c; OOC indices unchanged
        val obs = doubleArrayOf(10.5, 11.0, 9.5, 10.3, 12.0, 8.5)
        val shift = 1000.0
        val shiftedObs = DoubleArray(obs.size) { obs[it] + shift }

        val original = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.3, controlLimitWidth = 3.0)
        val shifted = ewma(shiftedObs, target = 10.0 + shift, sigma = 1.0, lambda = 0.3, controlLimitWidth = 3.0)

        for (i in obs.indices) {
            assertEquals(original.smoothedValues[i] + shift, shifted.smoothedValues[i], 1e-7, "Z[$i] shifts by c")
            assertEquals(original.ucl[i] + shift, shifted.ucl[i], 1e-7, "UCL[$i] shifts by c")
            assertEquals(original.lcl[i] + shift, shifted.lcl[i], 1e-7, "LCL[$i] shifts by c")
        }
        assertTrue(
            original.outOfControl.contentEquals(shifted.outOfControl),
            "OOC indices unchanged by translation"
        )
    }

    @Test
    fun testEwmaScaleInvariance() {
        // If we scale (obs - target) by c and sigma by c, result scales similarly
        // Specifically: centered_Z scales by c; UCL/LCL offsets scale by c
        val obs = doubleArrayOf(10.5, 11.0, 9.5, 10.3, 12.0, 8.5)
        val target = 10.0
        val c = 5.0
        val scaledObs = DoubleArray(obs.size) { target + c * (obs[it] - target) }

        val original = ewma(obs, target = target, sigma = 1.0, lambda = 0.3, controlLimitWidth = 3.0)
        val scaled = ewma(scaledObs, target = target, sigma = c, lambda = 0.3, controlLimitWidth = 3.0)

        for (i in obs.indices) {
            val origOffset = original.smoothedValues[i] - target
            val scaledOffset = scaled.smoothedValues[i] - target
            assertEquals(c * origOffset, scaledOffset, 1e-10, "Z offset scales by c at i=$i")

            val origUclOffset = original.ucl[i] - target
            val scaledUclOffset = scaled.ucl[i] - target
            assertEquals(c * origUclOffset, scaledUclOffset, 1e-10, "UCL offset scales by c at i=$i")
        }
        assertTrue(
            original.outOfControl.contentEquals(scaled.outOfControl),
            "OOC indices unchanged by scale"
        )
    }

    @Test
    fun testEwmaSymmetryByNegation() {
        // Negating (obs - target) around target negates (Z - target) and swaps roles of UCL/LCL boundaries
        val obs = doubleArrayOf(10.5, 11.0, 9.5, 10.3, 12.0, 8.5)
        val target = 10.0
        val negObs = DoubleArray(obs.size) { target - (obs[it] - target) }

        val original = ewma(obs, target = target, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
        val negated = ewma(negObs, target = target, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)

        for (i in obs.indices) {
            // Z - target negates
            assertEquals(
                (original.smoothedValues[i] - target),
                -(negated.smoothedValues[i] - target),
                1e-10,
                "Z[$i] centered negates"
            )
            // UCL/LCL boundaries at same magnitude from target
            assertEquals(original.ucl[i], negated.ucl[i], tol, "UCL[$i] symmetric")
            assertEquals(original.lcl[i], negated.lcl[i], tol, "LCL[$i] symmetric")
        }
    }

    @Test
    fun testEwmaOutOfControlIsSortedAscending() {
        // outOfControl indices are appended in traversal order => sorted ascending
        val obs = doubleArrayOf(15.0, 15.0, 15.0, 15.0, 5.0, 5.0, 5.0, 5.0)
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.3, controlLimitWidth = 3.0)

        for (i in 1 until result.outOfControl.size) {
            assertTrue(
                result.outOfControl[i] > result.outOfControl[i - 1],
                "outOfControl[$i]=${result.outOfControl[i]} > outOfControl[${i - 1}]=${result.outOfControl[i - 1]}"
            )
        }
    }

    @Test
    fun testEwmaOutOfControlMatchesZOutsideLimits() {
        // Property: i ∈ outOfControl iff z[i] > ucl[i] or z[i] < lcl[i]
        val obs = doubleArrayOf(10.5, 12.0, 8.0, 11.0, 10.0, 13.0, 7.0, 10.5)
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.3, controlLimitWidth = 3.0)

        val expected = mutableListOf<Int>()
        for (i in obs.indices) {
            val z = result.smoothedValues[i]
            if (z > result.ucl[i] || z < result.lcl[i]) {
                expected.add(i)
            }
        }
        assertTrue(
            result.outOfControl.contentEquals(expected.toIntArray()),
            "outOfControl should be exactly {i : z[i] ∉ [lcl[i], ucl[i]]}, " +
                "expected ${expected} got ${result.outOfControl.toList()}"
        )
    }

    // ===== ewma: Overload consistency =====

    @Test
    fun testEwmaIterableOverload() {
        // Iterable overload should give identical result to DoubleArray overload
        val array = doubleArrayOf(10.1, 10.3, 10.5, 10.8, 11.0, 11.3)
        val list: Iterable<Double> = array.toList()
        val expected = ewma(array, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
        val actual = ewma(list, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)

        assertEquals(expected, actual, "Iterable overload should match DoubleArray overload")
    }

    @Test
    fun testEwmaSequenceOverload() {
        // Sequence overload should give identical result to DoubleArray overload
        val array = doubleArrayOf(10.1, 10.3, 10.5, 10.8, 11.0, 11.3)
        val seq: Sequence<Double> = array.asSequence()
        val expected = ewma(array, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
        val actual = ewma(seq, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)

        assertEquals(expected, actual, "Sequence overload should match DoubleArray overload")
    }

    // ===== EwmaResult: data class =====

    @Test
    fun testEwmaResultEquality() {
        val obs = doubleArrayOf(10.1, 10.3, 10.5, 10.8, 11.0, 11.3)
        val r1 = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
        val r2 = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)

        assertEquals(r1, r2, "Same input should produce equal EwmaResult")
        assertEquals(r1.hashCode(), r2.hashCode(), "Equal results should have equal hashCode")
    }

    @Test
    fun testEwmaResultEqualityDifferentInstances() {
        // equals uses contentEquals, so different array instances with same values are equal
        val r1 = EwmaResult(
            smoothedValues = doubleArrayOf(10.0, 10.2, 10.4),
            ucl = doubleArrayOf(10.6, 10.77, 10.86),
            lcl = doubleArrayOf(9.4, 9.23, 9.14),
            outOfControl = intArrayOf(),
        )
        val r2 = EwmaResult(
            smoothedValues = doubleArrayOf(10.0, 10.2, 10.4),
            ucl = doubleArrayOf(10.6, 10.77, 10.86),
            lcl = doubleArrayOf(9.4, 9.23, 9.14),
            outOfControl = intArrayOf(),
        )
        assertTrue(r1 !== r2, "Different instances")
        assertEquals(r1, r2, "Different array instances with same content should be equal")
        assertEquals(r1.hashCode(), r2.hashCode(), "hashCode consistent with equals")
    }

    @Test
    fun testEwmaResultInequality() {
        val r1 = EwmaResult(
            smoothedValues = doubleArrayOf(10.0, 10.2),
            ucl = doubleArrayOf(10.6, 10.77),
            lcl = doubleArrayOf(9.4, 9.23),
            outOfControl = intArrayOf(),
        )

        // Different smoothedValues
        val r2 = EwmaResult(
            smoothedValues = doubleArrayOf(10.1, 10.2),
            ucl = doubleArrayOf(10.6, 10.77),
            lcl = doubleArrayOf(9.4, 9.23),
            outOfControl = intArrayOf(),
        )
        assertTrue(r1 != r2, "Different smoothedValues => not equal")

        // Different ucl
        val r3 = EwmaResult(
            smoothedValues = doubleArrayOf(10.0, 10.2),
            ucl = doubleArrayOf(10.7, 10.77),
            lcl = doubleArrayOf(9.4, 9.23),
            outOfControl = intArrayOf(),
        )
        assertTrue(r1 != r3, "Different ucl => not equal")

        // Different lcl
        val r4 = EwmaResult(
            smoothedValues = doubleArrayOf(10.0, 10.2),
            ucl = doubleArrayOf(10.6, 10.77),
            lcl = doubleArrayOf(9.3, 9.23),
            outOfControl = intArrayOf(),
        )
        assertTrue(r1 != r4, "Different lcl => not equal")

        // Different outOfControl
        val r5 = EwmaResult(
            smoothedValues = doubleArrayOf(10.0, 10.2),
            ucl = doubleArrayOf(10.6, 10.77),
            lcl = doubleArrayOf(9.4, 9.23),
            outOfControl = intArrayOf(1),
        )
        assertTrue(r1 != r5, "Different outOfControl => not equal")
    }

    @Test
    fun testEwmaResultEqualsSelf() {
        val r = EwmaResult(
            smoothedValues = doubleArrayOf(10.0),
            ucl = doubleArrayOf(10.6),
            lcl = doubleArrayOf(9.4),
            outOfControl = intArrayOf(),
        )
        assertEquals(r, r, "equals with self")
    }

    @Test
    fun testEwmaResultEqualsNonEwmaResult() {
        val r = EwmaResult(
            smoothedValues = doubleArrayOf(10.0),
            ucl = doubleArrayOf(10.6),
            lcl = doubleArrayOf(9.4),
            outOfControl = intArrayOf(),
        )
        assertTrue(!r.equals("not an EwmaResult"), "equals returns false for non-EwmaResult")
        assertTrue(!r.equals(null), "equals returns false for null")
    }

    @Test
    fun testEwmaResultDestructuring() {
        // componentN (destructuring) works on the data class
        val obs = doubleArrayOf(10.1, 10.3, 10.5)
        val result = ewma(obs, target = 10.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
        val (smoothedValues, ucl, lcl, outOfControl) = result

        assertEquals(result.smoothedValues.size, smoothedValues.size, "smoothedValues via destructuring")
        assertEquals(result.ucl.size, ucl.size, "ucl via destructuring")
        assertEquals(result.lcl.size, lcl.size, "lcl via destructuring")
        assertEquals(result.outOfControl.size, outOfControl.size, "outOfControl via destructuring")
    }

    @Test
    fun testEwmaResultToStringRendersArrayContents() {
        // toString must use contentToString() for DoubleArray/IntArray fields — the default
        // data-class toString would print `[D@<hash>` which is useless for diagnostics.
        val result = EwmaResult(
            smoothedValues = doubleArrayOf(10.0, 10.2, 10.4),
            ucl = doubleArrayOf(10.6, 10.77, 10.86),
            lcl = doubleArrayOf(9.4, 9.23, 9.14),
            outOfControl = intArrayOf(2),
        )
        val s = result.toString()
        assertTrue(s.contains("smoothedValues=[10.0, 10.2, 10.4]"), "toString should render smoothedValues, got: $s")
        assertTrue(s.contains("ucl=[10.6, 10.77, 10.86]"), "toString should render ucl, got: $s")
        assertTrue(s.contains("lcl=[9.4, 9.23, 9.14]"), "toString should render lcl, got: $s")
        assertTrue(s.contains("outOfControl=[2]"), "toString should render outOfControl, got: $s")
        assertTrue(!s.contains("[D@"), "toString must not leak default array identity, got: $s")
        assertTrue(!s.contains("[I@"), "toString must not leak default int-array identity, got: $s")
    }

    @Test
    fun testEwmaResultToStringEmptyOutOfControl() {
        val result = EwmaResult(
            smoothedValues = doubleArrayOf(10.0),
            ucl = doubleArrayOf(10.6),
            lcl = doubleArrayOf(9.4),
            outOfControl = intArrayOf(),
        )
        val s = result.toString()
        assertTrue(s.contains("outOfControl=[]"), "toString should render empty outOfControl, got: $s")
    }

    // ===== westernElectricRules: Basic correctness =====

    @Test
    fun testWesternElectricRulesRule1UpperTrigger() {
        // Reference: single point beyond +3σ triggers Rule 1.
        val obs = doubleArrayOf(0.1, 0.2, 0.0, 3.5, 0.1)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.contentEquals(intArrayOf(3)), "rule1 should flag index 3, got ${r.rule1.contentToString()}")
        assertTrue(r.rule2.isEmpty(), "rule2 should be empty, got ${r.rule2.contentToString()}")
        assertTrue(r.rule3.isEmpty(), "rule3 should be empty, got ${r.rule3.contentToString()}")
        assertTrue(r.rule4.isEmpty(), "rule4 should be empty, got ${r.rule4.contentToString()}")
    }

    @Test
    fun testWesternElectricRulesRule1LowerTrigger() {
        // Reference: single point beyond -3σ triggers Rule 1.
        val obs = doubleArrayOf(0.1, -3.5, 0.0, 0.2, 0.1)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.contentEquals(intArrayOf(1)), "rule1 should flag index 1, got ${r.rule1.contentToString()}")
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesRule2UpperTrigger() {
        // Reference: 2 of last 3 points beyond +2σ triggers Rule 2.
        //   obs = [0.1, 2.5, 0.0, 2.3, 0.1]  (sigma=1, center=0)
        //   i=3: window {2.5, 0.0, 2.3} → 2 above +2σ → rule2 fires at 3
        val obs = doubleArrayOf(0.1, 2.5, 0.0, 2.3, 0.1)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.contentEquals(intArrayOf(3)), "rule2 should flag index 3, got ${r.rule2.contentToString()}")
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesRule2TwoConsecutive() {
        // Reference: [0.1, 2.5, 2.3, 0.0, 0.1]
        //   i=2: {0.1, 2.5, 2.3} → 2 above → fires
        //   i=3: {2.5, 2.3, 0.0} → 2 above → fires
        val obs = doubleArrayOf(0.1, 2.5, 2.3, 0.0, 0.1)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.contentEquals(intArrayOf(2, 3)), "got ${r.rule2.contentToString()}")
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesRule2LowerTrigger() {
        // Reference: [0.1, -2.5, -2.3, 0.0, 0.1]
        //   i=2, 3: ≥2 below -2σ → rule2 fires
        val obs = doubleArrayOf(0.1, -2.5, -2.3, 0.0, 0.1)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.contentEquals(intArrayOf(2, 3)))
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesRule2CrossSide() {
        // Reference: 1 above and 1 below does NOT trigger Rule 2 — same-side requirement.
        //   [0.1, 2.5, 0.0, -2.3, 0.1]: no window with 2 same-side points beyond 2σ.
        val obs = doubleArrayOf(0.1, 2.5, 0.0, -2.3, 0.1)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty(), "Rule 2 requires same side, got ${r.rule2.contentToString()}")
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesRule3UpperTrigger() {
        // Reference: 4 of last 5 points beyond +1σ triggers Rule 3.
        //   [0.1, 1.5, 1.3, 0.2, 1.2, 1.4, 0.1] (sigma=1)
        //   i=5: window 1..5 = {1.5, 1.3, 0.2, 1.2, 1.4} → 4 above → rule3 fires
        val obs = doubleArrayOf(0.1, 1.5, 1.3, 0.2, 1.2, 1.4, 0.1)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.contentEquals(intArrayOf(5)), "got ${r.rule3.contentToString()}")
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesRule3MinimumTrigger() {
        // Reference: Rule 3 can trigger at i=4 (first index where window of 5 exists).
        //   [1.5, 1.3, 0.0, 1.2, 1.4]: 4 of 5 above +1σ → rule3 fires at 4.
        val obs = doubleArrayOf(1.5, 1.3, 0.0, 1.2, 1.4)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.contentEquals(intArrayOf(4)))
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesRule3Sliding() {
        // Reference: rule3 with sliding windows.
        //   [1.5, 0.0, 1.3, 1.2, 1.4, 0.0, 1.1]:
        //   i=4: win 0..4 {1.5, 0.0, 1.3, 1.2, 1.4} → 4 above → fires
        //   i=5: win 1..5 {0.0, 1.3, 1.2, 1.4, 0.0} → 3 above → no
        //   i=6: win 2..6 {1.3, 1.2, 1.4, 0.0, 1.1} → 4 above → fires
        val obs = doubleArrayOf(1.5, 0.0, 1.3, 1.2, 1.4, 0.0, 1.1)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.contentEquals(intArrayOf(4, 6)), "got ${r.rule3.contentToString()}")
    }

    @Test
    fun testWesternElectricRulesRule4AllAbove() {
        // Reference: 8 consecutive above center triggers Rule 4 at i=7.
        val obs = doubleArrayOf(0.5, 0.3, 0.6, 0.4, 0.2, 0.8, 0.7, 0.5)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.contentEquals(intArrayOf(7)), "got ${r.rule4.contentToString()}")
    }

    @Test
    fun testWesternElectricRulesRule4AllBelow() {
        // Reference: 8 consecutive below center triggers Rule 4 at i=7.
        val obs = doubleArrayOf(-0.5, -0.3, -0.6, -0.4, -0.2, -0.8, -0.7, -0.5)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.contentEquals(intArrayOf(7)))
    }

    @Test
    fun testWesternElectricRulesRule4BrokenByCenterValue() {
        // Reference: an exact-center value breaks the Rule-4 streak (strict inequality).
        //   [0.5, 0.3, 0.6, 0.4, 0.0, 0.8, 0.7, 0.5, 0.6]: 0.0 at index 4 → no rule 4.
        val obs = doubleArrayOf(0.5, 0.3, 0.6, 0.4, 0.0, 0.8, 0.7, 0.5, 0.6)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule4.isEmpty(), "Strict inequality: value at center breaks streak, got ${r.rule4.contentToString()}")
    }

    @Test
    fun testWesternElectricRulesRule4Sliding() {
        // Reference: 9 consecutive above center → Rule 4 fires at i=7 and i=8.
        val obs = DoubleArray(9) { 0.1 }
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule4.contentEquals(intArrayOf(7, 8)), "got ${r.rule4.contentToString()}")
    }

    @Test
    fun testWesternElectricRulesRule4DelayedStreak() {
        // Reference: streak starts at index 3, fires once streak length reaches 8.
        //   [0.0, 0.0, -0.1, 0.5×8]: at i=10 we have 8 consecutive above-center values.
        val obs = doubleArrayOf(0.0, 0.0, -0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule4.contentEquals(intArrayOf(10)), "got ${r.rule4.contentToString()}")
    }

    @Test
    fun testWesternElectricRulesNonZeroCenterAndSigma() {
        // Reference: center=10, sigma=2. Thresholds: ±1σ → [8,12], ±2σ → [6,14], ±3σ → [4,16].
        //   [10, 15, 12.5, 13.5, 12.1, 11.8, 11.9, 11.5, 11.2]:
        //   i=4: win 0..4 = {10, 15, 12.5, 13.5, 12.1} → 4 above +1σ=12 → rule3
        //   i=5: win 1..5 = {15, 12.5, 13.5, 12.1, 11.8} → 4 above 12 → rule3
        //   i=8: obs[1..8] = {15, 12.5, 13.5, 12.1, 11.8, 11.9, 11.5, 11.2} all > 10 → rule4
        val obs = doubleArrayOf(10.0, 15.0, 12.5, 13.5, 12.1, 11.8, 11.9, 11.5, 11.2)
        val r = westernElectricRules(obs, center = 10.0, sigma = 2.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.contentEquals(intArrayOf(4, 5)), "got ${r.rule3.contentToString()}")
        assertTrue(r.rule4.contentEquals(intArrayOf(8)), "got ${r.rule4.contentToString()}")
    }

    @Test
    fun testWesternElectricRulesAllFourRulesFire() {
        // Reference: comprehensive example where all four rules trigger.
        //   Observations: 0.1, 0.2, 0.3, 0.1, 3.5, 2.5, 2.3, 1.5, 1.3, 0.0, 1.2, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.6
        //   Expected (python reference):
        //     rule1 = [4]
        //     rule2 = [5, 6, 7]
        //     rule3 = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        //     rule4 = [7, 8, 17]
        val obs = doubleArrayOf(
            0.1, 0.2, 0.3, 0.1,
            3.5,
            2.5, 2.3,
            1.5, 1.3, 0.0, 1.2, 1.4,
            1.1, 1.2, 1.5, 1.3, 1.4, 1.6,
        )
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.contentEquals(intArrayOf(4)), "rule1: ${r.rule1.contentToString()}")
        assertTrue(r.rule2.contentEquals(intArrayOf(5, 6, 7)), "rule2: ${r.rule2.contentToString()}")
        assertTrue(
            r.rule3.contentEquals(intArrayOf(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17)),
            "rule3: ${r.rule3.contentToString()}",
        )
        assertTrue(r.rule4.contentEquals(intArrayOf(7, 8, 17)), "rule4: ${r.rule4.contentToString()}")
    }

    @Test
    fun testWesternElectricRulesRule1AlsoCountsTowardRule2() {
        // A point beyond +3σ is also beyond +2σ, so it contributes to Rule 2 as well.
        //   [3.5, 3.5, 3.5]: rule1 at 0,1,2; rule2 at 2 (3-of-3 ≥ 2 above +2σ).
        val obs = doubleArrayOf(3.5, 3.5, 3.5)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.contentEquals(intArrayOf(0, 1, 2)))
        assertTrue(r.rule2.contentEquals(intArrayOf(2)))
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesMultipleRule1Alarms() {
        // Reference: three points beyond ±3σ, alternating sides.
        val obs = doubleArrayOf(3.5, 0.0, -3.5, 0.0, 3.5)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.contentEquals(intArrayOf(0, 2, 4)))
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesRule2WithOppositeExcursionsInWindow() {
        // Reference: Within a window, 2 above +2σ AND 1 below −2σ should still trigger
        // (the "same-side" requirement counts per direction — 2 above already qualifies).
        //   [2.5, -2.3, 2.4]: above=2, below=1 → rule2 fires at i=2.
        val obs = doubleArrayOf(2.5, -2.3, 2.4)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule2.contentEquals(intArrayOf(2)))
    }

    // ===== westernElectricRules: Edge cases =====

    @Test
    fun testWesternElectricRulesSingleObservationBeyond3Sigma() {
        // Reference: single observation, Rule 1 can still fire (no window requirement).
        val obs = doubleArrayOf(5.0)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.contentEquals(intArrayOf(0)))
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesSingleObservationInControl() {
        // Reference: single observation within control → no alarms.
        val obs = doubleArrayOf(0.5)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesTwoObservations() {
        // Only rules 1 can apply with 2 observations (rule 2 requires window size 3).
        val obs = doubleArrayOf(3.5, 0.0)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.contentEquals(intArrayOf(0)))
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesThreeObservationsRule2Minimum() {
        // Minimum size for Rule 2: window of 3 → rule 2 can fire exactly at i=2.
        val obs = doubleArrayOf(2.5, 0.0, 2.3)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule2.contentEquals(intArrayOf(2)))
    }

    @Test
    fun testWesternElectricRulesSevenObservationsNoRule4() {
        // Reference: 7 observations all above center cannot fire Rule 4 (needs 8).
        val obs = DoubleArray(7) { 0.5 }
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule4.isEmpty(), "Rule 4 needs 8 consecutive, got ${r.rule4.contentToString()}")
    }

    @Test
    fun testWesternElectricRulesExactlyAt3SigmaBoundary() {
        // Strict inequality: x = +3σ exactly does NOT trigger Rule 1.
        val obs = doubleArrayOf(3.0, 3.0)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty(), "Strict inequality at +3σ boundary: ${r.rule1.contentToString()}")
    }

    @Test
    fun testWesternElectricRulesExactlyAt2SigmaBoundary() {
        // Strict inequality at +2σ: no Rule 2 trigger.
        val obs = doubleArrayOf(2.0, 2.0, 2.0)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule2.isEmpty())
    }

    @Test
    fun testWesternElectricRulesAllAtCenterNoRule4() {
        // Reference: 8 values exactly at center → strict inequality prevents Rule 4.
        val obs = DoubleArray(8) { 0.0 }
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule4.isEmpty(), "Strict above/below: values at center should not trigger")
    }

    @Test
    fun testWesternElectricRulesRule4BrokenBy7ThenDownstep() {
        // Reference: 7 above, then below → streak broken, no rule 4.
        val obs = doubleArrayOf(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.1)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule4.isEmpty())
    }

    // ===== westernElectricRules: Degenerate input =====

    @Test
    fun testWesternElectricRulesEmptyArray() {
        assertFailsWith<InsufficientDataException> {
            westernElectricRules(doubleArrayOf(), center = 0.0, sigma = 1.0)
        }
    }

    @Test
    fun testWesternElectricRulesEmptyIterable() {
        assertFailsWith<InsufficientDataException> {
            westernElectricRules(emptyList<Double>(), center = 0.0, sigma = 1.0)
        }
    }

    @Test
    fun testWesternElectricRulesEmptySequence() {
        assertFailsWith<InsufficientDataException> {
            westernElectricRules(emptySequence<Double>(), center = 0.0, sigma = 1.0)
        }
    }

    @Test
    fun testWesternElectricRulesZeroSigma() {
        assertFailsWith<InvalidParameterException> {
            westernElectricRules(doubleArrayOf(1.0, 2.0), center = 0.0, sigma = 0.0)
        }
    }

    @Test
    fun testWesternElectricRulesNegativeSigma() {
        assertFailsWith<InvalidParameterException> {
            westernElectricRules(doubleArrayOf(1.0, 2.0), center = 0.0, sigma = -1.0)
        }
    }

    @Test
    fun testWesternElectricRulesNegativeSigmaIterable() {
        assertFailsWith<InvalidParameterException> {
            westernElectricRules(listOf(1.0, 2.0), center = 0.0, sigma = -1.0)
        }
    }

    @Test
    fun testWesternElectricRulesNegativeSigmaSequence() {
        assertFailsWith<InvalidParameterException> {
            westernElectricRules(sequenceOf(1.0, 2.0), center = 0.0, sigma = -1.0)
        }
    }

    @Test
    fun testWesternElectricRulesAllConstantAtCenter() {
        // Constant data at the center: no excursions, but Rule 4 must not fire because
        // values are not strictly above or below center.
        val obs = DoubleArray(20) { 0.0 }
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty(), "Strict: constant-at-center should not trigger rule 4")
    }

    @Test
    fun testWesternElectricRulesAllConstantOffCenter() {
        // Constant data away from center: Rule 4 should fire once the streak reaches 8.
        val n = 12
        val obs = DoubleArray(n) { 0.5 }
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.contentEquals(intArrayOf(7, 8, 9, 10, 11)))
    }

    // ===== westernElectricRules: Extreme parameters =====

    @Test
    fun testWesternElectricRulesLargeOffsetData() {
        // Reference: very large center, small sigma. Thresholds well above 0.
        val obs = doubleArrayOf(1e6 + 3.5, 1e6 + 0.0, 1e6 + 0.0)
        val r = westernElectricRules(obs, center = 1e6, sigma = 1.0)
        assertTrue(r.rule1.contentEquals(intArrayOf(0)), "rule1: ${r.rule1.contentToString()}")
        assertTrue(r.rule2.isEmpty())
    }

    @Test
    fun testWesternElectricRulesVerySmallSigma() {
        // Reference: very small sigma → every non-zero point triggers Rule 1.
        val obs = doubleArrayOf(1.0, -1.0, 0.5)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1e-10)
        assertTrue(r.rule1.contentEquals(intArrayOf(0, 1, 2)))
        assertTrue(r.rule2.contentEquals(intArrayOf(2)))
    }

    @Test
    fun testWesternElectricRulesVeryLargeSigma() {
        // Reference: huge sigma → no ±σ alarms, but rule 4 can still fire (center-based).
        val obs = doubleArrayOf(1.0, -1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1000.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        // Rule 4 starts streak at i=2 (all indices 2..9 are > 0) → fires at i=9.
        assertTrue(r.rule4.contentEquals(intArrayOf(9)), "got ${r.rule4.contentToString()}")
    }

    @Test
    fun testWesternElectricRulesLongSeries() {
        // Many observations, no rules should fire (all within ±1σ and alternating sides).
        val obs = DoubleArray(500) { if (it % 2 == 0) 0.5 else -0.5 }
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    // ===== westernElectricRules: Non-finite input =====

    @Test
    fun testWesternElectricRulesNaNInObservations() {
        // Reference: NaN never satisfies strict inequality comparisons → never contributes
        // to a rule violation. With only NaN interspersed with near-zero values, no
        // rule should fire.
        val obs = doubleArrayOf(0.1, Double.NaN, 0.2, Double.NaN, 0.3)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesNaNBreaksRule4Streak() {
        // Reference: NaN in the middle of a streak breaks it (NaN is neither above nor below).
        //   [0.5, 0.5, 0.5, 0.5, NaN, 0.5, 0.5, 0.5]: NaN at i=4 breaks the streak.
        val obs = doubleArrayOf(0.5, 0.5, 0.5, 0.5, Double.NaN, 0.5, 0.5, 0.5)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule4.isEmpty(), "NaN must break rule 4 streak, got ${r.rule4.contentToString()}")
    }

    @Test
    fun testWesternElectricRulesPositiveInfinityInObservations() {
        // Reference: +Infinity > center+3σ → Rule 1 fires at that index.
        val obs = doubleArrayOf(Double.POSITIVE_INFINITY, 0.0, 0.0)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.contentEquals(intArrayOf(0)))
    }

    @Test
    fun testWesternElectricRulesNegativeInfinityInObservations() {
        // Reference: -Infinity < center-3σ → Rule 1 fires at that index.
        val obs = doubleArrayOf(Double.NEGATIVE_INFINITY, 0.0, 0.0)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.contentEquals(intArrayOf(0)))
    }

    @Test
    fun testWesternElectricRulesNaNCenterDoesNotThrow() {
        // NaN validation intentionally passes (NaN <= 0.0 is false per IEEE 754).
        // All comparisons against NaN thresholds are false, so no rules fire.
        val obs = doubleArrayOf(1.0, 2.0, 3.0)
        val r = westernElectricRules(obs, center = Double.NaN, sigma = 1.0)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        assertTrue(r.rule4.isEmpty())
    }

    @Test
    fun testWesternElectricRulesNaNSigmaDoesNotThrow() {
        // NaN sigma passes validation (NaN <= 0.0 is false per IEEE 754).
        // Thresholds become NaN → no rules fire.
        val obs = doubleArrayOf(1.0, 2.0, 3.0)
        val r = westernElectricRules(obs, center = 0.0, sigma = Double.NaN)
        assertTrue(r.rule1.isEmpty())
        assertTrue(r.rule2.isEmpty())
        assertTrue(r.rule3.isEmpty())
        // Rule 4 operates on center (not sigma); with all obs > 0 and finite center=0, it still fires.
        // But here n=3 < 8, so rule4 is empty anyway.
        assertTrue(r.rule4.isEmpty())
    }

    // ===== westernElectricRules: Property-based =====

    @Test
    fun testWesternElectricRulesTranslationInvariance() {
        // Shifting both observations and center by the same constant should not change
        // any of the rule-violation indices.
        val baseObs = doubleArrayOf(0.5, 0.3, 0.6, 0.4, 0.2, 0.8, 0.7, 0.5)
        val shift = 100.0
        val shiftedObs = DoubleArray(baseObs.size) { baseObs[it] + shift }
        val r1 = westernElectricRules(baseObs, center = 0.0, sigma = 1.0)
        val r2 = westernElectricRules(shiftedObs, center = shift, sigma = 1.0)
        assertTrue(r1.rule1.contentEquals(r2.rule1), "rule1 translation invariant")
        assertTrue(r1.rule2.contentEquals(r2.rule2), "rule2 translation invariant")
        assertTrue(r1.rule3.contentEquals(r2.rule3), "rule3 translation invariant")
        assertTrue(r1.rule4.contentEquals(r2.rule4), "rule4 translation invariant")
    }

    @Test
    fun testWesternElectricRulesScaleInvariance() {
        // Scaling observations, center, and sigma by the same positive factor should
        // produce the same rule violation indices.
        val baseObs = doubleArrayOf(3.5, 2.5, 1.5, -0.5, 2.3, 1.4, 1.3, 1.2, 3.5)
        val scale = 10.0
        val scaledObs = DoubleArray(baseObs.size) { baseObs[it] * scale }
        val r1 = westernElectricRules(baseObs, center = 0.0, sigma = 1.0)
        val r2 = westernElectricRules(scaledObs, center = 0.0, sigma = 1.0 * scale)
        assertTrue(r1.rule1.contentEquals(r2.rule1), "rule1 scale invariant")
        assertTrue(r1.rule2.contentEquals(r2.rule2), "rule2 scale invariant")
        assertTrue(r1.rule3.contentEquals(r2.rule3), "rule3 scale invariant")
        assertTrue(r1.rule4.contentEquals(r2.rule4), "rule4 scale invariant")
    }

    @Test
    fun testWesternElectricRulesSymmetryByNegation() {
        // Negating observations (with center=0) should produce identical rule-index
        // structure: upper excursions become lower excursions, but the count and index
        // layout is preserved.
        val obs = doubleArrayOf(0.5, 2.5, 2.3, 3.5, 0.0, 1.5, 1.3, 1.2, 1.4)
        val negObs = DoubleArray(obs.size) { -obs[it] }
        val r1 = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        val r2 = westernElectricRules(negObs, center = 0.0, sigma = 1.0)
        assertTrue(r1.rule1.contentEquals(r2.rule1), "rule1 mirrored")
        assertTrue(r1.rule2.contentEquals(r2.rule2), "rule2 mirrored")
        assertTrue(r1.rule3.contentEquals(r2.rule3), "rule3 mirrored")
        assertTrue(r1.rule4.contentEquals(r2.rule4), "rule4 mirrored")
    }

    @Test
    fun testWesternElectricRulesIndicesAreSortedAscending() {
        // Each rule array must be in strictly ascending order (indices reported in-order).
        val obs = doubleArrayOf(
            3.5, -3.5, 2.5, 2.3, 0.0, 1.5, 1.3, 1.2, 1.4, 1.6, 1.7, 1.8, 1.1,
        )
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        for (arr in listOf(r.rule1, r.rule2, r.rule3, r.rule4)) {
            for (i in 1 until arr.size) {
                assertTrue(arr[i] > arr[i - 1], "indices must be strictly ascending: ${arr.contentToString()}")
            }
        }
    }

    @Test
    fun testWesternElectricRulesIndicesInRange() {
        // All reported indices must fall in [0, n).
        val obs = doubleArrayOf(
            3.5, 2.5, 2.3, 1.5, 1.3, 1.2, 1.4, 1.1, 0.9, 0.8, 0.7, -3.5,
        )
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        val n = obs.size
        for (arr in listOf(r.rule1, r.rule2, r.rule3, r.rule4)) {
            for (idx in arr) {
                assertTrue(idx in 0 until n, "index $idx out of range [0, $n)")
            }
        }
    }

    @Test
    fun testWesternElectricRulesRule2RequiresWindowOfThree() {
        // Rule 2 cannot fire at indices 0 or 1 (window of 3 not yet available).
        val obs = doubleArrayOf(3.5, 3.5, 3.5, 3.5)  // all above +3σ
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        for (idx in r.rule2) {
            assertTrue(idx >= 2, "Rule 2 index must be ≥ 2, got $idx")
        }
    }

    @Test
    fun testWesternElectricRulesRule3RequiresWindowOfFive() {
        // Rule 3 cannot fire at indices 0..3.
        val obs = DoubleArray(10) { 1.5 }
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        for (idx in r.rule3) {
            assertTrue(idx >= 4, "Rule 3 index must be ≥ 4, got $idx")
        }
    }

    @Test
    fun testWesternElectricRulesRule4RequiresWindowOfEight() {
        // Rule 4 cannot fire at indices 0..6.
        val obs = DoubleArray(15) { 0.5 }
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        for (idx in r.rule4) {
            assertTrue(idx >= 7, "Rule 4 index must be ≥ 7, got $idx")
        }
    }

    @Test
    fun testWesternElectricRulesRule1SubsetConsistency() {
        // A Rule 1 violation at index i (|x| > 3σ) implies that point contributes
        // to the counts in any window containing i for rules 2 and 3. When we have
        // enough consecutive 3σ violations, rule 2 must fire at some point too.
        val obs = doubleArrayOf(3.5, 3.5, 3.5, 3.5, 3.5)
        val r = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertTrue(r.rule1.size == 5, "All 5 are beyond 3σ")
        assertTrue(r.rule2.isNotEmpty(), "Consecutive 3σ alarms must also trigger rule 2")
        assertTrue(r.rule3.isNotEmpty(), "5 consecutive 3σ alarms must also trigger rule 3")
    }

    // ===== westernElectricRules: Iterable/Sequence overloads =====

    @Test
    fun testWesternElectricRulesIterableOverload() {
        // Iterable overload must produce the same result as DoubleArray for identical values.
        val arr = doubleArrayOf(0.1, 0.2, 0.3, 0.5, 2.5, 2.3, 0.1, 3.5)
        val iter: List<Double> = arr.toList()
        val r1 = westernElectricRules(arr, center = 0.0, sigma = 1.0)
        val r2 = westernElectricRules(iter, center = 0.0, sigma = 1.0)
        assertEquals(r1, r2, "Iterable overload must match DoubleArray result")
    }

    @Test
    fun testWesternElectricRulesSequenceOverload() {
        // Sequence overload must produce the same result as DoubleArray for identical values.
        val arr = doubleArrayOf(0.1, 0.2, 0.3, 0.5, 2.5, 2.3, 0.1, 3.5)
        val seq: Sequence<Double> = arr.toList().asSequence()
        val r1 = westernElectricRules(arr, center = 0.0, sigma = 1.0)
        val r2 = westernElectricRules(seq, center = 0.0, sigma = 1.0)
        assertEquals(r1, r2, "Sequence overload must match DoubleArray result")
    }

    // ===== WesternElectricRulesResult: data class =====

    @Test
    fun testWesternElectricRulesResultEquality() {
        val obs = doubleArrayOf(3.5, 2.5, 2.3, 1.5, 1.3, 1.2, 1.4, 1.1)
        val r1 = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        val r2 = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        assertEquals(r1, r2, "Same input should produce equal WesternElectricRulesResult")
        assertEquals(r1.hashCode(), r2.hashCode(), "Equal results should have equal hashCode")
    }

    @Test
    fun testWesternElectricRulesResultEqualityDifferentInstances() {
        // equals uses contentEquals — different IntArray instances with same values are equal.
        val r1 = WesternElectricRulesResult(
            rule1 = intArrayOf(0),
            rule2 = intArrayOf(2, 3),
            rule3 = intArrayOf(),
            rule4 = intArrayOf(7),
        )
        val r2 = WesternElectricRulesResult(
            rule1 = intArrayOf(0),
            rule2 = intArrayOf(2, 3),
            rule3 = intArrayOf(),
            rule4 = intArrayOf(7),
        )
        assertTrue(r1 !== r2, "Different instances")
        assertEquals(r1, r2)
        assertEquals(r1.hashCode(), r2.hashCode())
    }

    @Test
    fun testWesternElectricRulesResultInequality() {
        val base = WesternElectricRulesResult(
            rule1 = intArrayOf(0),
            rule2 = intArrayOf(2),
            rule3 = intArrayOf(4),
            rule4 = intArrayOf(7),
        )

        val diffRule1 = WesternElectricRulesResult(
            rule1 = intArrayOf(1),
            rule2 = intArrayOf(2),
            rule3 = intArrayOf(4),
            rule4 = intArrayOf(7),
        )
        assertTrue(base != diffRule1, "Different rule1 => not equal")

        val diffRule2 = WesternElectricRulesResult(
            rule1 = intArrayOf(0),
            rule2 = intArrayOf(3),
            rule3 = intArrayOf(4),
            rule4 = intArrayOf(7),
        )
        assertTrue(base != diffRule2, "Different rule2 => not equal")

        val diffRule3 = WesternElectricRulesResult(
            rule1 = intArrayOf(0),
            rule2 = intArrayOf(2),
            rule3 = intArrayOf(5),
            rule4 = intArrayOf(7),
        )
        assertTrue(base != diffRule3, "Different rule3 => not equal")

        val diffRule4 = WesternElectricRulesResult(
            rule1 = intArrayOf(0),
            rule2 = intArrayOf(2),
            rule3 = intArrayOf(4),
            rule4 = intArrayOf(8),
        )
        assertTrue(base != diffRule4, "Different rule4 => not equal")
    }

    @Test
    fun testWesternElectricRulesResultEqualsSelf() {
        val r = WesternElectricRulesResult(
            rule1 = intArrayOf(0),
            rule2 = intArrayOf(),
            rule3 = intArrayOf(),
            rule4 = intArrayOf(),
        )
        assertEquals(r, r, "equals with self")
    }

    @Test
    fun testWesternElectricRulesResultEqualsNonResult() {
        val r = WesternElectricRulesResult(
            rule1 = intArrayOf(0),
            rule2 = intArrayOf(),
            rule3 = intArrayOf(),
            rule4 = intArrayOf(),
        )
        assertTrue(!r.equals("not a WesternElectricRulesResult"), "equals false for non-Result")
        assertTrue(!r.equals(null), "equals false for null")
    }

    @Test
    fun testWesternElectricRulesResultDestructuring() {
        // componentN (destructuring) works on the data class.
        val obs = doubleArrayOf(3.5, 2.5, 2.3)
        val result = westernElectricRules(obs, center = 0.0, sigma = 1.0)
        val (rule1, rule2, rule3, rule4) = result

        assertTrue(rule1.contentEquals(result.rule1), "rule1 via destructuring")
        assertTrue(rule2.contentEquals(result.rule2), "rule2 via destructuring")
        assertTrue(rule3.contentEquals(result.rule3), "rule3 via destructuring")
        assertTrue(rule4.contentEquals(result.rule4), "rule4 via destructuring")
    }

    @Test
    fun testWesternElectricRulesResultToStringRendersArrayContents() {
        // toString must use contentToString() for IntArray fields — the default data-class
        // toString would print `[I@<hash>` which is useless for diagnostics.
        val r = WesternElectricRulesResult(
            rule1 = intArrayOf(0, 4),
            rule2 = intArrayOf(3),
            rule3 = intArrayOf(),
            rule4 = intArrayOf(7, 8),
        )
        val s = r.toString()
        assertTrue(s.contains("rule1=[0, 4]"), "toString should render rule1, got: $s")
        assertTrue(s.contains("rule2=[3]"), "toString should render rule2, got: $s")
        assertTrue(s.contains("rule3=[]"), "toString should render empty rule3, got: $s")
        assertTrue(s.contains("rule4=[7, 8]"), "toString should render rule4, got: $s")
        assertTrue(!s.contains("[I@"), "toString must not leak default int-array identity, got: $s")
    }
}
