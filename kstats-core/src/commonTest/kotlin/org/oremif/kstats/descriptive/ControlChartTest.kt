package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
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
}
