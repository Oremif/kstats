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
}
