package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException

// ── SPC Constants ──────────────────────────────────────────────────────────────

/**
 * Standard Statistical Process Control (SPC) constants for a given subgroup size.
 *
 * These constants are used to compute control limits for x-bar, R, and S control charts.
 * They are derived from the sampling distribution of the range and standard deviation
 * of normally distributed data. Values are tabulated for subgroup sizes 2 through 25.
 *
 * ### Example:
 * ```kotlin
 * val c = spcConstants(5)
 * c.a2  // 0.577 — used for x̄-R chart limits
 * c.a3  // 1.427 — used for x̄-S chart limits
 * c.c4  // 0.9400 — bias-correction factor for sample standard deviation
 * ```
 *
 * @property a2 factor for computing x-bar chart control limits from the mean range (R-bar).
 * @property a3 factor for computing x-bar chart control limits from the mean standard deviation (S-bar).
 * @property d3 factor for computing the lower control limit of the R chart. Zero for small subgroup sizes.
 * @property d4 factor for computing the upper control limit of the R chart.
 * @property b3 factor for computing the lower control limit of the S chart. Zero for small subgroup sizes.
 * @property b4 factor for computing the upper control limit of the S chart.
 * @property c4 bias-correction factor that relates the expected sample standard deviation to the
 * population standard deviation for normally distributed data.
 * @see spcConstants
 * @see xBarRChart
 * @see xBarSChart
 */
public data class SpcConstants(
    val a2: Double,
    val a3: Double,
    val d3: Double,
    val d4: Double,
    val b3: Double,
    val b4: Double,
    val c4: Double,
) {
    internal companion object {
        // Standard SPC constants for subgroup sizes n = 2..25.
        // Source: Montgomery "Introduction to Statistical Quality Control" (7th ed.), Appendix VI;
        //         NIST/SEMATECH e-Handbook of Statistical Methods, Table 6.3.2.
        //               A₂      A₃      D₃      D₄      B₃      B₄      c₄
        val TABLE: Array<SpcConstants> = arrayOf(
            SpcConstants(1.880, 2.659, 0.000, 3.267, 0.000, 3.267, 0.7979), // n=2
            SpcConstants(1.023, 1.954, 0.000, 2.574, 0.000, 2.568, 0.8862), // n=3
            SpcConstants(0.729, 1.628, 0.000, 2.282, 0.000, 2.266, 0.9213), // n=4
            SpcConstants(0.577, 1.427, 0.000, 2.114, 0.000, 2.089, 0.9400), // n=5
            SpcConstants(0.483, 1.287, 0.000, 2.004, 0.030, 1.970, 0.9515), // n=6
            SpcConstants(0.419, 1.182, 0.076, 1.924, 0.118, 1.882, 0.9594), // n=7
            SpcConstants(0.373, 1.099, 0.136, 1.864, 0.185, 1.815, 0.9650), // n=8
            SpcConstants(0.337, 1.032, 0.184, 1.816, 0.239, 1.761, 0.9693), // n=9
            SpcConstants(0.308, 0.975, 0.223, 1.777, 0.284, 1.716, 0.9727), // n=10
            SpcConstants(0.285, 0.927, 0.256, 1.744, 0.321, 1.679, 0.9754), // n=11
            SpcConstants(0.266, 0.886, 0.283, 1.717, 0.354, 1.646, 0.9776), // n=12
            SpcConstants(0.249, 0.850, 0.307, 1.693, 0.382, 1.618, 0.9794), // n=13
            SpcConstants(0.235, 0.817, 0.328, 1.672, 0.406, 1.594, 0.9810), // n=14
            SpcConstants(0.223, 0.789, 0.347, 1.653, 0.428, 1.572, 0.9823), // n=15
            SpcConstants(0.212, 0.763, 0.363, 1.637, 0.448, 1.552, 0.9835), // n=16
            SpcConstants(0.203, 0.739, 0.378, 1.622, 0.466, 1.534, 0.9845), // n=17
            SpcConstants(0.194, 0.718, 0.391, 1.608, 0.482, 1.518, 0.9854), // n=18
            SpcConstants(0.187, 0.698, 0.403, 1.597, 0.497, 1.503, 0.9862), // n=19
            SpcConstants(0.180, 0.680, 0.415, 1.585, 0.510, 1.490, 0.9869), // n=20
            SpcConstants(0.173, 0.663, 0.425, 1.575, 0.523, 1.477, 0.9876), // n=21
            SpcConstants(0.167, 0.647, 0.434, 1.566, 0.534, 1.466, 0.9882), // n=22
            SpcConstants(0.162, 0.633, 0.443, 1.557, 0.545, 1.455, 0.9887), // n=23
            SpcConstants(0.157, 0.619, 0.451, 1.548, 0.555, 1.445, 0.9892), // n=24
            SpcConstants(0.153, 0.606, 0.459, 1.541, 0.565, 1.435, 0.9896), // n=25
        )
    }
}

/**
 * Returns the standard SPC constants for the given [subgroupSize].
 *
 * The constants are tabulated for subgroup sizes 2 through 25, covering the range used
 * in practice for Shewhart control charts. They are sourced from Montgomery's
 * "Introduction to Statistical Quality Control" (7th ed.), Appendix VI.
 *
 * ### Example:
 * ```kotlin
 * val c = spcConstants(4)
 * c.a2  // 0.729
 * c.d4  // 2.282
 * c.b4  // 2.266
 * ```
 *
 * @param subgroupSize the number of observations in each subgroup (must be in 2..25).
 * @return the [SpcConstants] for that subgroup size.
 * @see SpcConstants
 */
public fun spcConstants(subgroupSize: Int): SpcConstants {
    if (subgroupSize !in 2..25) throw InvalidParameterException(
        "Subgroup size must be in 2..25, got $subgroupSize"
    )
    return SpcConstants.TABLE[subgroupSize - 2]
}

// ── Control chart result types ─────────────────────────────────────────────────

/**
 * Control limits for a single Shewhart control chart.
 *
 * Contains the center line and the upper and lower control limits that define the
 * expected range of variation for a stable (in-control) process. Points falling outside
 * [ucl] or [lcl] signal a potential out-of-control condition.
 *
 * @property centerLine the process average used as the chart's center line.
 * @property ucl the upper control limit (typically center line plus three sigma).
 * @property lcl the lower control limit (typically center line minus three sigma).
 * @see XBarRChartResult
 * @see XBarSChartResult
 */
public data class ControlChartLimits(
    val centerLine: Double,
    val ucl: Double,
    val lcl: Double,
)

/**
 * Results of an x-bar and R (range) control chart analysis.
 *
 * The top-level properties ([centerLine], [ucl], [lcl]) describe the x-bar chart,
 * which monitors the process mean. The nested [rChart] describes the R chart, which
 * monitors the within-subgroup variability using the range.
 *
 * @property centerLine the grand mean (average of subgroup means) — center line of the x-bar chart.
 * @property ucl the upper control limit of the x-bar chart, computed as grand mean plus A₂ times R-bar.
 * @property lcl the lower control limit of the x-bar chart, computed as grand mean minus A₂ times R-bar.
 * @property rChart the control limits for the accompanying R chart.
 * @see xBarRChart
 */
public data class XBarRChartResult(
    val centerLine: Double,
    val ucl: Double,
    val lcl: Double,
    val rChart: ControlChartLimits,
) {
    /** The x-bar chart limits as a [ControlChartLimits] instance for uniform handling. */
    public val xBarChart: ControlChartLimits get() = ControlChartLimits(centerLine, ucl, lcl)
}

/**
 * Results of an x-bar and S (standard deviation) control chart analysis.
 *
 * The top-level properties ([centerLine], [ucl], [lcl]) describe the x-bar chart,
 * which monitors the process mean. The nested [sChart] describes the S chart, which
 * monitors the within-subgroup variability using the sample standard deviation.
 *
 * The S chart is generally preferred over the R chart for subgroup sizes above 10, because
 * the standard deviation uses all observations while the range uses only the extremes.
 *
 * @property centerLine the grand mean (average of subgroup means) — center line of the x-bar chart.
 * @property ucl the upper control limit of the x-bar chart, computed as grand mean plus A₃ times S-bar.
 * @property lcl the lower control limit of the x-bar chart, computed as grand mean minus A₃ times S-bar.
 * @property sChart the control limits for the accompanying S chart.
 * @see xBarSChart
 */
public data class XBarSChartResult(
    val centerLine: Double,
    val ucl: Double,
    val lcl: Double,
    val sChart: ControlChartLimits,
) {
    /** The x-bar chart limits as a [ControlChartLimits] instance for uniform handling. */
    public val xBarChart: ControlChartLimits get() = ControlChartLimits(centerLine, ucl, lcl)
}

// ── Chart computation ──────────────────────────────────────────────────────────

/**
 * Computes x-bar and R control chart limits for the given [subgroups].
 *
 * The x-bar chart tracks the process mean over time, while the R chart tracks the
 * within-subgroup variability using the range (max minus min) of each subgroup.
 * Control limits are set at three sigma from the center line using standard SPC
 * constants (A₂, D₃, D₄).
 *
 * All subgroups must have the same size (between 2 and 25). For subgroup sizes above
 * 10, consider using [xBarSChart] instead — the standard deviation is a more efficient
 * estimator of variability than the range for larger samples.
 *
 * NaN values in the data propagate through the computation (IEEE 754 semantics).
 *
 * ### Example:
 * ```kotlin
 * val subgroups = listOf(
 *     doubleArrayOf(10.1, 10.3, 9.8),
 *     doubleArrayOf(10.0, 10.2, 9.9),
 *     doubleArrayOf(10.1, 10.0, 10.3),
 * )
 * val chart = xBarRChart(subgroups)
 * chart.centerLine  // grand mean (x-double-bar)
 * chart.ucl         // upper control limit for x-bar chart
 * chart.lcl         // lower control limit for x-bar chart
 * chart.rChart.ucl  // upper control limit for R chart
 * ```
 *
 * @param subgroups a list of equal-sized subgroups, each containing at least 2 observations.
 * At least 2 subgroups are required. Subgroup size must be at most 25.
 * @return an [XBarRChartResult] containing x-bar and R chart limits.
 * @see xBarSChart
 * @see spcConstants
 */
public fun xBarRChart(subgroups: List<DoubleArray>): XBarRChartResult {
    validateSubgroups(subgroups)
    val n = subgroups[0].size
    val constants = spcConstants(n)

    val subgroupMeans = DoubleArray(subgroups.size) { subgroups[it].mean() }
    val xBarBar = subgroupMeans.mean()

    val subgroupRanges = DoubleArray(subgroups.size) { subgroups[it].range() }
    val rBar = subgroupRanges.mean()

    return XBarRChartResult(
        centerLine = xBarBar,
        ucl = xBarBar + constants.a2 * rBar,
        lcl = xBarBar - constants.a2 * rBar,
        rChart = ControlChartLimits(
            centerLine = rBar,
            ucl = constants.d4 * rBar,
            lcl = constants.d3 * rBar,
        ),
    )
}

/**
 * Computes x-bar and S control chart limits for the given [subgroups].
 *
 * The x-bar chart tracks the process mean over time, while the S chart tracks the
 * within-subgroup variability using the sample standard deviation. Control limits are
 * set at three sigma from the center line using standard SPC constants (A₃, B₃, B₄).
 *
 * The S chart is generally preferred over the R chart ([xBarRChart]) for subgroup sizes
 * above 10, because the standard deviation uses all observations while the range uses
 * only the two extremes.
 *
 * All subgroups must have the same size (between 2 and 25).
 *
 * NaN values in the data propagate through the computation (IEEE 754 semantics).
 *
 * ### Example:
 * ```kotlin
 * val subgroups = listOf(
 *     doubleArrayOf(10.1, 10.3, 9.8),
 *     doubleArrayOf(10.0, 10.2, 9.9),
 *     doubleArrayOf(10.1, 10.0, 10.3),
 * )
 * val chart = xBarSChart(subgroups)
 * chart.centerLine  // grand mean (x-double-bar)
 * chart.ucl         // upper control limit for x-bar chart
 * chart.sChart.ucl  // upper control limit for S chart
 * ```
 *
 * @param subgroups a list of equal-sized subgroups, each containing at least 2 observations.
 * At least 2 subgroups are required. Subgroup size must be at most 25.
 * @return an [XBarSChartResult] containing x-bar and S chart limits.
 * @see xBarRChart
 * @see spcConstants
 */
public fun xBarSChart(subgroups: List<DoubleArray>): XBarSChartResult {
    validateSubgroups(subgroups)
    val n = subgroups[0].size
    val constants = spcConstants(n)

    val subgroupMeans = DoubleArray(subgroups.size) { subgroups[it].mean() }
    val xBarBar = subgroupMeans.mean()

    val subgroupStdDevs = DoubleArray(subgroups.size) { subgroups[it].standardDeviation() }
    val sBar = subgroupStdDevs.mean()

    return XBarSChartResult(
        centerLine = xBarBar,
        ucl = xBarBar + constants.a3 * sBar,
        lcl = xBarBar - constants.a3 * sBar,
        sChart = ControlChartLimits(
            centerLine = sBar,
            ucl = constants.b4 * sBar,
            lcl = constants.b3 * sBar,
        ),
    )
}

// ── Validation ─────────────────────────────────────────────────────────────────

private fun validateSubgroups(subgroups: List<DoubleArray>) {
    if (subgroups.size < 2) throw InsufficientDataException(
        "Control chart requires at least 2 subgroups, got ${subgroups.size}"
    )
    val n = subgroups[0].size
    if (n < 2) throw InsufficientDataException(
        "Subgroup size must be at least 2, got $n"
    )
    if (n > 25) throw InvalidParameterException(
        "Subgroup size must be at most 25 (SPC constants are tabulated for n=2..25), got $n"
    )
    for ((i, sg) in subgroups.withIndex()) {
        if (sg.size != n) throw InvalidParameterException(
            "All subgroups must have equal size: subgroup 0 has $n elements, but subgroup $i has ${sg.size}"
        )
    }
}
