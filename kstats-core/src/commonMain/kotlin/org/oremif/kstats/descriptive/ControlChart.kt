package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.sqrt

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

// ── CUSUM chart ────────────────────────────────────────────────────────────────

/**
 * Results of a two-sided tabular CUSUM control chart analysis.
 *
 * Holds the per-observation upper (C⁺) and lower (C⁻) cumulative sums produced by the
 * tabular CUSUM algorithm, together with the index of the first observation at which an
 * out-of-control signal fired. The upper sum C⁺ accumulates positive deviations above
 * the target; the lower sum C⁻ accumulates negative deviations below the target. An
 * alarm is signaled as soon as either sum exceeds the decision interval H.
 *
 * @property sPlus the sequence of upper cumulative sums C⁺ᵢ, one entry per observation.
 * Each value is non-negative; a rising C⁺ indicates a sustained positive shift from target.
 * @property sMinus the sequence of lower cumulative sums C⁻ᵢ, one entry per observation.
 * Each value is non-negative; a rising C⁻ indicates a sustained negative shift from target.
 * @property alarmIndex the zero-based index of the first observation at which C⁺ > H or
 * C⁻ > H, or `-1` if no alarm was triggered anywhere in the series.
 * @see cusum
 */
public data class CusumResult(
    val sPlus: DoubleArray,
    val sMinus: DoubleArray,
    val alarmIndex: Int,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is CusumResult) return false
        return alarmIndex == other.alarmIndex &&
            sPlus.contentEquals(other.sPlus) &&
            sMinus.contentEquals(other.sMinus)
    }

    override fun hashCode(): Int {
        var result = sPlus.contentHashCode()
        result = 31 * result + sMinus.contentHashCode()
        result = 31 * result + alarmIndex
        return result
    }

    override fun toString(): String =
        "CusumResult(sPlus=${sPlus.contentToString()}, sMinus=${sMinus.contentToString()}, " +
            "alarmIndex=$alarmIndex)"
}

/**
 * Computes a two-sided tabular CUSUM control chart for the given [observations].
 *
 * The cumulative sum (CUSUM) chart is designed to detect small, sustained shifts in the
 * process mean more quickly than a Shewhart chart such as [xBarRChart] or [xBarSChart].
 * Where a Shewhart chart reacts only to each single observation, CUSUM accumulates
 * deviations from the reference mean over time, so even a drift of roughly 0.5σ–1σ is
 * detected within a few observations.
 *
 * This implementation uses the standard tabular two-sided formulation:
 *
 * ```
 *   C⁺ᵢ = max(0, C⁺ᵢ₋₁ + (xᵢ − μ₀ − K))
 *   C⁻ᵢ = max(0, C⁻ᵢ₋₁ + (μ₀ − K − xᵢ))
 * ```
 *
 * with C⁺₀ = C⁻₀ = 0. An out-of-control signal is raised at the first index where
 * C⁺ᵢ > H or C⁻ᵢ > H.
 *
 * The allowance [k] (often denoted K) is typically chosen as half the magnitude of the
 * shift (in σ units) that one wants to detect — a common default is K ≈ 0.5σ, which
 * targets a 1σ shift. The decision interval [h] (often denoted H) is typically set to
 * 4σ or 5σ, giving an in-control average run length comparable to a 3σ Shewhart chart
 * while reacting much faster to small shifts.
 *
 * NaN values in the data propagate through the computation (IEEE 754 semantics).
 *
 * ### Example:
 * ```kotlin
 * // Process with target 10.0, drifting upward after observation 4.
 * val observations = doubleArrayOf(10.0, 10.1, 9.9, 10.0, 10.2, 10.5, 10.7, 10.9, 11.1)
 * val result = cusum(observations, target = 10.0, k = 0.25, h = 4.0)
 * result.sPlus       // non-negative running upper CUSUM values
 * result.sMinus      // non-negative running lower CUSUM values
 * result.alarmIndex  // zero-based index where C⁺ > H first, or -1 if never
 * ```
 *
 * References: Page (1954), "Continuous Inspection Schemes"; Montgomery, "Introduction
 * to Statistical Quality Control" (7th ed.), §9.1.1.
 *
 * @param observations the sequence of individual measurements to monitor. Must contain
 * at least 1 element.
 * @param target the reference mean μ₀ to monitor against. Any finite value is allowed.
 * @param k the allowance (slack) value K, half the magnitude of the shift to detect.
 * Must be non-negative. Typically chosen as ≈ 0.5σ of the in-control process.
 * @param h the decision interval H above which the CUSUM signals an out-of-control
 * condition. Must be strictly positive. Typically chosen as 4σ or 5σ.
 * @return a [CusumResult] with the full C⁺ and C⁻ series and the index of the first alarm.
 * @throws InsufficientDataException if [observations] is empty.
 * @throws InvalidParameterException if [k] is negative or [h] is non-positive.
 * @see CusumResult
 * @see xBarRChart
 * @see xBarSChart
 */
public fun cusum(
    observations: DoubleArray,
    target: Double,
    k: Double,
    h: Double,
): CusumResult {
    if (observations.isEmpty()) throw InsufficientDataException(
        "CUSUM requires at least 1 observation, got 0"
    )
    if (k < 0.0) throw InvalidParameterException("allowance k must be non-negative, got $k")
    if (h <= 0.0) throw InvalidParameterException("decision interval h must be positive, got $h")

    // Tabular two-sided CUSUM (Page 1954; Montgomery "Introduction to Statistical Quality
    // Control" 7th ed., §9.1.1):
    //   C⁺ᵢ = max(0, C⁺ᵢ₋₁ + (xᵢ − μ₀ − K))
    //   C⁻ᵢ = max(0, C⁻ᵢ₋₁ + (μ₀ − K − xᵢ))
    // Alarm fires when C⁺ᵢ > H or C⁻ᵢ > H.
    val n = observations.size
    val sPlus = DoubleArray(n)
    val sMinus = DoubleArray(n)
    var prevPlus = 0.0
    var prevMinus = 0.0
    var alarmIndex = -1

    for (i in 0 until n) {
        val x = observations[i]
        val cPlus = maxOf(0.0, prevPlus + (x - target - k))
        val cMinus = maxOf(0.0, prevMinus + (target - k - x))
        sPlus[i] = cPlus
        sMinus[i] = cMinus
        if (alarmIndex == -1 && (cPlus > h || cMinus > h)) {
            alarmIndex = i
        }
        prevPlus = cPlus
        prevMinus = cMinus
    }

    return CusumResult(sPlus = sPlus, sMinus = sMinus, alarmIndex = alarmIndex)
}

/**
 * Computes a two-sided tabular CUSUM control chart for an [Iterable] of observations.
 *
 * Convenience overload that collects [observations] into a `DoubleArray` and delegates
 * to the primary [cusum] function. See [cusum] for the full description of the
 * algorithm, parameter guidance, and references.
 *
 * ### Example:
 * ```kotlin
 * val observations: List<Double> = listOf(10.0, 10.1, 9.9, 10.0, 10.2, 10.5, 10.7, 10.9, 11.1)
 * val result = cusum(observations, target = 10.0, k = 0.25, h = 4.0)
 * result.alarmIndex  // zero-based index of first alarm, or -1 if never
 * ```
 *
 * @param observations the sequence of individual measurements to monitor. Must contain
 * at least 1 element.
 * @param target the reference mean μ₀ to monitor against. Any finite value is allowed.
 * @param k the allowance (slack) value K. Must be non-negative. Typically ≈ 0.5σ.
 * @param h the decision interval H. Must be strictly positive. Typically 4σ or 5σ.
 * @return a [CusumResult] with the full C⁺ and C⁻ series and the index of the first alarm.
 * @throws InsufficientDataException if [observations] is empty.
 * @throws InvalidParameterException if [k] is negative or [h] is non-positive.
 * @see cusum
 * @see CusumResult
 */
public fun cusum(
    observations: Iterable<Double>,
    target: Double,
    k: Double,
    h: Double,
): CusumResult = cusum(observations.toList().toDoubleArray(), target, k, h)

/**
 * Computes a two-sided tabular CUSUM control chart for a [Sequence] of observations.
 *
 * Convenience overload that collects [observations] into a `DoubleArray` and delegates
 * to the primary [cusum] function. See [cusum] for the full description of the
 * algorithm, parameter guidance, and references.
 *
 * ### Example:
 * ```kotlin
 * val observations: Sequence<Double> = sequenceOf(10.0, 10.1, 9.9, 10.0, 10.2, 10.5, 10.7, 10.9, 11.1)
 * val result = cusum(observations, target = 10.0, k = 0.25, h = 4.0)
 * result.alarmIndex  // zero-based index of first alarm, or -1 if never
 * ```
 *
 * @param observations the sequence of individual measurements to monitor. Must contain
 * at least 1 element.
 * @param target the reference mean μ₀ to monitor against. Any finite value is allowed.
 * @param k the allowance (slack) value K. Must be non-negative. Typically ≈ 0.5σ.
 * @param h the decision interval H. Must be strictly positive. Typically 4σ or 5σ.
 * @return a [CusumResult] with the full C⁺ and C⁻ series and the index of the first alarm.
 * @throws InsufficientDataException if [observations] is empty.
 * @throws InvalidParameterException if [k] is negative or [h] is non-positive.
 * @see cusum
 * @see CusumResult
 */
public fun cusum(
    observations: Sequence<Double>,
    target: Double,
    k: Double,
    h: Double,
): CusumResult = cusum(observations.toList().toDoubleArray(), target, k, h)

// ── EWMA chart ─────────────────────────────────────────────────────────────────

/**
 * Results of an EWMA (Exponentially Weighted Moving Average) control chart analysis.
 *
 * Holds the per-observation smoothed statistic Zₜ and the time-varying upper and lower
 * control limits, together with the indices of all observations at which the smoothed
 * statistic fell outside its control limits.
 *
 * @property smoothedValues the sequence of exponentially weighted moving averages Zₜ, one
 * entry per observation.
 * @property ucl the sequence of upper control limits UCLₜ, one entry per observation.
 * The limits widen with t and approach their steady-state value.
 * @property lcl the sequence of lower control limits LCLₜ, one entry per observation.
 * @property outOfControl the zero-based indices of observations whose smoothed statistic
 * Zₜ is strictly above [ucl] or strictly below [lcl]. Empty when the process stays in
 * control.
 * @see ewma
 */
public data class EwmaResult(
    val smoothedValues: DoubleArray,
    val ucl: DoubleArray,
    val lcl: DoubleArray,
    val outOfControl: IntArray,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is EwmaResult) return false
        return smoothedValues.contentEquals(other.smoothedValues) &&
            ucl.contentEquals(other.ucl) &&
            lcl.contentEquals(other.lcl) &&
            outOfControl.contentEquals(other.outOfControl)
    }

    override fun hashCode(): Int {
        var result = smoothedValues.contentHashCode()
        result = 31 * result + ucl.contentHashCode()
        result = 31 * result + lcl.contentHashCode()
        result = 31 * result + outOfControl.contentHashCode()
        return result
    }

    override fun toString(): String =
        "EwmaResult(smoothedValues=${smoothedValues.contentToString()}, " +
            "ucl=${ucl.contentToString()}, lcl=${lcl.contentToString()}, " +
            "outOfControl=${outOfControl.contentToString()})"
}

/**
 * Computes an EWMA (Exponentially Weighted Moving Average) control chart for the given
 * [observations].
 *
 * The EWMA chart, due to Roberts (1959), is designed to detect small, sustained shifts in
 * the process mean more quickly than a Shewhart chart such as [xBarRChart] or [xBarSChart].
 * Instead of reacting to each single observation, it tracks an exponentially weighted
 * moving average that gives recent observations higher weight while retaining some memory
 * of the past, so that a drift of roughly 0.5σ–1σ is detected within a few observations.
 * EWMA is typically competitive with [cusum] for detecting small shifts and more robust
 * for non-normal data.
 *
 * The smoothed statistic is computed recursively:
 *
 * ```
 *   Zₜ = λ·xₜ + (1 − λ)·Zₜ₋₁,   Z₀ = μ₀
 * ```
 *
 * with time-varying control limits:
 *
 * ```
 *   σ_{Zₜ} = σ · √(λ/(2 − λ) · (1 − (1 − λ)^(2t)))
 *   UCLₜ / LCLₜ = μ₀ ± L · σ_{Zₜ}
 * ```
 *
 * The smoothing parameter [lambda] controls the memory of the chart — small values
 * (λ ≈ 0.05–0.25) give high sensitivity to small shifts but slow response to large shifts,
 * while larger values (λ ≈ 0.40) behave more like a Shewhart chart. Common choices are
 * λ = 0.2 with L ≈ 2.7–3.0, which gives an in-control average run length comparable to a
 * 3σ Shewhart chart while reacting much faster to small shifts.
 *
 * NaN values in the data propagate through the computation (IEEE 754 semantics). An
 * observation of NaN yields a NaN smoothed statistic from that point forward and never
 * triggers an out-of-control signal (NaN comparisons are always false).
 *
 * ### Example:
 * ```kotlin
 * val observations = doubleArrayOf(25.0, 24.5, 25.2, 26.1, 25.8, 27.0, 26.5, 28.0)
 * val result = ewma(observations, target = 25.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
 * result.smoothedValues // exponentially weighted moving averages Zₜ
 * result.ucl            // upper control limits at each point
 * result.lcl            // lower control limits at each point
 * result.outOfControl   // indices of points outside control limits
 * ```
 *
 * References: Roberts (1959), "Control Chart Tests Based on Geometric Moving Averages";
 * Montgomery, "Introduction to Statistical Quality Control" (7th ed.), §9.2.
 *
 * @param observations the sequence of individual measurements to monitor. Must contain
 * at least 1 element.
 * @param target the reference mean μ₀ to monitor against, also used as the initial value
 * Z₀ of the smoothed statistic.
 * @param sigma the in-control process standard deviation σ. Must be strictly positive.
 * @param lambda the smoothing parameter λ ∈ (0, 1] controlling the weight of new
 * observations. Typical values are 0.05–0.40; λ = 0.2 is a common default.
 * @param controlLimitWidth the control limit width L, in units of σ_{Zₜ}. Must be strictly
 * positive. Typical values are 2.7–3.0.
 * @return an [EwmaResult] with the full Zₜ series, per-observation control limits, and
 * the indices of any out-of-control points.
 * @throws InsufficientDataException if [observations] is empty.
 * @throws InvalidParameterException if [sigma] is non-positive, [lambda] is outside
 * (0, 1], or [controlLimitWidth] is non-positive.
 * @see EwmaResult
 * @see cusum
 * @see xBarRChart
 * @see xBarSChart
 */
public fun ewma(
    observations: DoubleArray,
    target: Double,
    sigma: Double,
    lambda: Double,
    controlLimitWidth: Double,
): EwmaResult {
    if (observations.isEmpty()) throw InsufficientDataException(
        "EWMA requires at least 1 observation, got 0"
    )
    if (sigma <= 0.0) throw InvalidParameterException("sigma must be positive, got $sigma")
    if (lambda <= 0.0 || lambda > 1.0) throw InvalidParameterException(
        "lambda must be in (0, 1], got $lambda"
    )
    if (controlLimitWidth <= 0.0) throw InvalidParameterException(
        "controlLimitWidth must be positive, got $controlLimitWidth"
    )

    // EWMA chart (Roberts 1959, "Control Chart Tests Based on Geometric Moving Averages";
    // Montgomery "Introduction to Statistical Quality Control" 7th ed., §9.2):
    //   Zₜ = λ·xₜ + (1 − λ)·Zₜ₋₁,   Z₀ = μ₀
    //   σ_{Zₜ} = σ · √(λ/(2 − λ) · (1 − (1 − λ)^(2t)))
    //   UCLₜ / LCLₜ = μ₀ ± L · σ_{Zₜ}
    val n = observations.size
    val smoothed = DoubleArray(n)
    val ucl = DoubleArray(n)
    val lcl = DoubleArray(n)
    val outOfControl = mutableListOf<Int>()

    val oneMinusLambda = 1.0 - lambda
    val steadyStateFactor = lambda / (2.0 - lambda)

    var prev = target
    var oneMinusLambdaPowT = 1.0 // (1 − λ)^0
    for (i in 0 until n) {
        val z = lambda * observations[i] + oneMinusLambda * prev
        smoothed[i] = z
        oneMinusLambdaPowT *= oneMinusLambda // (1 − λ)^(i+1) = (1 − λ)^t
        val r = oneMinusLambdaPowT
        // (1 − r) · (1 + r) instead of 1 − r·r to avoid catastrophic cancellation when
        // r ≈ 1 (small λ with small t).
        val variance = steadyStateFactor * (1.0 - r) * (1.0 + r)
        val sigmaZt = sigma * sqrt(variance)
        val width = controlLimitWidth * sigmaZt
        ucl[i] = target + width
        lcl[i] = target - width
        if (z > ucl[i] || z < lcl[i]) {
            outOfControl.add(i)
        }
        prev = z
    }

    return EwmaResult(
        smoothedValues = smoothed,
        ucl = ucl,
        lcl = lcl,
        outOfControl = outOfControl.toIntArray(),
    )
}

/**
 * Computes an EWMA control chart for an [Iterable] of observations.
 *
 * Convenience overload that collects [observations] into a `DoubleArray` and delegates
 * to the primary [ewma] function. See [ewma] for the full description of the algorithm,
 * parameter guidance, and references.
 *
 * ### Example:
 * ```kotlin
 * val observations: List<Double> = listOf(25.0, 24.5, 25.2, 26.1, 25.8, 27.0, 26.5, 28.0)
 * val result = ewma(observations, target = 25.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
 * result.outOfControl  // indices of points outside control limits
 * ```
 *
 * @param observations the sequence of individual measurements to monitor. Must contain
 * at least 1 element.
 * @param target the reference mean μ₀, also used as the initial value Z₀.
 * @param sigma the in-control process standard deviation σ. Must be strictly positive.
 * @param lambda the smoothing parameter λ ∈ (0, 1].
 * @param controlLimitWidth the control limit width L. Must be strictly positive.
 * @return an [EwmaResult] with the full Zₜ series, control limits, and out-of-control indices.
 * @throws InsufficientDataException if [observations] is empty.
 * @throws InvalidParameterException if [sigma] is non-positive, [lambda] is outside
 * (0, 1], or [controlLimitWidth] is non-positive.
 * @see ewma
 * @see EwmaResult
 */
public fun ewma(
    observations: Iterable<Double>,
    target: Double,
    sigma: Double,
    lambda: Double,
    controlLimitWidth: Double,
): EwmaResult = ewma(observations.toList().toDoubleArray(), target, sigma, lambda, controlLimitWidth)

/**
 * Computes an EWMA control chart for a [Sequence] of observations.
 *
 * Convenience overload that collects [observations] into a `DoubleArray` and delegates
 * to the primary [ewma] function. See [ewma] for the full description of the algorithm,
 * parameter guidance, and references.
 *
 * ### Example:
 * ```kotlin
 * val observations: Sequence<Double> = sequenceOf(25.0, 24.5, 25.2, 26.1, 25.8, 27.0, 26.5, 28.0)
 * val result = ewma(observations, target = 25.0, sigma = 1.0, lambda = 0.2, controlLimitWidth = 3.0)
 * result.outOfControl  // indices of points outside control limits
 * ```
 *
 * @param observations the sequence of individual measurements to monitor. Must contain
 * at least 1 element.
 * @param target the reference mean μ₀, also used as the initial value Z₀.
 * @param sigma the in-control process standard deviation σ. Must be strictly positive.
 * @param lambda the smoothing parameter λ ∈ (0, 1].
 * @param controlLimitWidth the control limit width L. Must be strictly positive.
 * @return an [EwmaResult] with the full Zₜ series, control limits, and out-of-control indices.
 * @throws InsufficientDataException if [observations] is empty.
 * @throws InvalidParameterException if [sigma] is non-positive, [lambda] is outside
 * (0, 1], or [controlLimitWidth] is non-positive.
 * @see ewma
 * @see EwmaResult
 */
public fun ewma(
    observations: Sequence<Double>,
    target: Double,
    sigma: Double,
    lambda: Double,
    controlLimitWidth: Double,
): EwmaResult = ewma(observations.toList().toDoubleArray(), target, sigma, lambda, controlLimitWidth)

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
