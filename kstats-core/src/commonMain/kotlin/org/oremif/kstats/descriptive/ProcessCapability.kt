package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Results of a process capability analysis.
 *
 * Contains four standard SPC (Statistical Process Control) indices that measure how well a
 * process fits within its specification limits. Values above 1.0 indicate a capable process;
 * values above 1.33 are generally considered good.
 *
 * Cp and Cpk use the sample standard deviation (divides by n-1) as the spread estimate,
 * while Pp and Ppk use the population standard deviation (divides by n).
 *
 * @property cp potential capability — compares the tolerance width to 6 times the sample
 * standard deviation. Ignores process centering.
 * @property cpk actual capability — like [cp] but penalizes the process for being off-center
 * relative to the specification limits. Always less than or equal to [cp].
 * @property pp potential performance — same formula as [cp] but uses the population standard
 * deviation instead of the sample standard deviation.
 * @property ppk actual performance — same formula as [cpk] but uses the population standard
 * deviation. Always less than or equal to [pp].
 * @see processCapability
 */
public data class ProcessCapabilityResult(
    val cp: Double,
    val cpk: Double,
    val pp: Double,
    val ppk: Double,
)

/**
 * Computes process capability indices (Cp, Cpk, Pp, Ppk) for the values in this array.
 *
 * These indices measure whether a manufacturing or business process produces output that fits
 * within specification limits. The lower specification limit [lsl] and upper specification
 * limit [usl] define the acceptable range. The indices compare this tolerance width to the
 * observed spread of the data:
 *
 * - **Cp** = tolerance / (6 × sample σ) — potential capability, ignoring centering.
 * - **Cpk** = min of the distance from the mean to each spec limit, divided by 3 × sample σ.
 * - **Pp** and **Ppk** — same formulas using population σ (divides by n instead of n-1).
 *
 * Uses Welford's numerically stable single-pass algorithm for mean and variance.
 *
 * NaN values in the data propagate through the computation (IEEE 754 semantics).
 *
 * ### Example:
 * ```kotlin
 * val data = doubleArrayOf(10.1, 10.3, 9.8, 10.0, 10.2, 9.9, 10.1)
 * val result = data.processCapability(lsl = 9.0, usl = 11.0)
 * result.cp   // 1.94...
 * result.cpk  // 1.81...
 * ```
 *
 * @param lsl the lower specification limit. Must be less than [usl].
 * @param usl the upper specification limit. Must be greater than [lsl].
 * @return a [ProcessCapabilityResult] containing Cp, Cpk, Pp, and Ppk.
 * @throws DegenerateDataException if all values are identical (standard deviation is zero).
 */
public fun DoubleArray.processCapability(lsl: Double, usl: Double): ProcessCapabilityResult {
    if (size < 2) throw InsufficientDataException("Process capability requires at least 2 elements, got $size")
    if (lsl >= usl) throw InvalidParameterException("lsl must be less than usl, got lsl=$lsl, usl=$usl")

    return welford { mean, m2 ->
        val sampleSigma = sqrt(m2 / (size - 1))
        val populationSigma = sqrt(m2 / size)

        if (sampleSigma == 0.0) throw DegenerateDataException(
            "Process capability is undefined when standard deviation is zero"
        )

        val tolerance = usl - lsl

        // Cp/Cpk use sample (within) sigma
        val cp = tolerance / (6.0 * sampleSigma)
        val cpk = min((usl - mean) / (3.0 * sampleSigma), (mean - lsl) / (3.0 * sampleSigma))

        // Pp/Ppk use population (overall) sigma
        val pp = tolerance / (6.0 * populationSigma)
        val ppk = min((usl - mean) / (3.0 * populationSigma), (mean - lsl) / (3.0 * populationSigma))

        ProcessCapabilityResult(cp = cp, cpk = cpk, pp = pp, ppk = ppk)
    }
}

/**
 * Computes process capability indices (Cp, Cpk, Pp, Ppk) for the values in this iterable.
 *
 * These indices measure whether a process produces output that fits within specification
 * limits defined by [lsl] (lower) and [usl] (upper). Values above 1.0 indicate a capable
 * process; values above 1.33 are generally considered good.
 *
 * NaN values in the data propagate through the computation (IEEE 754 semantics).
 *
 * ### Example:
 * ```kotlin
 * val result = listOf(10.1, 10.3, 9.8, 10.0, 10.2).processCapability(lsl = 9.0, usl = 11.0)
 * result.cp  // 1.68...
 * ```
 *
 * @param lsl the lower specification limit. Must be less than [usl].
 * @param usl the upper specification limit. Must be greater than [lsl].
 * @return a [ProcessCapabilityResult] containing Cp, Cpk, Pp, and Ppk.
 * @throws DegenerateDataException if all values are identical (standard deviation is zero).
 */
public fun Iterable<Double>.processCapability(lsl: Double, usl: Double): ProcessCapabilityResult =
    toList().toDoubleArray().processCapability(lsl, usl)

/**
 * Computes process capability indices (Cp, Cpk, Pp, Ppk) for the values in this sequence.
 *
 * These indices measure whether a process produces output that fits within specification
 * limits defined by [lsl] (lower) and [usl] (upper). Values above 1.0 indicate a capable
 * process; values above 1.33 are generally considered good.
 *
 * NaN values in the data propagate through the computation (IEEE 754 semantics).
 *
 * ### Example:
 * ```kotlin
 * val result = sequenceOf(10.1, 10.3, 9.8, 10.0, 10.2).processCapability(lsl = 9.0, usl = 11.0)
 * result.cpk // 1.57...
 * ```
 *
 * @param lsl the lower specification limit. Must be less than [usl].
 * @param usl the upper specification limit. Must be greater than [lsl].
 * @return a [ProcessCapabilityResult] containing Cp, Cpk, Pp, and Ppk.
 * @throws DegenerateDataException if all values are identical (standard deviation is zero).
 */
public fun Sequence<Double>.processCapability(lsl: Double, usl: Double): ProcessCapabilityResult =
    toList().toDoubleArray().processCapability(lsl, usl)
