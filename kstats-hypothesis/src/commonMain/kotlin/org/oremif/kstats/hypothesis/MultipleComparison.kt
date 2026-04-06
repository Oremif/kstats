package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException

/**
 * Adjusts p-values using the Bonferroni correction for multiple comparisons.
 *
 * The Bonferroni correction controls the family-wise error rate (FWER) by multiplying
 * each p-value by the total number of tests. This is the simplest and most conservative
 * correction — it guarantees that the probability of any false positive across all tests
 * stays below the significance level, but may miss true effects when many tests are performed.
 *
 * Adjusted p-values are clamped to a maximum of 1.0. NaN p-values pass through unchanged,
 * but they still count towards the total number of tests used as the multiplier.
 *
 * ### Example:
 * ```kotlin
 * val pValues = doubleArrayOf(0.01, 0.04, 0.03, 0.005)
 * bonferroniCorrection(pValues) // [0.04, 0.16, 0.12, 0.02]
 * ```
 *
 * @param pValues the raw p-values to adjust. Each value must be in [0, 1] or NaN.
 * @return a new array of adjusted p-values in the same order as the input.
 * @see holmBonferroniCorrection for a less conservative step-down alternative.
 * @see benjaminiHochbergCorrection for false discovery rate control.
 */
public fun bonferroniCorrection(pValues: DoubleArray): DoubleArray {
    requireNonEmptyPValues(pValues)
    validatePValues(pValues)
    val m = pValues.size
    return DoubleArray(m) { i ->
        val p = pValues[i]
        if (p.isNaN()) Double.NaN else (p * m).coerceAtMost(1.0)
    }
}

/**
 * Adjusts p-values using the Holm-Bonferroni step-down correction for multiple comparisons.
 *
 * The Holm-Bonferroni method controls the family-wise error rate (FWER) like Bonferroni but
 * is uniformly more powerful. It sorts p-values from smallest to largest and multiplies each
 * by a decreasing factor (number of remaining tests), enforcing monotonicity so that a p-value
 * at a higher rank is never smaller than one at a lower rank.
 *
 * Adjusted p-values are clamped to a maximum of 1.0. NaN p-values pass through unchanged
 * and are placed last during ranking, but they still count towards the total number of tests.
 *
 * ### Example:
 * ```kotlin
 * val pValues = doubleArrayOf(0.01, 0.04, 0.03, 0.005)
 * holmBonferroniCorrection(pValues) // [0.03, 0.06, 0.06, 0.02]
 * ```
 *
 * @param pValues the raw p-values to adjust. Each value must be in [0, 1] or NaN.
 * @return a new array of adjusted p-values in the same order as the input.
 * @see bonferroniCorrection for a simpler but more conservative alternative.
 * @see benjaminiHochbergCorrection for false discovery rate control.
 */
public fun holmBonferroniCorrection(pValues: DoubleArray): DoubleArray {
    requireNonEmptyPValues(pValues)
    validatePValues(pValues)
    val m = pValues.size
    val result = DoubleArray(m)

    // Indices sorted by p-value ascending; NaN goes last
    val sortedIndices = (0 until m).sortedWith(
        compareBy { if (pValues[it].isNaN()) Double.POSITIVE_INFINITY else pValues[it] }
    )

    // Step-down: p[rank] * (m - rank), enforce monotonicity via cumulative max
    var cumulativeMax = 0.0
    for (rank in 0 until m) {
        val origIdx = sortedIndices[rank]
        val p = pValues[origIdx]
        if (p.isNaN()) {
            result[origIdx] = Double.NaN
        } else {
            val adjusted = (p * (m - rank)).coerceAtMost(1.0)
            cumulativeMax = maxOf(cumulativeMax, adjusted)
            result[origIdx] = cumulativeMax
        }
    }
    return result
}

/**
 * Adjusts p-values using the Benjamini-Hochberg procedure for false discovery rate control.
 *
 * Unlike Bonferroni and Holm, which control the family-wise error rate (probability of any
 * false positive), Benjamini-Hochberg controls the false discovery rate (FDR) — the expected
 * proportion of false positives among all rejected hypotheses. This makes it substantially
 * more powerful when many tests are performed, at the cost of allowing a controlled fraction
 * of false discoveries.
 *
 * The procedure sorts p-values from largest to smallest and multiplies each by the ratio of
 * total tests to rank, enforcing monotonicity so that a p-value at a lower rank is never
 * larger than one at a higher rank.
 *
 * Adjusted p-values are clamped to a maximum of 1.0. NaN p-values pass through unchanged,
 * but they still count towards the total number of tests.
 *
 * ### Example:
 * ```kotlin
 * val pValues = doubleArrayOf(0.01, 0.04, 0.03, 0.005)
 * benjaminiHochbergCorrection(pValues) // [0.02, 0.04, 0.04, 0.02]
 * ```
 *
 * @param pValues the raw p-values to adjust. Each value must be in [0, 1] or NaN.
 * @return a new array of adjusted p-values in the same order as the input.
 * @see bonferroniCorrection for family-wise error rate control.
 * @see holmBonferroniCorrection for a step-down FWER method.
 */
public fun benjaminiHochbergCorrection(pValues: DoubleArray): DoubleArray {
    requireNonEmptyPValues(pValues)
    validatePValues(pValues)
    val m = pValues.size
    val result = DoubleArray(m)

    // Indices sorted by p-value descending; NaN goes first (will be skipped)
    val sortedIndices = (0 until m).sortedWith(
        compareByDescending { if (pValues[it].isNaN()) Double.POSITIVE_INFINITY else pValues[it] }
    )

    // Step-up from largest: p[rank] * m / rank, enforce monotonicity via cumulative min
    var cumulativeMin = 1.0
    for (i in 0 until m) {
        val origIdx = sortedIndices[i]
        val p = pValues[origIdx]
        if (p.isNaN()) {
            result[origIdx] = Double.NaN
        } else {
            // rank in ascending order: largest p-value has rank m
            val rank = m - i
            val adjusted = (p * m.toDouble() / rank).coerceAtMost(1.0)
            cumulativeMin = minOf(cumulativeMin, adjusted)
            result[origIdx] = cumulativeMin
        }
    }
    return result
}

private fun requireNonEmptyPValues(pValues: DoubleArray) {
    if (pValues.isEmpty()) throw InsufficientDataException("pValues must not be empty")
}

private fun validatePValues(pValues: DoubleArray) {
    for (i in pValues.indices) {
        val p = pValues[i]
        if (p.isNaN()) continue
        if (p < 0.0 || p > 1.0) {
            throw InvalidParameterException("p-values must be in [0, 1], got $p at index $i")
        }
    }
}
