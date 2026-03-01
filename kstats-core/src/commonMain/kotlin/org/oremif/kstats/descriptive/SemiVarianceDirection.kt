package org.oremif.kstats.descriptive

/**
 * Selects which side of the threshold to measure in a semi-variance calculation.
 *
 * Semi-variance splits the total variance into downside risk (values below the threshold)
 * and upside potential (values above). This enum controls which side is measured.
 */
public enum class SemiVarianceDirection {
    /**
     * Measure variability of values strictly above the threshold.
     *
     * Only deviations where the value exceeds the threshold contribute to the result.
     * Values at or below the threshold are ignored.
     */
    UPSIDE,

    /**
     * Measure variability of values strictly below the threshold.
     *
     * Only deviations where the value falls below the threshold contribute to the result.
     * Values at or above the threshold are ignored.
     */
    DOWNSIDE
}
