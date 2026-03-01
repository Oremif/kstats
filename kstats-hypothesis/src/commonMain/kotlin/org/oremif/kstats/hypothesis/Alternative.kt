package org.oremif.kstats.hypothesis

/**
 * Specifies the direction of the alternative hypothesis in a statistical test.
 *
 * The alternative hypothesis is the claim being tested against the null hypothesis.
 * The choice of alternative affects how the p-value is computed from the test statistic.
 */
public enum class Alternative {
    /**
     * Tests whether the parameter differs from the null value in either direction.
     *
     * This is the most common choice and is appropriate when you have no prior expectation
     * about which direction the difference lies.
     */
    TWO_SIDED,

    /**
     * Tests whether the parameter is less than the null value.
     *
     * Use this when you expect the effect to go in one specific direction (lower).
     */
    LESS,

    /**
     * Tests whether the parameter is greater than the null value.
     *
     * Use this when you expect the effect to go in one specific direction (higher).
     */
    GREATER
}
