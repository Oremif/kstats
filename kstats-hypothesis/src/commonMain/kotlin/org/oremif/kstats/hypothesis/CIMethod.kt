package org.oremif.kstats.hypothesis

/**
 * Selects the method used to compute the confidence interval for a proportion.
 *
 * Different methods offer different trade-offs between coverage accuracy and interval width.
 * The Clopper-Pearson interval is exact but conservative (wider than necessary), while the
 * Wilson and Agresti-Coull intervals are approximate but tend to have coverage closer to the
 * nominal confidence level.
 *
 * @see binomialTest
 */
public enum class CIMethod {
    /**
     * Clopper-Pearson exact interval based on the Beta distribution.
     *
     * Guarantees that the true coverage probability is at least the nominal confidence level
     * for any sample size and true proportion. This makes it conservative — the interval is
     * typically wider than necessary. Recommended when exact coverage guarantees are required.
     */
    CLOPPER_PEARSON,

    /**
     * Wilson score interval based on inverting the normal approximation test.
     *
     * Produces narrower intervals than Clopper-Pearson with coverage closer to the nominal
     * level, especially for moderate sample sizes. The interval is always contained in
     * the range from zero to one. Recommended as the default choice for A/B testing and general use.
     */
    WILSON,

    /**
     * Agresti-Coull adjusted Wald interval.
     *
     * Adds a few "pseudo-observations" to the data before applying the standard Wald formula,
     * which corrects the poor coverage of the plain Wald interval for small samples or
     * extreme proportions. Simpler to compute than Wilson and gives very similar results
     * for moderate to large sample sizes. The interval bounds are clamped to the range
     * from zero to one.
     */
    AGRESTI_COULL
}
