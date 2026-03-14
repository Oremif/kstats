package org.oremif.kstats.descriptive

/**
 * Selects between sample and population formulas for variance and standard deviation.
 *
 * The choice affects the denominator: sample formulas divide by n-1 to correct for
 * estimation bias, while population formulas divide by n.
 */
public enum class PopulationKind {
    /**
     * Use the population formula, which divides by n.
     *
     * This is appropriate when the data represents the entire population of interest.
     */
    POPULATION,

    /**
     * Use the sample formula, which divides by n-1 (Bessel's correction).
     *
     * This is appropriate when the data is a sample drawn from a larger population
     * and you want an unbiased estimate of the population parameter.
     */
    SAMPLE
}
