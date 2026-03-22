package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.distributions.ChiSquaredDistribution

/**
 * Performs a chi-squared goodness-of-fit test.
 *
 * The null hypothesis is that the observed frequency counts follow the expected distribution.
 * This test compares how well observed category counts match the expected counts by computing
 * the sum of squared differences between observed and expected, each divided by the expected count.
 *
 * ### Example:
 * ```kotlin
 * // Test if a die is fair (60 rolls, expect 10 per face)
 * val observed = intArrayOf(8, 12, 11, 9, 10, 10)
 * val result = chiSquaredTest(observed)
 * result.statistic       // chi-squared statistic
 * result.pValue          // p-value
 * result.degreesOfFreedom // 5.0 (six categories minus one)
 * result.isSignificant()  // false (data is consistent with a fair die)
 * ```
 *
 * @param observed the observed frequency counts for each category. Must have at least 2 categories.
 * @param expected the expected frequency counts for each category. If `null`, assumes a uniform
 * distribution where each category has the same expected count (total / number of categories).
 * Defaults to `null`.
 * @return a [TestResult] containing the chi-squared statistic, p-value, and degrees of freedom
 * (number of categories minus one).
 */
public fun chiSquaredTest(
    observed: IntArray,
    expected: DoubleArray? = null
): TestResult {
    if (observed.size < 2) throw InsufficientDataException("Need at least 2 categories")

    val n = observed.size
    val exp = expected ?: run {
        val total = observed.sum().toDouble()
        DoubleArray(n) { total / n }
    }
    if (observed.size != exp.size) throw InvalidParameterException("Observed and expected must have the same size")
    if (!exp.all { it > 0.0 }) throw InvalidParameterException("All expected values must be positive")

    var chi2 = 0.0
    for (i in observed.indices) {
        val diff = observed[i].toDouble() - exp[i]
        chi2 += diff * diff / exp[i]
    }

    val df = (n - 1).toDouble()
    val dist = ChiSquaredDistribution(df)
    val pValue = dist.sf(chi2)

    return TestResult(
        testName = "Chi-Squared Goodness-of-Fit Test",
        statistic = chi2,
        pValue = pValue.coerceIn(0.0, 1.0),
        degreesOfFreedom = df
    )
}

/**
 * Performs a chi-squared test of independence for a contingency table.
 *
 * The null hypothesis is that the row and column variables are independent — that is,
 * knowing the row category does not help predict the column category. The test compares
 * the observed cell counts to the counts expected under independence.
 *
 * ### Example:
 * ```kotlin
 * // 2x2 contingency table: treatment vs outcome
 * val table = arrayOf(
 *     intArrayOf(10, 30),
 *     intArrayOf(20, 40)
 * )
 * val result = chiSquaredIndependenceTest(table)
 * result.statistic        // chi-squared statistic
 * result.pValue           // p-value
 * result.degreesOfFreedom // 1.0 ((2-1) * (2-1))
 * ```
 *
 * @param contingencyTable a matrix of observed frequency counts with at least 2 rows and 2 columns.
 * All rows must have the same number of columns.
 * @return a [TestResult] containing the chi-squared statistic, p-value, and degrees of freedom
 * ((rows - 1) * (columns - 1)).
 */
public fun chiSquaredIndependenceTest(
    contingencyTable: Array<IntArray>
): TestResult {
    val rows = contingencyTable.size
    if (rows < 2) throw InsufficientDataException("Table must have at least 2 rows")
    val cols = contingencyTable[0].size
    if (cols < 2) throw InsufficientDataException("Table must have at least 2 columns")
    if (!contingencyTable.all { it.size == cols }) throw InvalidParameterException("All rows must have the same number of columns")

    val rowTotals = IntArray(rows) { r -> contingencyTable[r].sum() }
    val colTotals = IntArray(cols) { c -> contingencyTable.sumOf { it[c] } }
    if (rowTotals.any { it == 0 }) throw InvalidParameterException("All row totals must be positive")
    if (colTotals.any { it == 0 }) throw InvalidParameterException("All column totals must be positive")
    val total = rowTotals.sum().toDouble()

    var chi2 = 0.0
    for (r in 0 until rows) {
        for (c in 0 until cols) {
            val expected = rowTotals[r].toDouble() * colTotals[c] / total
            val diff = contingencyTable[r][c].toDouble() - expected
            chi2 += diff * diff / expected
        }
    }

    val df = ((rows - 1) * (cols - 1)).toDouble()
    val dist = ChiSquaredDistribution(df)
    val pValue = dist.sf(chi2)

    return TestResult(
        testName = "Chi-Squared Test of Independence",
        statistic = chi2,
        pValue = pValue.coerceIn(0.0, 1.0),
        degreesOfFreedom = df
    )
}
