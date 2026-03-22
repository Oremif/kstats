package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.distributions.ChiSquaredDistribution
import kotlin.math.ln

/**
 * Performs a G-test (log-likelihood ratio test) for goodness-of-fit.
 *
 * The G-test is an alternative to the chi-squared test that uses a log-likelihood ratio
 * statistic. It computes twice the sum of each observed count times the log of the ratio
 * of observed to expected. Categories with zero observed counts contribute nothing to the
 * statistic. The G-test is asymptotically equivalent to the chi-squared test but can be
 * more accurate for small samples. Under the null hypothesis, the test statistic follows
 * a chi-squared distribution.
 *
 * ### Example:
 * ```kotlin
 * // Test if a die is fair (60 rolls, expect 10 per face)
 * val observed = intArrayOf(8, 12, 11, 9, 10, 10)
 * val result = gTest(observed)
 * result.statistic        // G statistic
 * result.pValue           // p-value
 * result.degreesOfFreedom // 5.0 (six categories minus one)
 * result.isSignificant()  // false (data is consistent with a fair die)
 * ```
 *
 * @param observed the observed frequency counts for each category. Must have at least 2 categories.
 * @param expected the expected frequency counts for each category. If `null`, assumes a uniform
 * distribution where each category has the same expected count (total / number of categories).
 * Defaults to `null`.
 * @return a [TestResult] containing the G statistic, p-value, and degrees of freedom
 * (number of categories minus one).
 */
public fun gTest(
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

    var g = 0.0
    for (i in observed.indices) {
        val o = observed[i].toDouble()
        if (o > 0.0) {
            g += o * ln(o / exp[i])
        }
    }
    g *= 2.0

    val df = (n - 1).toDouble()
    val dist = ChiSquaredDistribution(df)
    val pValue = dist.sf(g)

    return TestResult(
        testName = "G-Test",
        statistic = g,
        pValue = pValue.coerceIn(0.0, 1.0),
        degreesOfFreedom = df
    )
}

/**
 * Performs a G-test (log-likelihood ratio test) of independence for a contingency table.
 *
 * The null hypothesis is that the row and column variables are independent — that is,
 * knowing the row category does not help predict the column category. This test uses
 * a log-likelihood ratio statistic as an alternative to the chi-squared test of independence.
 *
 * ### Example:
 * ```kotlin
 * // 2x2 contingency table: treatment vs outcome
 * val table = arrayOf(
 *     intArrayOf(10, 30),
 *     intArrayOf(20, 40)
 * )
 * val result = gIndependenceTest(table)
 * result.statistic        // G statistic
 * result.pValue           // p-value
 * result.degreesOfFreedom // 1.0 ((2-1) * (2-1))
 * ```
 *
 * @param contingencyTable a matrix of observed frequency counts with at least 2 rows and 2 columns.
 * All rows must have the same number of columns.
 * @return a [TestResult] containing the G statistic, p-value, and degrees of freedom
 * ((rows - 1) * (columns - 1)).
 */
public fun gIndependenceTest(
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

    var g = 0.0
    for (r in 0 until rows) {
        for (c in 0 until cols) {
            val o = contingencyTable[r][c].toDouble()
            if (o > 0.0) {
                val expected = rowTotals[r].toDouble() * colTotals[c] / total
                g += o * ln(o / expected)
            }
        }
    }
    g *= 2.0

    val df = ((rows - 1) * (cols - 1)).toDouble()
    val dist = ChiSquaredDistribution(df)
    val pValue = dist.sf(g)

    return TestResult(
        testName = "G-Test of Independence",
        statistic = g,
        pValue = pValue.coerceIn(0.0, 1.0),
        degreesOfFreedom = df
    )
}
