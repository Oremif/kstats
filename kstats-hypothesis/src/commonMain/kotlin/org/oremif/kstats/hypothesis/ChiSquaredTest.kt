package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.distributions.ChiSquaredDistribution

/**
 * Chi-squared goodness-of-fit test.
 * If [expected] is null, assumes uniform distribution.
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
 * Chi-squared test of independence for a contingency table.
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
