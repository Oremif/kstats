package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.lnFactorial
import kotlin.math.exp

/**
 * Performs Fisher's exact test for a 2x2 contingency table.
 *
 * The null hypothesis is that the row and column variables are independent.
 * Unlike the chi-squared test, Fisher's exact test computes the exact p-value using
 * the hypergeometric distribution, making it appropriate for small sample sizes where
 * the chi-squared approximation may be unreliable. The test statistic reported is the
 * odds ratio.
 *
 * ### Example:
 * ```kotlin
 * val table = arrayOf(
 *     intArrayOf(10, 5),
 *     intArrayOf(3, 12)
 * )
 * val result = fisherExactTest(table)
 * result.statistic                   // odds ratio (8.0)
 * result.pValue                      // two-sided exact p-value
 * result.additionalInfo["oddsRatio"] // same as statistic
 * ```
 *
 * @param table a 2x2 contingency table with non-negative integer counts.
 * @param alternative the direction of the alternative hypothesis. Defaults to [Alternative.TWO_SIDED].
 * [Alternative.LESS] tests if the odds ratio is less than 1, [Alternative.GREATER] tests
 * if it is greater than 1.
 * @return a [TestResult] containing the odds ratio as the statistic, the exact p-value,
 * and additional info with the "oddsRatio" entry.
 */
public fun fisherExactTest(
    table: Array<IntArray>,
    alternative: Alternative = Alternative.TWO_SIDED
): TestResult {
    if (!(table.size == 2 && table.all { it.size == 2 })) throw InvalidParameterException("Table must be 2×2")
    if (!table.all { row -> row.all { it >= 0 } }) throw InvalidParameterException("All values must be non-negative")

    val a = table[0][0]
    val b = table[0][1]
    val c = table[1][0]
    val d = table[1][1]
    val n = a + b + c + d

    val logPObserved = hypergeometricLogPmf(a, a + b, a + c, n)

    val minA = maxOf(0, (a + b) + (a + c) - n)
    val maxA = minOf(a + b, a + c)

    val pValue = when (alternative) {
        Alternative.TWO_SIDED -> {
            val pObserved = exp(logPObserved)
            var p = 0.0
            for (i in minA..maxA) {
                val pi = exp(hypergeometricLogPmf(i, a + b, a + c, n))
                if (pi <= pObserved * (1.0 + 1e-7)) {
                    p += pi
                }
            }
            p
        }

        Alternative.LESS -> {
            var p = 0.0
            for (i in minA..a) {
                p += exp(hypergeometricLogPmf(i, a + b, a + c, n))
            }
            p
        }

        Alternative.GREATER -> {
            var p = 0.0
            for (i in a..maxA) {
                p += exp(hypergeometricLogPmf(i, a + b, a + c, n))
            }
            p
        }
    }

    // Odds ratio
    val oddsRatio = if (b > 0 && c > 0) (a.toDouble() * d) / (b.toDouble() * c) else Double.POSITIVE_INFINITY

    return TestResult(
        testName = "Fisher's Exact Test",
        statistic = oddsRatio,
        pValue = pValue.coerceIn(0.0, 1.0),
        alternative = alternative,
        additionalInfo = mapOf("oddsRatio" to oddsRatio)
    )
}

/**
 * Log-probability mass function for the hypergeometric distribution.
 *
 * Computes in log-space to avoid overflow with large factorials.
 *
 * @param k number of successes drawn
 * @param rowSum total count in the first row (a + b)
 * @param colSum total count in the first column (a + c)
 * @param total grand total of the table (a + b + c + d)
 */
private fun hypergeometricLogPmf(k: Int, rowSum: Int, colSum: Int, total: Int): Double {
    return lnFactorial(rowSum) + lnFactorial(total - rowSum) +
        lnFactorial(colSum) + lnFactorial(total - colSum) -
        lnFactorial(total) - lnFactorial(k) - lnFactorial(rowSum - k) -
        lnFactorial(colSum - k) - lnFactorial(total - rowSum - colSum + k)
}
