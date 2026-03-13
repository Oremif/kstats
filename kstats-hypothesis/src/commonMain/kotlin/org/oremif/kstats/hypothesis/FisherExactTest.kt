package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.lnFactorial
import kotlin.math.exp

/**
 * Fisher's exact test for 2×2 contingency tables.
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

private fun hypergeometricLogPmf(k: Int, n1: Int, n2: Int, n: Int): Double {
    return lnFactorial(n1) + lnFactorial(n - n1) + lnFactorial(n2) + lnFactorial(n - n2) -
        lnFactorial(n) - lnFactorial(k) - lnFactorial(n1 - k) -
        lnFactorial(n2 - k) - lnFactorial(n - n1 - n2 + k)
}
