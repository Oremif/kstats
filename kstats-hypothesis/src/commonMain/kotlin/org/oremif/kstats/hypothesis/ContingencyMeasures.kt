package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.distributions.NormalDistribution
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

private val standardNormal = NormalDistribution(0.0, 1.0)

/**
 * The result of a risk or association measure computed from a 2×2 contingency table.
 *
 * Contains the point estimate (e.g. odds ratio or relative risk) together with a
 * confidence interval computed at the requested confidence level.
 *
 * ### Example:
 * ```kotlin
 * val table = arrayOf(intArrayOf(10, 20), intArrayOf(30, 40))
 * val result = oddsRatio(table)
 * result.estimate         // 0.6667 (point estimate)
 * result.ci               // (0.272, 1.632) Woolf 95% CI
 * result.confidenceLevel  // 0.95
 * ```
 *
 * @property estimate the point estimate of the association measure. May be `0.0`,
 * [Double.POSITIVE_INFINITY], or [Double.NaN] when the contingency table contains zero cells.
 * @property ci the confidence interval as a (lower, upper) pair. Both bounds are [Double.NaN]
 * when the interval cannot be computed (e.g. when a cell is zero or the confidence level is NaN).
 * @property confidenceLevel the confidence level at which [ci] was computed.
 */
public data class RiskEstimate(
    val estimate: Double,
    val ci: Pair<Double, Double>,
    val confidenceLevel: Double,
)

/**
 * Computes the odds ratio with a Woolf logit confidence interval for a 2×2 contingency table.
 *
 * The odds ratio measures the strength of association between two binary variables. A value
 * of 1.0 means no association, values above 1.0 indicate a positive association (exposure
 * increases the odds of the outcome), and values below 1.0 indicate a negative association.
 * The Woolf confidence interval is constructed on the log scale and exponentiated back, which
 * ensures the interval is always positive and handles skewed sampling distributions well.
 *
 * For a table `[[a, b], [c, d]]`, the odds ratio is `(a * d) / (b * c)`. When any cell
 * is zero, the point estimate follows IEEE 754 arithmetic (zero, infinity, or NaN) and
 * the confidence interval is `(NaN, NaN)` because the Woolf method requires all cells
 * to be positive.
 *
 * ### Example:
 * ```kotlin
 * val table = arrayOf(intArrayOf(10, 5), intArrayOf(3, 12))
 * val result = oddsRatio(table, confidenceLevel = 0.95)
 * result.estimate // 8.0
 * result.ci       // (1.522, 42.042) Woolf 95% CI
 * ```
 *
 * @param table a 2×2 contingency table with non-negative integer counts.
 * @param confidenceLevel the confidence level for the Woolf interval. Must be in `(0, 1)`.
 * Defaults to `0.95` (95%).
 * @return a [RiskEstimate] containing the odds ratio and its Woolf logit confidence interval.
 * @see relativeRisk for the ratio of proportions rather than odds.
 * @see fisherExactTest for a hypothesis test that also reports an odds ratio.
 */
public fun oddsRatio(
    table: Array<IntArray>,
    confidenceLevel: Double = 0.95,
): RiskEstimate {
    if (!(table.size == 2 && table.all { it.size == 2 })) {
        throw InvalidParameterException("Table must be 2\u00d72")
    }
    if (!table.all { row -> row.all { it >= 0 } }) {
        throw InvalidParameterException("All values must be non-negative")
    }
    if (confidenceLevel <= 0.0 || confidenceLevel >= 1.0) {
        throw InvalidParameterException("confidenceLevel must be in (0, 1), got $confidenceLevel")
    }

    val a = table[0][0].toDouble()
    val b = table[0][1].toDouble()
    val c = table[1][0].toDouble()
    val d = table[1][1].toDouble()

    // Point estimate: OR = (a*d) / (b*c)
    // IEEE 754 handles edge cases: 0/0 -> NaN, x/0 -> Inf
    val or = (a * d) / (b * c)

    // Woolf logit CI: exp(log(OR) +/- z * sqrt(1/a + 1/b + 1/c + 1/d))
    // Requires all cells > 0 for log(OR) and SE to be finite
    val alpha = 1.0 - confidenceLevel
    val ci = if (alpha.isNaN()) {
        Pair(Double.NaN, Double.NaN)
    } else if (a > 0.0 && b > 0.0 && c > 0.0 && d > 0.0) {
        val logOr = ln(or)
        val se = sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
        val z = standardNormal.quantile(1.0 - alpha / 2.0)
        Pair(exp(logOr - z * se), exp(logOr + z * se))
    } else {
        Pair(Double.NaN, Double.NaN)
    }

    return RiskEstimate(estimate = or, ci = ci, confidenceLevel = confidenceLevel)
}

/**
 * Computes the relative risk with a log-based confidence interval for a 2×2 contingency table.
 *
 * The relative risk (also called the risk ratio) compares the probability of the outcome in
 * the exposed group to the probability in the unexposed group. A value of 1.0 means equal
 * risk, values above 1.0 indicate higher risk in the first (exposed) group, and values
 * below 1.0 indicate lower risk. Unlike the odds ratio, the relative risk has a direct
 * probabilistic interpretation and is preferred when the outcome is common.
 *
 * For a table `[[a, b], [c, d]]`, the relative risk is `(a / (a + b)) / (c / (c + d))`.
 * The confidence interval is constructed on the log scale using the standard error of
 * `log(RR)` and exponentiated back. When the first cell in either row is zero, the
 * confidence interval is `(NaN, NaN)` because the log transform is undefined.
 *
 * ### Example:
 * ```kotlin
 * val table = arrayOf(intArrayOf(10, 20), intArrayOf(30, 40))
 * val result = relativeRisk(table, confidenceLevel = 0.95)
 * result.estimate // 0.7778 (exposed group has lower risk)
 * result.ci       // (0.438, 1.381) log-based 95% CI
 * ```
 *
 * @param table a 2×2 contingency table with non-negative integer counts.
 * @param confidenceLevel the confidence level for the log-based interval. Must be in `(0, 1)`.
 * Defaults to `0.95` (95%).
 * @return a [RiskEstimate] containing the relative risk and its log-based confidence interval.
 * @see oddsRatio for the ratio of odds rather than probabilities.
 */
public fun relativeRisk(
    table: Array<IntArray>,
    confidenceLevel: Double = 0.95,
): RiskEstimate {
    if (!(table.size == 2 && table.all { it.size == 2 })) {
        throw InvalidParameterException("Table must be 2\u00d72")
    }
    if (!table.all { row -> row.all { it >= 0 } }) {
        throw InvalidParameterException("All values must be non-negative")
    }
    if (confidenceLevel <= 0.0 || confidenceLevel >= 1.0) {
        throw InvalidParameterException("confidenceLevel must be in (0, 1), got $confidenceLevel")
    }

    val a = table[0][0].toDouble()
    val b = table[0][1].toDouble()
    val c = table[1][0].toDouble()
    val d = table[1][1].toDouble()

    val row1 = a + b
    val row2 = c + d

    // Point estimate: RR = (a/(a+b)) / (c/(c+d))
    // IEEE 754 handles edge cases: 0/0 -> NaN, x/0 -> Inf
    val rr = (a / row1) / (c / row2)

    // Log-based CI: exp(log(RR) +/- z * sqrt(b/(a*(a+b)) + d/(c*(c+d))))
    // Requires a > 0 and c > 0 for log(RR) and SE to be finite
    val alpha = 1.0 - confidenceLevel
    val ci = if (alpha.isNaN()) {
        Pair(Double.NaN, Double.NaN)
    } else if (a > 0.0 && c > 0.0 && row1 > 0.0 && row2 > 0.0) {
        val logRr = ln(rr)
        val se = sqrt(b / (a * row1) + d / (c * row2))
        val z = standardNormal.quantile(1.0 - alpha / 2.0)
        Pair(exp(logRr - z * se), exp(logRr + z * se))
    } else {
        Pair(Double.NaN, Double.NaN)
    }

    return RiskEstimate(estimate = rr, ci = ci, confidenceLevel = confidenceLevel)
}
