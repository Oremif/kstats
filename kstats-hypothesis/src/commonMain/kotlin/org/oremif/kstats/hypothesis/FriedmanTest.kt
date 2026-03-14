package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.distributions.ChiSquaredDistribution
import org.oremif.kstats.sampling.TieMethod
import org.oremif.kstats.sampling.rank

/**
 * Performs the Friedman test for differences among repeated measures.
 *
 * The Friedman test is a non-parametric alternative to one-way repeated measures ANOVA.
 * It tests whether k related treatments (measured on the same n subjects or blocks)
 * have identical effects. The test ranks the treatment values within each block using
 * average tie-breaking, sums the ranks per treatment, and computes a chi-squared
 * statistic from the rank sums. Under the null hypothesis of no treatment effect,
 * the statistic follows a chi-squared distribution with k - 1 degrees of freedom.
 *
 * ### Example:
 * ```kotlin
 * val treatment1 = doubleArrayOf(7.0, 9.8, 6.5, 7.2, 8.3)
 * val treatment2 = doubleArrayOf(5.4, 6.8, 5.0, 4.8, 6.1)
 * val treatment3 = doubleArrayOf(8.2, 10.5, 7.1, 8.5, 9.0)
 * val result = friedmanTest(treatment1, treatment2, treatment3)
 * result.statistic        // chi-squared test statistic Q
 * result.pValue           // p-value from chi-squared distribution
 * result.degreesOfFreedom // k - 1
 * ```
 *
 * @param groups two or more treatment groups, each with the same number of observations (one per block).
 * @return a [TestResult] containing the chi-squared statistic, p-value, degrees of freedom (k - 1),
 * and additional info with "numGroups" and "numBlocks".
 */
public fun friedmanTest(vararg groups: DoubleArray): TestResult {
    if (groups.size < 2) throw InsufficientDataException(
        "Friedman test requires at least 2 groups, got ${groups.size}"
    )

    val n = groups[0].size
    for (i in 1 until groups.size) {
        if (groups[i].size != n) throw InvalidParameterException(
            "All groups must have the same size, group 0 has $n but group $i has ${groups[i].size}"
        )
    }

    if (n < 2) throw InsufficientDataException(
        "Each group must have at least 2 elements, got $n"
    )

    val k = groups.size
    val df = k - 1

    // Non-finite check: NaN or Infinity in any group
    for (group in groups) {
        if (group.any { !it.isFinite() }) {
            return TestResult(
                testName = "Friedman Test",
                statistic = Double.NaN,
                pValue = Double.NaN,
                degreesOfFreedom = df.toDouble(),
                additionalInfo = mapOf("numGroups" to k.toDouble(), "numBlocks" to n.toDouble())
            )
        }
    }

    // Step 1: Rank within each block
    val rankSums = DoubleArray(k)
    for (i in 0 until n) {
        val blockValues = DoubleArray(k) { j -> groups[j][i] }
        val blockRanks = blockValues.rank(TieMethod.AVERAGE)
        for (j in 0 until k) {
            rankSums[j] += blockRanks[j]
        }
    }

    // Degenerate check: all rank sums equal → no treatment effect
    val allEqual = rankSums.all { it == rankSums[0] }
    if (allEqual) {
        return TestResult(
            testName = "Friedman Test",
            statistic = 0.0,
            pValue = 1.0,
            degreesOfFreedom = df.toDouble(),
            additionalInfo = mapOf("numGroups" to k.toDouble(), "numBlocks" to n.toDouble())
        )
    }

    // Step 2: Compute Q statistic
    var sumRjSquared = 0.0
    for (j in 0 until k) {
        sumRjSquared += rankSums[j] * rankSums[j]
    }
    val q = (12.0 / (n * k * (k + 1))) * sumRjSquared - 3.0 * n * (k + 1)

    // p-value from chi-squared distribution with k-1 degrees of freedom
    val pValue = ChiSquaredDistribution(df.toDouble()).sf(q)

    return TestResult(
        testName = "Friedman Test",
        statistic = q,
        pValue = pValue.coerceIn(0.0, 1.0),
        degreesOfFreedom = df.toDouble(),
        additionalInfo = mapOf("numGroups" to k.toDouble(), "numBlocks" to n.toDouble())
    )
}
